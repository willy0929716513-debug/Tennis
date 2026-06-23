#!/usr/bin/env python3
"""
Tennis Bot v4.0 — Walk-Forward Out-of-Sample Backtest
======================================================
方法：
  訓練集 = 所有比賽中，早於 LOOKBACK_MONTHS 個月前的資料
  測試集 = 最近 LOOKBACK_MONTHS 個月（已知結果的比賽）

  對每場測試比賽，以「實際勝者 = p1」呼叫 predict()，
  評估模型能否給出 blend_p1 > 0.5（正確預測方向）。

輸出指標：
  - Accuracy       : 預測方向正確率（隨機基準 50%）
  - Brier Score    : 概率誤差平方均值（隨機基準 0.25，越低越好）
  - Log-Loss       : 信息論損失（越低越好）
  - Calibration    : 各信心區間的實際勝率 vs 預測概率
  - Surface 分項   : 各球場類型的準確率
  - 儲存           : docs/backtest_results.json
"""

import csv
import datetime
import io
import json
import logging
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import requests

# ── 環境設定 ─────────────────────────────────────────────────────────────────
log = logging.getLogger("backtest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

LOOKBACK_MONTHS   = 6    # 測試窗口（月）
MIN_PROFILE_N     = 3    # 兩位球員至少需要 N 場訓練資料
RESULTS_PATH      = "docs/backtest_results.json"

# ── 載入 tennis_bot 模組 ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tennis_bot as bot


# ── 輔助函式 ─────────────────────────────────────────────────────────────────

def _parse_date(td: str) -> Optional[datetime.datetime]:
    if len(td) == 8:
        try:
            return datetime.datetime(int(td[:4]), int(td[4:6]), int(td[6:8]))
        except ValueError:
            pass
    return None


def _brier_score(probs: List[float]) -> float:
    """Brier score when actual outcome = 1.0 (winner is always p1)."""
    return sum((p - 1.0) ** 2 for p in probs) / len(probs)


def _log_loss(probs: List[float]) -> float:
    """Log-loss when actual outcome = 1.0."""
    eps = 1e-9
    return -sum(math.log(max(p, eps)) for p in probs) / len(probs)


def _calibration(probs: List[float], bin_size: float = 0.05) -> List[dict]:
    """
    Group predictions into bins and show predicted vs. actual win rate.
    Since winner is always p1, actual WR in each bin should equal bin center.
    """
    bins: Dict[int, List[float]] = {}
    for p in probs:
        b = int(p / bin_size)
        bins.setdefault(b, []).append(p)
    result = []
    for b in sorted(bins):
        lo = b * bin_size
        hi = lo + bin_size
        vals = bins[b]
        avg_pred   = sum(vals) / len(vals)
        actual_wr  = sum(1 for v in vals if v > 0.5) / len(vals)  # correct direction
        result.append({
            "range":       f"{lo:.2f}–{hi:.2f}",
            "count":       len(vals),
            "avg_pred":    round(avg_pred, 3),
            "actual_acc":  round(actual_wr * 100, 1),
        })
    return result


# ── 主要回測邏輯 ──────────────────────────────────────────────────────────────

def run_backtest() -> dict:
    now     = datetime.datetime.utcnow()
    cutoff  = now - datetime.timedelta(days=LOOKBACK_MONTHS * 30)

    # 1. 下載資料（tennis_bot 已有此函式）
    log.info("Downloading 2 years of Sackmann match data …")
    all_matches = bot.fetch_sackmann_matches()
    if not all_matches:
        log.error("No match data — check network")
        return {}
    log.info("Total rows fetched: %d", len(all_matches))

    # 2. 分割訓練 / 測試
    train_matches, test_matches = [], []
    for row in all_matches:
        md = _parse_date(row.get("tourney_date", ""))
        if md is None:
            continue
        (train_matches if md < cutoff else test_matches).append(row)

    log.info("Train: %d matches (before %s)", len(train_matches), cutoff.strftime("%Y-%m-%d"))
    log.info("Test : %d matches (from %s to now)", len(test_matches), cutoff.strftime("%Y-%m-%d"))

    if len(test_matches) < 50:
        log.warning("Test set too small (%d matches) — extend LOOKBACK_MONTHS", len(test_matches))
        if not test_matches:
            return {}

    # 3. 用訓練集建立球員檔案（臨時替換全域快取）
    log.info("Building player profiles from training data only …")
    _saved = {
        "profiles": dict(bot._SACKMANN_PROFILES),
        "elo":      dict(bot._LIVE_ELO),
        "dh2h":     dict(bot._DYNAMIC_H2H),
        "dh2h_ov":  dict(bot._DYNAMIC_H2H_OVERALL),
    }
    bot._SACKMANN_PROFILES.clear()
    bot._LIVE_ELO.clear()
    bot._DYNAMIC_H2H.clear()
    bot._DYNAMIC_H2H_OVERALL.clear()

    bot.load_sackmann_data(train_matches)
    log.info("Profiles built: %d players", len(bot._SACKMANN_PROFILES))

    # 4. 逐場預測
    results      = []
    skipped      = 0
    error_count  = 0

    for row in test_matches:
        wname = (row.get("winner_name") or "").strip()
        lname = (row.get("loser_name") or "").strip()
        if not wname or not lname:
            skipped += 1
            continue

        # 排除退賽 / 棄賽
        score = (row.get("score") or "").upper()
        if "RET" in score or "W/O" in score:
            skipped += 1
            continue

        wkey = bot.norm_player(wname.lower())
        lkey = bot.norm_player(lname.lower())

        surf_raw = (row.get("surface") or "hard").lower()
        surface  = surf_raw if surf_raw in ("hard", "clay", "grass") else "hard"

        lvl_raw  = (row.get("tourney_level") or "").upper()
        best_of  = 5 if lvl_raw == "G" else 3

        is_wta = "W" in lvl_raw or (row.get("tourney_id", "").startswith("W"))

        # 跳過資料不足的球員
        p1_n = bot._SACKMANN_PROFILES.get(wkey, {}).get("n_matches", 0)
        p2_n = bot._SACKMANN_PROFILES.get(lkey, {}).get("n_matches", 0)
        if p1_n < MIN_PROFILE_N or p2_n < MIN_PROFILE_N:
            skipped += 1
            continue

        try:
            sport_key = "tennis_wta" if is_wta else "tennis_atp"
            pred = bot.predict(
                wkey, lkey, surface,
                best_of=best_of, sport_key=sport_key,
            )
            pred_p_winner = pred["blend_p1"]   # winner is always p1

            # 從訓練資料推估市場概率 (ELO-based fair price proxy)
            elo_w = pred.get("elo1", 1800)
            elo_l = pred.get("elo2", 1800)
            elo_p = bot.elo_win_prob(elo_w, elo_l)

            results.append({
                "date":    row.get("tourney_date", ""),
                "surface": surface,
                "winner":  wname,
                "loser":   lname,
                "best_of": best_of,
                "tour_lvl": lvl_raw,
                "pred_p":  round(pred_p_winner, 4),
                "elo_p":   round(elo_p, 4),
                "elo_w":   round(elo_w, 1),
                "elo_l":   round(elo_l, 1),
                "correct": pred_p_winner > 0.5,
            })
        except Exception as e:
            error_count += 1
            log.debug("Predict error %s vs %s: %s", wkey, lkey, e)

    # 5. 還原全域快取
    bot._SACKMANN_PROFILES.clear();     bot._SACKMANN_PROFILES.update(_saved["profiles"])
    bot._LIVE_ELO.clear();              bot._LIVE_ELO.update(_saved["elo"])
    bot._DYNAMIC_H2H.clear();           bot._DYNAMIC_H2H.update(_saved["dh2h"])
    bot._DYNAMIC_H2H_OVERALL.clear();   bot._DYNAMIC_H2H_OVERALL.update(_saved["dh2h_ov"])

    log.info("Predictions done: %d valid, %d skipped, %d errors",
             len(results), skipped, error_count)

    if not results:
        log.error("No predictions generated")
        return {}

    # 6. 計算指標
    probs    = [r["pred_p"] for r in results]
    elo_probs= [r["elo_p"]  for r in results]
    n        = len(results)
    correct  = sum(1 for r in results if r["correct"])

    accuracy        = correct / n
    model_brier     = _brier_score(probs)
    model_logloss   = _log_loss(probs)
    elo_brier       = _brier_score(elo_probs)
    random_brier    = 0.25
    improvement_pct = round((random_brier - model_brier) / random_brier * 100, 1)

    # 各球場分項
    surf_stats: Dict[str, Dict] = {}
    for r in results:
        s = r["surface"]
        if s not in surf_stats:
            surf_stats[s] = {"total": 0, "correct": 0}
        surf_stats[s]["total"] += 1
        if r["correct"]:
            surf_stats[s]["correct"] += 1

    surface_breakdown = {
        s: {
            "accuracy": round(v["correct"] / v["total"] * 100, 1),
            "count":    v["total"],
        }
        for s, v in surf_stats.items()
    }

    # 信心區間
    bands = [
        ("50–55%", 0.50, 0.55),
        ("55–60%", 0.55, 0.60),
        ("60–65%", 0.60, 0.65),
        ("65–70%", 0.65, 0.70),
        ("70–80%", 0.70, 0.80),
        ("80–90%", 0.80, 0.90),
        ("90%+",   0.90, 1.01),
    ]
    confidence_bands = []
    for label, lo, hi in bands:
        bucket = [r for r in results if lo <= r["pred_p"] < hi]
        if bucket:
            acc = sum(1 for r in bucket if r["correct"]) / len(bucket)
            confidence_bands.append({
                "band":     label,
                "count":    len(bucket),
                "accuracy": round(acc * 100, 1),
            })

    # BO5 vs BO3
    bo5 = [r for r in results if r["best_of"] == 5]
    bo3 = [r for r in results if r["best_of"] == 3]

    # 校準分析
    calib = _calibration(probs)

    # 7. 組合最終結果
    metrics = {
        "generated_at":  now.strftime("%Y-%m-%d %H:%M UTC"),
        "model_version": "v4.0",
        "test_period": {
            "from": cutoff.strftime("%Y-%m-%d"),
            "to":   now.strftime("%Y-%m-%d"),
        },
        "sample_size": {
            "train_matches": len(train_matches),
            "test_matches":  n,
            "skipped":       skipped,
            "errors":        error_count,
        },
        "overall": {
            "accuracy":          round(accuracy * 100, 1),
            "brier_score":       round(model_brier, 4),
            "log_loss":          round(model_logloss, 4),
            "elo_baseline_brier": round(elo_brier, 4),
            "random_brier":      round(random_brier, 4),
            "brier_improvement_vs_random": f"{improvement_pct}%",
        },
        "by_surface":    surface_breakdown,
        "by_confidence": confidence_bands,
        "by_format": {
            "bo5_accuracy": round(sum(1 for r in bo5 if r["correct"]) / max(len(bo5), 1) * 100, 1),
            "bo5_count":    len(bo5),
            "bo3_accuracy": round(sum(1 for r in bo3 if r["correct"]) / max(len(bo3), 1) * 100, 1),
            "bo3_count":    len(bo3),
        },
        "calibration_bins": calib,
        "benchmarks": {
            "random_accuracy":    "50.0%",
            "elo_only_accuracy":  round(sum(1 for r in results if r["elo_p"] > 0.5) / n * 100, 1),
            "pass_threshold":     "Accuracy ≥ 58%, Brier ≤ 0.22",
        },
    }

    # 判定是否通過
    passed = accuracy >= 0.58 and model_brier <= 0.22
    metrics["verdict"] = "PASS ✅" if passed else "NEEDS REVIEW ⚠️"

    return metrics


# ── 輸出與儲存 ────────────────────────────────────────────────────────────────

def _print_report(m: dict) -> None:
    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  TENNIS BOT v4.0 — BACKTEST REPORT")
    print(SEP)
    tp = m["test_period"]
    ss = m["sample_size"]
    ov = m["overall"]
    bm = m["benchmarks"]
    print(f"  Period : {tp['from']}  →  {tp['to']}")
    print(f"  Matches: {ss['test_matches']} tested | {ss['train_matches']} training | {ss['skipped']} skipped")
    print()
    print(f"  {'Metric':<30}  {'Model':>8}  {'Baseline':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}")
    print(f"  {'Accuracy':<30}  {ov['accuracy']:>7.1f}%  {bm['random_accuracy']:>8}")
    print(f"  {'ELO-only Accuracy':<30}  {bm['elo_only_accuracy']:>7.1f}%  {'50.0%':>8}")
    print(f"  {'Brier Score (↓ better)':<30}  {ov['brier_score']:>8.4f}  {ov['random_brier']:>8.4f}")
    print(f"  {'Log-Loss (↓ better)':<30}  {ov['log_loss']:>8.4f}  {'0.6931':>8}")
    print(f"  {'vs-Random Brier Improve':<30}  {ov['brier_improvement_vs_random']:>8}")
    print()
    print("  By Surface:")
    for surf, s in m["by_surface"].items():
        print(f"    {surf:<8}  {s['accuracy']:5.1f}%  ({s['count']} matches)")
    print()
    print("  By Model Confidence — higher conf should yield higher accuracy:")
    for b in m["by_confidence"]:
        bar_len = int(b["accuracy"] / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"    {b['band']:<8}  {b['accuracy']:5.1f}%  n={b['count']:4d}  {bar[:30]}")
    print()
    bf = m["by_format"]
    print(f"  BO3:  {bf['bo3_accuracy']}%  ({bf['bo3_count']} matches)")
    print(f"  BO5:  {bf['bo5_accuracy']}%  ({bf['bo5_count']} matches)")
    print()
    print(f"  VERDICT:  {m['verdict']}")
    print(f"  Pass criterion: {bm['pass_threshold']}")
    print(SEP)


if __name__ == "__main__":
    log.info("Tennis Bot v4.0 — Backtest starting")

    metrics = run_backtest()

    if not metrics:
        print("\n[ERROR] Backtest produced no results — check logs above")
        sys.exit(1)

    _print_report(metrics)

    # Save JSON
    os.makedirs("docs", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info("Saved → %s", RESULTS_PATH)

    # Exit code: 0 = pass, 1 = fail
    sys.exit(0 if "PASS" in metrics.get("verdict", "") else 1)
