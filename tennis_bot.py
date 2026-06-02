#!/usr/bin/env python3
"""
Tennis Bot v3.2 — ATP/WTA 巡迴賽預測系統
9因子模型：Surface ELO 25% + Markov Chain 25% + Hold/Break 20% + Advanced Stats 30%
附加調整：體能(年齡加權) ±10% | 場地狀態 ±5% | H2H ±5% | 搶七/關鍵分 ±7%
         雙誤懲罰 ±4% | 左手剋制 ±3% | 反拍剋制 ±2% | 室內場速(進入發球模型)
資料來源：Jeff Sackmann ATP/WTA CSVs + The Odds API
"""