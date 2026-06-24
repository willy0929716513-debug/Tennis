#!/usr/bin/env python3
"""
Tennis Bot v4.0 — ATP/WTA 巡迴賽預測系統
核心模型：Surface ELO(近期加權) 25% + Markov Chain 25% + Hold/Break 20% + Advanced Stats 30%
調整因子：體能(年齡加權) | 球場速度 | H2H(動態) | 搶七/關鍵分 | 雙誤 | 左手 | 反拍
          高度 | 風速 | 連勝 | BO5 | 發球 | 破發 | 體能負荷 | 風格 | 場地過渡 | 後場 | 傷病
改進亮點：Power Devig | 動態H2H | 指數衰退Form | 近期加權ELO | 概率校正
資料來源：Jeff Sackmann ATP/WTA CSVs (雙年) + The Odds API
"""

import csv
import datetime
import io
import json
import logging
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import requests

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("tennis_bot")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
NTFY_TOPIC    = os.environ.get("NTFY_TOPIC", "tennis-picks")
DISCORD_HOOK  = os.environ.get("DISCORD_WEBHOOK", "")
GIST_TOKEN    = os.environ.get("GIST_TOKEN", "")
GIST_ID       = os.environ.get("GIST_ID", "")

JSON_PATH     = "docs/picks_latest.json"

KELLY         = 0.25
KELLY_MAX     = 200.0
KELLY_FLOOR   = 50.0
BANKROLL      = 1000.0
MAX_DAILY_EXP = 500.0

MIN_EDGE_ML   = 0.06
MIN_CONF_ML   = 0.60
MIN_BOOKS     = 3
MAX_PICKS     = 6

# v4.0 model improvements
PROB_CALIB_ALPHA  = 0.88   # regress extreme probs toward 0.5 (over-confidence correction)
FORM_DECAY_LAMBDA = 0.12   # exponential form decay per match position
ELO_RECENT_MULT   = {      # K-factor recency multiplier by days ago
    180: 1.00,  # last 6 months: full weight
    540: 0.70,  # 6–18 months
}
ELO_RECENCY_FLOOR = 0.45   # older than 18 months

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED MODEL CONSTANTS  (v3)
# ─────────────────────────────────────────────────────────────────────────────
AGE_FATIGUE_SCALE = 0.06    # extra fatigue multiplier per year above 28
LEFTY_SERVE_BONUS = 0.012   # serve point adj for lefty serving vs righty
LEFTY_GRASS_EXTRA = 0.006   # additional grass lefty bonus
BH_TOPSPIN_VULN   = 0.008   # 1h backhand vulnerability vs lefty topspin (clay)

INDOOR_TOURNAMENTS = {
    "paris", "rotterdam", "vienna", "sofia", "marseille",
    "montpellier", "dallas", "memphis", "zhuhai", "moscow",
    "basel", "cologne", "st_petersburg", "astana", "nur-sultan",
    "bercy", "indoor",
}

COURT_SPEED_ADJ: Dict[str, float] = {
    # Indoor hard (fast)
    "paris":            +0.018,
    "rotterdam":        +0.020,
    "vienna":           +0.018,
    "sofia":            +0.016,
    "marseille":        +0.016,
    "dallas":           +0.014,
    # Outdoor hard variations
    "us_open":          +0.010,
    "australian_open":  +0.008,
    "miami":            +0.005,
    "indian_wells":     +0.005,
    # Slow clay
    "monte_carlo":      -0.005,
    "hamburg":          -0.003,
    # Fast grass
    "halle":            +0.006,
    "queens":           +0.006,
    "eastbourne":       +0.004,
}

# ─────────────────────────────────────────────────────────────────────────────
# ALTITUDE  (metres above sea level for tournament cities)
# ─────────────────────────────────────────────────────────────────────────────
ALTITUDE_M: Dict[str, int] = {
    "buenos_aires": 1138, "bogota": 2600, "quito": 2850, "lima": 154,
    "santiago": 520, "mexico_city": 2250, "guadalajara": 1566,
    "madrid": 667, "kitzbuhel": 762, "gstaad": 1060,
    "granada": 685, "lyon": 173, "chengdu": 506, "kunming": 1895,
}

# ─────────────────────────────────────────────────────────────────────────────
# TOURNAMENT COORDINATES  (lat, lon) for weather fetch
# ─────────────────────────────────────────────────────────────────────────────
TOURNAMENT_COORDS: Dict[str, Tuple[float, float]] = {
    "roland_garros":    (48.847,   2.250),
    "french_open":      (48.847,   2.250),
    "wimbledon":        (51.434,  -0.214),
    "us_open":          (40.750, -73.846),
    "australian_open":  (-37.821, 144.981),
    "indian_wells":     (33.720, -116.369),
    "miami":            (25.683,  -80.180),
    "monte_carlo":      (43.745,   7.427),
    "madrid":           (40.416,  -3.703),
    "rome":             (41.897,  12.469),
    "barcelona":        (41.389,   2.165),
    "halle":            (51.932,   8.660),
    "queens":           (51.490,  -0.212),
    "eastbourne":       (50.768,   0.280),
    "hamburg":          (53.553,   9.992),
    "toronto":          (43.641,  -79.382),
    "cincinnati":       (39.104,  -84.510),
    "beijing":          (39.906, 116.391),
    "shanghai":         (31.225, 121.474),
    "kitzbuhel":        (47.444,  12.391),
    "buenos_aires":     (-34.614, -58.382),
    "rio":              (-22.906, -43.172),
    "bogota":           ( 4.711,  -74.072),
    "santiago":         (-33.437, -70.650),
    "umag":             (45.434,  13.524),
    "bastad":           (56.430,  12.854),
    "gstaad":           (46.474,   7.288),
}

ODDS_PREV_PATH = "docs/.odds_prev.json"

# ─────────────────────────────────────────────────────────────────────────────
# ATP PLAYER DATABASE
# svpt_won  : P(server wins a point when THIS player is serving)
# rtpt_won  : P(THIS player wins a return point vs any server)
# elo       : surface-specific Elo rating
# birth_year: for age-based fatigue multiplier
# backhand  : "1h" or "2h"
# ─────────────────────────────────────────────────────────────────────────────
ATP_STATS: Dict[str, dict] = {
    "djokovic": {
        "full_name": "Novak Djokovic", "hand": "R", "rank": 2, "country": "SRB",
        "birth_year": 1987, "backhand": "2h",
        "hard":  {"svpt_won": 0.663, "rtpt_won": 0.388, "elo": 2375},
        "clay":  {"svpt_won": 0.652, "rtpt_won": 0.392, "elo": 2420},
        "grass": {"svpt_won": 0.671, "rtpt_won": 0.385, "elo": 2355},
    },
    "alcaraz": {
        "full_name": "Carlos Alcaraz", "hand": "R", "rank": 1, "country": "ESP",
        "birth_year": 2003, "backhand": "2h",
        "hard":  {"svpt_won": 0.658, "rtpt_won": 0.382, "elo": 2300},
        "clay":  {"svpt_won": 0.660, "rtpt_won": 0.390, "elo": 2340},
        "grass": {"svpt_won": 0.670, "rtpt_won": 0.378, "elo": 2285},
    },
    "sinner": {
        "full_name": "Jannik Sinner", "hand": "R", "rank": 1, "country": "ITA",
        "birth_year": 2001, "backhand": "2h",
        "hard":  {"svpt_won": 0.665, "rtpt_won": 0.383, "elo": 2310},
        "clay":  {"svpt_won": 0.655, "rtpt_won": 0.375, "elo": 2265},
        "grass": {"svpt_won": 0.668, "rtpt_won": 0.372, "elo": 2250},
    },
    "medvedev": {
        "full_name": "Daniil Medvedev", "hand": "R", "rank": 5, "country": "RUS",
        "birth_year": 1996, "backhand": "2h",
        "hard":  {"svpt_won": 0.662, "rtpt_won": 0.375, "elo": 2240},
        "clay":  {"svpt_won": 0.638, "rtpt_won": 0.345, "elo": 2085},
        "grass": {"svpt_won": 0.660, "rtpt_won": 0.355, "elo": 2145},
    },
    "zverev": {
        "full_name": "Alexander Zverev", "hand": "R", "rank": 3, "country": "GER",
        "birth_year": 1997, "backhand": "2h",
        "hard":  {"svpt_won": 0.650, "rtpt_won": 0.360, "elo": 2200},
        "clay":  {"svpt_won": 0.648, "rtpt_won": 0.365, "elo": 2215},
        "grass": {"svpt_won": 0.655, "rtpt_won": 0.352, "elo": 2160},
    },
    "rublev": {
        "full_name": "Andrey Rublev", "hand": "R", "rank": 7, "country": "RUS",
        "birth_year": 1997, "backhand": "2h",
        "hard":  {"svpt_won": 0.635, "rtpt_won": 0.355, "elo": 2120},
        "clay":  {"svpt_won": 0.638, "rtpt_won": 0.360, "elo": 2140},
        "grass": {"svpt_won": 0.638, "rtpt_won": 0.345, "elo": 2080},
    },
    "tsitsipas": {
        "full_name": "Stefanos Tsitsipas", "hand": "R", "rank": 11, "country": "GRE",
        "birth_year": 1998, "backhand": "1h",
        "hard":  {"svpt_won": 0.638, "rtpt_won": 0.358, "elo": 2110},
        "clay":  {"svpt_won": 0.648, "rtpt_won": 0.370, "elo": 2175},
        "grass": {"svpt_won": 0.645, "rtpt_won": 0.348, "elo": 2065},
    },
    "fritz": {
        "full_name": "Taylor Fritz", "hand": "R", "rank": 4, "country": "USA",
        "birth_year": 1997, "backhand": "2h",
        "hard":  {"svpt_won": 0.660, "rtpt_won": 0.358, "elo": 2155},
        "clay":  {"svpt_won": 0.638, "rtpt_won": 0.338, "elo": 2020},
        "grass": {"svpt_won": 0.668, "rtpt_won": 0.355, "elo": 2120},
    },
    "de_minaur": {
        "full_name": "Alex de Minaur", "hand": "R", "rank": 9, "country": "AUS",
        "birth_year": 1999, "backhand": "2h",
        "hard":  {"svpt_won": 0.635, "rtpt_won": 0.368, "elo": 2100},
        "clay":  {"svpt_won": 0.628, "rtpt_won": 0.365, "elo": 2070},
        "grass": {"svpt_won": 0.640, "rtpt_won": 0.365, "elo": 2085},
    },
    "hurkacz": {
        "full_name": "Hubert Hurkacz", "hand": "R", "rank": 10, "country": "POL",
        "birth_year": 1997, "backhand": "2h",
        "hard":  {"svpt_won": 0.665, "rtpt_won": 0.348, "elo": 2095},
        "clay":  {"svpt_won": 0.638, "rtpt_won": 0.325, "elo": 1960},
        "grass": {"svpt_won": 0.678, "rtpt_won": 0.345, "elo": 2110},
    },
    "dimitrov": {
        "full_name": "Grigor Dimitrov", "hand": "R", "rank": 13, "country": "BUL",
        "birth_year": 1991, "backhand": "1h",
        "hard":  {"svpt_won": 0.645, "rtpt_won": 0.355, "elo": 2060},
        "clay":  {"svpt_won": 0.638, "rtpt_won": 0.348, "elo": 2020},
        "grass": {"svpt_won": 0.652, "rtpt_won": 0.352, "elo": 2045},
    },
    "paul": {
        "full_name": "Tommy Paul", "hand": "R", "rank": 12, "country": "USA",
        "birth_year": 1997, "backhand": "2h",
        "hard":  {"svpt_won": 0.640, "rtpt_won": 0.355, "elo": 2040},
        "clay":  {"svpt_won": 0.632, "rtpt_won": 0.345, "elo": 2005},
        "grass": {"svpt_won": 0.645, "rtpt_won": 0.348, "elo": 2025},
    },
    "auger_aliassime": {
        "full_name": "Felix Auger-Aliassime", "hand": "R", "rank": 20, "country": "CAN",
        "birth_year": 2000, "backhand": "2h",
        "hard":  {"svpt_won": 0.655, "rtpt_won": 0.352, "elo": 2035},
        "clay":  {"svpt_won": 0.638, "rtpt_won": 0.338, "elo": 1980},
        "grass": {"svpt_won": 0.662, "rtpt_won": 0.348, "elo": 2020},
    },
    "musetti": {
        "full_name": "Lorenzo Musetti", "hand": "L", "rank": 16, "country": "ITA",
        "birth_year": 2002, "backhand": "1h",
        "hard":  {"svpt_won": 0.625, "rtpt_won": 0.348, "elo": 2010},
        "clay":  {"svpt_won": 0.635, "rtpt_won": 0.358, "elo": 2055},
        "grass": {"svpt_won": 0.638, "rtpt_won": 0.348, "elo": 2035},
    },
    "tiafoe": {
        "full_name": "Frances Tiafoe", "hand": "R", "rank": 15, "country": "USA",
        "birth_year": 1998, "backhand": "2h",
        "hard":  {"svpt_won": 0.638, "rtpt_won": 0.352, "elo": 2025},
        "clay":  {"svpt_won": 0.620, "rtpt_won": 0.335, "elo": 1950},
        "grass": {"svpt_won": 0.648, "rtpt_won": 0.345, "elo": 1985},
    },
    "berrettini": {
        "full_name": "Matteo Berrettini", "hand": "R", "rank": 35, "country": "ITA",
        "birth_year": 1996, "backhand": "1h",
        "hard":  {"svpt_won": 0.658, "rtpt_won": 0.345, "elo": 2050},
        "clay":  {"svpt_won": 0.648, "rtpt_won": 0.342, "elo": 2015},
        "grass": {"svpt_won": 0.680, "rtpt_won": 0.345, "elo": 2085},
    },
    "ruud": {
        "full_name": "Casper Ruud", "hand": "R", "rank": 14, "country": "NOR",
        "birth_year": 1998, "backhand": "2h",
        "hard":  {"svpt_won": 0.630, "rtpt_won": 0.348, "elo": 2025},
        "clay":  {"svpt_won": 0.645, "rtpt_won": 0.362, "elo": 2095},
        "grass": {"svpt_won": 0.628, "rtpt_won": 0.332, "elo": 1945},
    },
    "draper": {
        "full_name": "Jack Draper", "hand": "L", "rank": 17, "country": "GBR",
        "birth_year": 2001, "backhand": "2h",
        "hard":  {"svpt_won": 0.648, "rtpt_won": 0.355, "elo": 2020},
        "clay":  {"svpt_won": 0.638, "rtpt_won": 0.348, "elo": 1985},
        "grass": {"svpt_won": 0.655, "rtpt_won": 0.352, "elo": 2030},
    },
    "shelton": {
        "full_name": "Ben Shelton", "hand": "L", "rank": 21, "country": "USA",
        "birth_year": 2002, "backhand": "2h",
        "hard":  {"svpt_won": 0.658, "rtpt_won": 0.348, "elo": 2000},
        "clay":  {"svpt_won": 0.628, "rtpt_won": 0.325, "elo": 1890},
        "grass": {"svpt_won": 0.668, "rtpt_won": 0.340, "elo": 1985},
    },
    "khachanov": {
        "full_name": "Karen Khachanov", "hand": "R", "rank": 22, "country": "RUS",
        "birth_year": 1996, "backhand": "2h",
        "hard":  {"svpt_won": 0.645, "rtpt_won": 0.345, "elo": 2000},
        "clay":  {"svpt_won": 0.635, "rtpt_won": 0.338, "elo": 1975},
        "grass": {"svpt_won": 0.650, "rtpt_won": 0.335, "elo": 1975},
    },
    "bublik": {
        "full_name": "Alexander Bublik", "hand": "R", "rank": 24, "country": "KAZ",
        "birth_year": 1997, "backhand": "2h",
        "hard":  {"svpt_won": 0.658, "rtpt_won": 0.328, "elo": 1955},
        "clay":  {"svpt_won": 0.635, "rtpt_won": 0.312, "elo": 1880},
        "grass": {"svpt_won": 0.668, "rtpt_won": 0.322, "elo": 1965},
    },
    "humbert": {
        "full_name": "Ugo Humbert", "hand": "L", "rank": 19, "country": "FRA",
        "birth_year": 1998, "backhand": "2h",
        "hard":  {"svpt_won": 0.648, "rtpt_won": 0.355, "elo": 2005},
        "clay":  {"svpt_won": 0.632, "rtpt_won": 0.342, "elo": 1945},
        "grass": {"svpt_won": 0.655, "rtpt_won": 0.348, "elo": 1990},
    },
    "jarry": {
        "full_name": "Nicolas Jarry", "hand": "R", "rank": 28, "country": "CHI",
        "birth_year": 1995, "backhand": "2h",
        "hard":  {"svpt_won": 0.648, "rtpt_won": 0.332, "elo": 1935},
        "clay":  {"svpt_won": 0.645, "rtpt_won": 0.338, "elo": 1955},
        "grass": {"svpt_won": 0.645, "rtpt_won": 0.325, "elo": 1905},
    },
    "cobolli": {
        "full_name": "Flavio Cobolli", "hand": "R", "rank": 30, "country": "ITA",
        "birth_year": 2002, "backhand": "2h",
        "hard":  {"svpt_won": 0.628, "rtpt_won": 0.335, "elo": 1930},
        "clay":  {"svpt_won": 0.635, "rtpt_won": 0.342, "elo": 1965},
        "grass": {"svpt_won": 0.625, "rtpt_won": 0.325, "elo": 1885},
    },
    "carreno_busta": {
        "full_name": "Pablo Carreno Busta", "hand": "R", "rank": 35, "country": "ESP",
        "birth_year": 1991, "backhand": "2h",
        "hard":  {"svpt_won": 0.622, "rtpt_won": 0.358, "elo": 1980},
        "clay":  {"svpt_won": 0.632, "rtpt_won": 0.382, "elo": 2080},
        "grass": {"svpt_won": 0.618, "rtpt_won": 0.348, "elo": 1940},
    },
    "tirante": {
        "full_name": "Thiago Agustin Tirante", "hand": "R", "rank": 85, "country": "ARG",
        "birth_year": 2002, "backhand": "2h",
        "hard":  {"svpt_won": 0.608, "rtpt_won": 0.338, "elo": 1830},
        "clay":  {"svpt_won": 0.618, "rtpt_won": 0.358, "elo": 1900},
        "grass": {"svpt_won": 0.608, "rtpt_won": 0.328, "elo": 1790},
    },
    "rune": {"full_name":"Holger Rune","hand":"R","rank":15,"country":"DEN","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.640,"rtpt_won":0.348,"elo":2000},
        "clay":{"svpt_won":0.642,"rtpt_won":0.355,"elo":2040},
        "grass":{"svpt_won":0.648,"rtpt_won":0.342,"elo":1990}},
    "lehecka": {"full_name":"Jiri Lehecka","hand":"R","rank":25,"country":"CZE","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.645,"rtpt_won":0.340,"elo":1975},
        "clay":{"svpt_won":0.638,"rtpt_won":0.338,"elo":1955},
        "grass":{"svpt_won":0.650,"rtpt_won":0.335,"elo":1965}},
    "korda": {"full_name":"Sebastian Korda","hand":"R","rank":30,"country":"USA","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.640,"rtpt_won":0.345,"elo":1960},
        "clay":{"svpt_won":0.628,"rtpt_won":0.340,"elo":1930},
        "grass":{"svpt_won":0.645,"rtpt_won":0.338,"elo":1945}},
    "fils": {"full_name":"Arthur Fils","hand":"R","rank":25,"country":"FRA","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.640,"rtpt_won":0.350,"elo":1970},
        "clay":{"svpt_won":0.645,"rtpt_won":0.355,"elo":1990},
        "grass":{"svpt_won":0.638,"rtpt_won":0.342,"elo":1945}},
    "sonego": {"full_name":"Lorenzo Sonego","hand":"R","rank":40,"country":"ITA","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.638,"rtpt_won":0.338,"elo":1940},
        "clay":{"svpt_won":0.640,"rtpt_won":0.340,"elo":1955},
        "grass":{"svpt_won":0.645,"rtpt_won":0.335,"elo":1940}},
    "arnaldi": {"full_name":"Matteo Arnaldi","hand":"R","rank":30,"country":"ITA","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.635,"rtpt_won":0.342,"elo":1950},
        "clay":{"svpt_won":0.638,"rtpt_won":0.345,"elo":1965},
        "grass":{"svpt_won":0.638,"rtpt_won":0.335,"elo":1935}},
    "griekspoor": {"full_name":"Tallon Griekspoor","hand":"R","rank":35,"country":"NED","birth_year":1996,"backhand":"2h",
        "hard":{"svpt_won":0.648,"rtpt_won":0.335,"elo":1940},
        "clay":{"svpt_won":0.635,"rtpt_won":0.330,"elo":1905},
        "grass":{"svpt_won":0.652,"rtpt_won":0.330,"elo":1940}},
    "baez": {"full_name":"Sebastian Baez","hand":"R","rank":35,"country":"ARG","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.615,"rtpt_won":0.348,"elo":1920},
        "clay":{"svpt_won":0.628,"rtpt_won":0.360,"elo":1980},
        "grass":{"svpt_won":0.610,"rtpt_won":0.330,"elo":1880}},
    "etcheverry": {"full_name":"Tomas Martin Etcheverry","hand":"R","rank":50,"country":"ARG","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.618,"rtpt_won":0.342,"elo":1900},
        "clay":{"svpt_won":0.630,"rtpt_won":0.355,"elo":1965},
        "grass":{"svpt_won":0.612,"rtpt_won":0.328,"elo":1870}},
    "tabilo": {"full_name":"Alejandro Tabilo","hand":"R","rank":30,"country":"CHI","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.638,"rtpt_won":0.350,"elo":1960},
        "clay":{"svpt_won":0.635,"rtpt_won":0.352,"elo":1970},
        "grass":{"svpt_won":0.635,"rtpt_won":0.340,"elo":1935}},
    "cerundolo": {"full_name":"Francisco Cerundolo","hand":"R","rank":35,"country":"ARG","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.628,"rtpt_won":0.345,"elo":1940},
        "clay":{"svpt_won":0.638,"rtpt_won":0.358,"elo":1985},
        "grass":{"svpt_won":0.620,"rtpt_won":0.330,"elo":1900}},
    "juan_cerundolo": {"full_name":"Juan Manuel Cerundolo","hand":"R","rank":50,"country":"ARG","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.622,"rtpt_won":0.340,"elo":1910},
        "clay":{"svpt_won":0.632,"rtpt_won":0.352,"elo":1960},
        "grass":{"svpt_won":0.615,"rtpt_won":0.325,"elo":1880}},
    "davidovich": {"full_name":"Alejandro Davidovich Fokina","hand":"R","rank":35,"country":"ESP","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.622,"rtpt_won":0.345,"elo":1935},
        "clay":{"svpt_won":0.632,"rtpt_won":0.355,"elo":1975},
        "grass":{"svpt_won":0.618,"rtpt_won":0.332,"elo":1905}},
    "navone": {"full_name":"Mariano Navone","hand":"R","rank":60,"country":"ARG","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.612,"rtpt_won":0.340,"elo":1890},
        "clay":{"svpt_won":0.625,"rtpt_won":0.352,"elo":1950},
        "grass":{"svpt_won":0.608,"rtpt_won":0.322,"elo":1860}},
    "fonseca": {"full_name":"Joao Fonseca","hand":"R","rank":40,"country":"BRA","birth_year":2006,"backhand":"2h",
        "hard":{"svpt_won":0.638,"rtpt_won":0.348,"elo":1960},
        "clay":{"svpt_won":0.632,"rtpt_won":0.345,"elo":1945},
        "grass":{"svpt_won":0.638,"rtpt_won":0.342,"elo":1940}},
    "mensik": {"full_name":"Jakub Mensik","hand":"R","rank":45,"country":"CZE","birth_year":2005,"backhand":"2h",
        "hard":{"svpt_won":0.648,"rtpt_won":0.338,"elo":1940},
        "clay":{"svpt_won":0.632,"rtpt_won":0.330,"elo":1900},
        "grass":{"svpt_won":0.652,"rtpt_won":0.335,"elo":1935}},
    "borges": {"full_name":"Nuno Borges","hand":"R","rank":40,"country":"POR","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.632,"rtpt_won":0.340,"elo":1930},
        "clay":{"svpt_won":0.628,"rtpt_won":0.342,"elo":1940},
        "grass":{"svpt_won":0.635,"rtpt_won":0.335,"elo":1920}},
    "kecmanovic": {"full_name":"Miomir Kecmanovic","hand":"R","rank":50,"country":"SRB","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.628,"rtpt_won":0.340,"elo":1920},
        "clay":{"svpt_won":0.625,"rtpt_won":0.342,"elo":1930},
        "grass":{"svpt_won":0.625,"rtpt_won":0.332,"elo":1900}},
    "marozsan": {"full_name":"Fabian Marozsan","hand":"R","rank":55,"country":"HUN","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.622,"rtpt_won":0.338,"elo":1900},
        "clay":{"svpt_won":0.635,"rtpt_won":0.348,"elo":1950},
        "grass":{"svpt_won":0.618,"rtpt_won":0.325,"elo":1870}},
    "shapovalov": {"full_name":"Denis Shapovalov","hand":"L","rank":60,"country":"CAN","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.638,"rtpt_won":0.348,"elo":1940},
        "clay":{"svpt_won":0.622,"rtpt_won":0.335,"elo":1880},
        "grass":{"svpt_won":0.645,"rtpt_won":0.342,"elo":1940}},
    "norrie": {"full_name":"Cameron Norrie","hand":"L","rank":55,"country":"GBR","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.625,"rtpt_won":0.342,"elo":1900},
        "clay":{"svpt_won":0.625,"rtpt_won":0.345,"elo":1910},
        "grass":{"svpt_won":0.630,"rtpt_won":0.340,"elo":1920}},
    "mannarino": {"full_name":"Adrian Mannarino","hand":"L","rank":60,"country":"FRA","birth_year":1988,"backhand":"2h",
        "hard":{"svpt_won":0.618,"rtpt_won":0.342,"elo":1890},
        "clay":{"svpt_won":0.618,"rtpt_won":0.345,"elo":1900},
        "grass":{"svpt_won":0.622,"rtpt_won":0.338,"elo":1895}},
    "moutet": {"full_name":"Corentin Moutet","hand":"L","rank":65,"country":"FRA","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.610,"rtpt_won":0.338,"elo":1870},
        "clay":{"svpt_won":0.618,"rtpt_won":0.345,"elo":1910},
        "grass":{"svpt_won":0.612,"rtpt_won":0.330,"elo":1860}},
    "svajda": {"full_name":"Zachary Svajda","hand":"R","rank":80,"country":"USA","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.632,"rtpt_won":0.328,"elo":1860},
        "clay":{"svpt_won":0.618,"rtpt_won":0.322,"elo":1830},
        "grass":{"svpt_won":0.635,"rtpt_won":0.325,"elo":1850}},
    "jodar": {"full_name":"Rafael Jodar","hand":"R","rank":90,"country":"ESP","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.615,"rtpt_won":0.325,"elo":1840},
        "clay":{"svpt_won":0.618,"rtpt_won":0.328,"elo":1855},
        "grass":{"svpt_won":0.612,"rtpt_won":0.318,"elo":1820}},
    "coric": {"full_name":"Borna Coric","hand":"R","rank":70,"country":"CRO","birth_year":1996,"backhand":"2h",
        "hard":{"svpt_won":0.630,"rtpt_won":0.338,"elo":1910},
        "clay":{"svpt_won":0.625,"rtpt_won":0.338,"elo":1900},
        "grass":{"svpt_won":0.628,"rtpt_won":0.330,"elo":1890}},
    "rinderknech": {"full_name":"Arthur Rinderknech","hand":"R","rank":65,"country":"FRA","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.645,"rtpt_won":0.328,"elo":1890},
        "clay":{"svpt_won":0.630,"rtpt_won":0.320,"elo":1870},
        "grass":{"svpt_won":0.645,"rtpt_won":0.322,"elo":1890}},
    "darderi": {"full_name":"Luciano Darderi","hand":"R","rank":45,"country":"ITA","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.628,"rtpt_won":0.340,"elo":1930},
        "clay":{"svpt_won":0.638,"rtpt_won":0.350,"elo":1970},
        "grass":{"svpt_won":0.622,"rtpt_won":0.328,"elo":1890}},
    "popyrin": {"full_name":"Alexei Popyrin","hand":"R","rank":35,"country":"AUS","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.648,"rtpt_won":0.335,"elo":1950},
        "clay":{"svpt_won":0.628,"rtpt_won":0.322,"elo":1890},
        "grass":{"svpt_won":0.652,"rtpt_won":0.330,"elo":1940}},
    "wawrinka": {"full_name":"Stan Wawrinka","hand":"R","rank":100,"country":"SUI","birth_year":1985,"backhand":"1h",
        "hard":{"svpt_won":0.638,"rtpt_won":0.345,"elo":1920},
        "clay":{"svpt_won":0.640,"rtpt_won":0.352,"elo":1960},
        "grass":{"svpt_won":0.638,"rtpt_won":0.335,"elo":1920}},
    "gaston": {"full_name":"Hugo Gaston","hand":"R","rank":75,"country":"FRA","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.608,"rtpt_won":0.335,"elo":1870},
        "clay":{"svpt_won":0.618,"rtpt_won":0.342,"elo":1910},
        "grass":{"svpt_won":0.612,"rtpt_won":0.325,"elo":1860}},
    # ── Ranks 85-300 (qualifying / lower-ranked tour players) ─────────────────
    "michael_zheng": {"full_name":"Michael Zheng","hand":"R","rank":100,"country":"USA","birth_year":2005,"backhand":"2h",
        "hard":{"svpt_won":0.601,"rtpt_won":0.320,"elo":1725},"clay":{"svpt_won":0.598,"rtpt_won":0.318,"elo":1705},"grass":{"svpt_won":0.600,"rtpt_won":0.318,"elo":1705}},
    "christopher_o'connell": {"full_name":"Christopher O'Connell","hand":"R","rank":85,"country":"AUS","birth_year":1994,"backhand":"2h",
        "hard":{"svpt_won":0.610,"rtpt_won":0.328,"elo":1762},"clay":{"svpt_won":0.608,"rtpt_won":0.326,"elo":1742},"grass":{"svpt_won":0.612,"rtpt_won":0.328,"elo":1742}},
    "marcelo_tomas_barrios_vera": {"full_name":"Marcelo Tomas Barrios Vera","hand":"R","rank":90,"country":"CHI","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.607,"rtpt_won":0.325,"elo":1750},"clay":{"svpt_won":0.608,"rtpt_won":0.326,"elo":1770},"grass":{"svpt_won":0.605,"rtpt_won":0.323,"elo":1725}},
    "roberto_carballes_baena": {"full_name":"Roberto Carballes Baena","hand":"R","rank":98,"country":"ESP","birth_year":1993,"backhand":"2h",
        "hard":{"svpt_won":0.602,"rtpt_won":0.321,"elo":1730},"clay":{"svpt_won":0.605,"rtpt_won":0.324,"elo":1755},"grass":{"svpt_won":0.598,"rtpt_won":0.318,"elo":1705}},
    "daniel_evans": {"full_name":"Daniel Evans","hand":"R","rank":100,"country":"GBR","birth_year":1990,"backhand":"2h",
        "hard":{"svpt_won":0.601,"rtpt_won":0.320,"elo":1725},"clay":{"svpt_won":0.596,"rtpt_won":0.315,"elo":1705},"grass":{"svpt_won":0.608,"rtpt_won":0.326,"elo":1755}},
    "mackenzie_mcdonald": {"full_name":"Mackenzie McDonald","hand":"R","rank":110,"country":"USA","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.595,"rtpt_won":0.315,"elo":1700},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1680},"grass":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1675}},
    "soonwoo_kwon": {"full_name":"Soonwoo Kwon","hand":"R","rank":105,"country":"KOR","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.598,"rtpt_won":0.318,"elo":1713},"clay":{"svpt_won":0.594,"rtpt_won":0.316,"elo":1693},"grass":{"svpt_won":0.594,"rtpt_won":0.316,"elo":1688}},
    "harold_mayot": {"full_name":"Harold Mayot","hand":"R","rank":108,"country":"FRA","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.596,"rtpt_won":0.316,"elo":1705},"clay":{"svpt_won":0.596,"rtpt_won":0.316,"elo":1695},"grass":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1680}},
    "bu_yunchaokete": {"full_name":"Bu Yunchaokete","hand":"R","rank":112,"country":"CHN","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.594,"rtpt_won":0.315,"elo":1695},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1675},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1670}},
    "laslo_djere": {"full_name":"Laslo Djere","hand":"R","rank":115,"country":"SRB","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1688},"clay":{"svpt_won":0.595,"rtpt_won":0.315,"elo":1708},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1663}},
    "luca_nardi": {"full_name":"Luca Nardi","hand":"R","rank":115,"country":"ITA","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1688},"clay":{"svpt_won":0.594,"rtpt_won":0.315,"elo":1698},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1663}},
    "darwin_blanch": {"full_name":"Darwin Blanch","hand":"R","rank":118,"country":"USA","birth_year":2005,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1680},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1670},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1655}},
    "timofey_skatov": {"full_name":"Timofey Skatov","hand":"R","rank":118,"country":"KAZ","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1680},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1668},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1655}},
    "dusan_lajovic": {"full_name":"Dusan Lajovic","hand":"R","rank":120,"country":"SRB","birth_year":1990,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1675},"clay":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1685},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1650}},
    "alejandro_moro_canas": {"full_name":"Alejandro Moro Canas","hand":"R","rank":125,"country":"ESP","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1663},"clay":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1673},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1638}},
    "alexis_galarneau": {"full_name":"Alexis Galarneau","hand":"R","rank":128,"country":"CAN","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1655},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1635},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1630}},
    "otto_virtanen": {"full_name":"Otto Virtanen","hand":"R","rank":128,"country":"FIN","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1655},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1635},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1630}},
    "roman_safiullin": {"full_name":"Roman Safiullin","hand":"R","rank":130,"country":"RUS","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1650},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1630},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1625}},
    "zsombor_piros": {"full_name":"Zsombor Piros","hand":"R","rank":135,"country":"HUN","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1638},"clay":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1648},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1613}},
    "clement_tabur": {"full_name":"Clement Tabur","hand":"R","rank":135,"country":"FRA","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1638},"clay":{"svpt_won":0.591,"rtpt_won":0.315,"elo":1648},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1613}},
    "billy_harris": {"full_name":"Billy Harris","hand":"R","rank":138,"country":"GBR","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1630},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1610},"grass":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1655}},
    "borna_gojo": {"full_name":"Borna Gojo","hand":"R","rank":140,"country":"CRO","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1625},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1605},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600}},
    "tristan_boyer": {"full_name":"Tristan Boyer","hand":"R","rank":148,"country":"USA","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1605},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1585},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "aziz_dougaz": {"full_name":"Aziz Dougaz","hand":"R","rank":148,"country":"TUN","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1605},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1585},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "dane_sweeny": {"full_name":"Dane Sweeny","hand":"R","rank":148,"country":"AUS","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1605},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1585},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1585}},
    "elias_ymer": {"full_name":"Elias Ymer","hand":"R","rank":152,"country":"SWE","birth_year":1996,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "august_holmgren": {"full_name":"August Holmgren","hand":"R","rank":165,"country":"DEN","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "jaime_faria": {"full_name":"Jaime Faria","hand":"R","rank":160,"country":"POR","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1585},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "andrea_pellegrino": {"full_name":"Andrea Pellegrino","hand":"R","rank":172,"country":"ITA","birth_year":1993,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1585},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "arthur_gea": {"full_name":"Arthur Gea","hand":"R","rank":175,"country":"FRA","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1585},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "gustavo_heide": {"full_name":"Gustavo Heide","hand":"R","rank":170,"country":"BRA","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1582},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "oliver_tarvet": {"full_name":"Oliver Tarvet","hand":"R","rank":170,"country":"GBR","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1615}},
    "luka_pavlovic": {"full_name":"Luka Pavlovic","hand":"R","rank":175,"country":"USA","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "tristan_schoolkate": {"full_name":"Tristan Schoolkate","hand":"R","rank":185,"country":"AUS","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "shintaro_mochizuki": {"full_name":"Shintaro Mochizuki","hand":"R","rank":178,"country":"JPN","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "jerome_kym": {"full_name":"Jerome Kym","hand":"R","rank":188,"country":"SUI","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "nicolas_mejia": {"full_name":"Nicolas Mejia","hand":"R","rank":192,"country":"COL","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1585},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "paul_jubb": {"full_name":"Paul Jubb","hand":"R","rank":192,"country":"GBR","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1620}},
    "kimmer_coppejans": {"full_name":"Kimmer Coppejans","hand":"R","rank":195,"country":"BEL","birth_year":1993,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "chris_rodesch": {"full_name":"Chris Rodesch","hand":"R","rank":205,"country":"LUX","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "henry_searle": {"full_name":"Henry Searle","hand":"R","rank":205,"country":"GBR","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1630}},
    "stefano_travaglia": {"full_name":"Stefano Travaglia","hand":"R","rank":210,"country":"ITA","birth_year":1991,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "rei_sakamoto": {"full_name":"Rei Sakamoto","hand":"R","rank":215,"country":"JPN","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "vilius_gaubas": {"full_name":"Vilius Gaubas","hand":"R","rank":220,"country":"LTU","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "gauthier_onclin": {"full_name":"Gauthier Onclin","hand":"R","rank":228,"country":"BEL","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "colton_smith": {"full_name":"Colton Smith","hand":"R","rank":245,"country":"USA","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "bernard_tomic": {"full_name":"Bernard Tomic","hand":"R","rank":250,"country":"AUS","birth_year":1992,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580},"grass":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1615}},
    "pol_martin_tiffon": {"full_name":"Pol Martin Tiffon","hand":"R","rank":268,"country":"ESP","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1582},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "pablo_llamas_ruiz": {"full_name":"Pablo Llamas Ruiz","hand":"R","rank":280,"country":"ESP","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1600},"clay":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1582},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1580}},
    "federico_cina": {"full_name":"Federico Cina","hand":"R","rank":130,"country":"ITA","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1650},"clay":{"svpt_won":0.592,"rtpt_won":0.315,"elo":1665},"grass":{"svpt_won":0.590,"rtpt_won":0.315,"elo":1625}},
}

# ─────────────────────────────────────────────────────────────────────────────
# WTA PLAYER DATABASE
# ─────────────────────────────────────────────────────────────────────────────
WTA_STATS: Dict[str, dict] = {
    "swiatek": {
        "full_name": "Iga Swiatek", "hand": "R", "rank": 2, "country": "POL",
        "birth_year": 2001, "backhand": "2h",
        "hard":  {"svpt_won": 0.580, "rtpt_won": 0.440, "elo": 2250},
        "clay":  {"svpt_won": 0.590, "rtpt_won": 0.455, "elo": 2355},
        "grass": {"svpt_won": 0.568, "rtpt_won": 0.418, "elo": 2120},
    },
    "sabalenka": {
        "full_name": "Aryna Sabalenka", "hand": "R", "rank": 1, "country": "BLR",
        "birth_year": 1998, "backhand": "2h",
        "hard":  {"svpt_won": 0.598, "rtpt_won": 0.418, "elo": 2215},
        "clay":  {"svpt_won": 0.582, "rtpt_won": 0.408, "elo": 2120},
        "grass": {"svpt_won": 0.595, "rtpt_won": 0.405, "elo": 2145},
    },
    "gauff": {
        "full_name": "Coco Gauff", "hand": "R", "rank": 3, "country": "USA",
        "birth_year": 2004, "backhand": "2h",
        "hard":  {"svpt_won": 0.578, "rtpt_won": 0.415, "elo": 2125},
        "clay":  {"svpt_won": 0.572, "rtpt_won": 0.412, "elo": 2090},
        "grass": {"svpt_won": 0.565, "rtpt_won": 0.400, "elo": 2055},
    },
    "rybakina": {
        "full_name": "Elena Rybakina", "hand": "R", "rank": 7, "country": "KAZ",
        "birth_year": 1999, "backhand": "2h",
        "hard":  {"svpt_won": 0.595, "rtpt_won": 0.408, "elo": 2155},
        "clay":  {"svpt_won": 0.578, "rtpt_won": 0.398, "elo": 2075},
        "grass": {"svpt_won": 0.605, "rtpt_won": 0.408, "elo": 2175},
    },
    "pegula": {
        "full_name": "Jessica Pegula", "hand": "R", "rank": 6, "country": "USA",
        "birth_year": 1994, "backhand": "2h",
        "hard":  {"svpt_won": 0.572, "rtpt_won": 0.405, "elo": 2070},
        "clay":  {"svpt_won": 0.558, "rtpt_won": 0.392, "elo": 1985},
        "grass": {"svpt_won": 0.560, "rtpt_won": 0.388, "elo": 1985},
    },
    "keys": {
        "full_name": "Madison Keys", "hand": "R", "rank": 5, "country": "USA",
        "birth_year": 1995, "backhand": "2h",
        "hard":  {"svpt_won": 0.582, "rtpt_won": 0.395, "elo": 2065},
        "clay":  {"svpt_won": 0.565, "rtpt_won": 0.378, "elo": 1985},
        "grass": {"svpt_won": 0.580, "rtpt_won": 0.380, "elo": 2020},
    },
    "zheng": {
        "full_name": "Qinwen Zheng", "hand": "R", "rank": 8, "country": "CHN",
        "birth_year": 2002, "backhand": "2h",
        "hard":  {"svpt_won": 0.575, "rtpt_won": 0.400, "elo": 2060},
        "clay":  {"svpt_won": 0.568, "rtpt_won": 0.395, "elo": 2035},
        "grass": {"svpt_won": 0.565, "rtpt_won": 0.385, "elo": 2005},
    },
    "paolini": {
        "full_name": "Jasmine Paolini", "hand": "R", "rank": 4, "country": "ITA",
        "birth_year": 1996, "backhand": "2h",
        "hard":  {"svpt_won": 0.562, "rtpt_won": 0.402, "elo": 2050},
        "clay":  {"svpt_won": 0.568, "rtpt_won": 0.410, "elo": 2090},
        "grass": {"svpt_won": 0.555, "rtpt_won": 0.388, "elo": 2010},
    },
    "navarro": {
        "full_name": "Emma Navarro", "hand": "R", "rank": 9, "country": "USA",
        "birth_year": 2001, "backhand": "2h",
        "hard":  {"svpt_won": 0.562, "rtpt_won": 0.395, "elo": 2020},
        "clay":  {"svpt_won": 0.552, "rtpt_won": 0.382, "elo": 1965},
        "grass": {"svpt_won": 0.568, "rtpt_won": 0.392, "elo": 2025},
    },
    "krejcikova": {
        "full_name": "Barbora Krejcikova", "hand": "R", "rank": 10, "country": "CZE",
        "birth_year": 1996, "backhand": "1h",
        "hard":  {"svpt_won": 0.555, "rtpt_won": 0.388, "elo": 1975},
        "clay":  {"svpt_won": 0.565, "rtpt_won": 0.400, "elo": 2025},
        "grass": {"svpt_won": 0.568, "rtpt_won": 0.395, "elo": 2030},
    },
    "sakkari": {
        "full_name": "Maria Sakkari", "hand": "R", "rank": 12, "country": "GRE",
        "birth_year": 1995, "backhand": "2h",
        "hard":  {"svpt_won": 0.562, "rtpt_won": 0.385, "elo": 2000},
        "clay":  {"svpt_won": 0.558, "rtpt_won": 0.382, "elo": 1990},
        "grass": {"svpt_won": 0.555, "rtpt_won": 0.370, "elo": 1955},
    },
    "kasatkina": {
        "full_name": "Daria Kasatkina", "hand": "R", "rank": 15, "country": "RUS",
        "birth_year": 1997, "backhand": "2h",
        "hard":  {"svpt_won": 0.555, "rtpt_won": 0.388, "elo": 1975},
        "clay":  {"svpt_won": 0.562, "rtpt_won": 0.395, "elo": 2005},
        "grass": {"svpt_won": 0.548, "rtpt_won": 0.375, "elo": 1935},
    },
    "kvitova": {
        "full_name": "Petra Kvitova", "hand": "L", "rank": 80, "country": "CZE",
        "birth_year": 1990, "backhand": "2h",
        "hard":  {"svpt_won": 0.575, "rtpt_won": 0.378, "elo": 1955},
        "clay":  {"svpt_won": 0.558, "rtpt_won": 0.360, "elo": 1880},
        "grass": {"svpt_won": 0.590, "rtpt_won": 0.378, "elo": 2010},
    },
    "haddad_maia": {
        "full_name": "Beatriz Haddad Maia", "hand": "L", "rank": 24, "country": "BRA",
        "birth_year": 1996, "backhand": "2h",
        "hard":  {"svpt_won": 0.552, "rtpt_won": 0.378, "elo": 1935},
        "clay":  {"svpt_won": 0.562, "rtpt_won": 0.392, "elo": 1985},
        "grass": {"svpt_won": 0.548, "rtpt_won": 0.368, "elo": 1900},
    },
    "kostyuk": {
        "full_name": "Marta Kostyuk", "hand": "R", "rank": 22, "country": "UKR",
        "birth_year": 2002, "backhand": "2h",
        "hard":  {"svpt_won": 0.562, "rtpt_won": 0.385, "elo": 1975},
        "clay":  {"svpt_won": 0.552, "rtpt_won": 0.378, "elo": 1940},
        "grass": {"svpt_won": 0.558, "rtpt_won": 0.375, "elo": 1945},
    },
    "bencic": {
        "full_name": "Belinda Bencic", "hand": "R", "rank": 45, "country": "SUI",
        "birth_year": 1997, "backhand": "1h",
        "hard":  {"svpt_won": 0.558, "rtpt_won": 0.385, "elo": 1965},
        "clay":  {"svpt_won": 0.548, "rtpt_won": 0.375, "elo": 1920},
        "grass": {"svpt_won": 0.555, "rtpt_won": 0.375, "elo": 1940},
    },
    "collins": {
        "full_name": "Danielle Collins", "hand": "R", "rank": 50, "country": "USA",
        "birth_year": 1994, "backhand": "2h",
        "hard":  {"svpt_won": 0.568, "rtpt_won": 0.385, "elo": 1985},
        "clay":  {"svpt_won": 0.555, "rtpt_won": 0.372, "elo": 1935},
        "grass": {"svpt_won": 0.558, "rtpt_won": 0.365, "elo": 1920},
    },
    "osaka": {"full_name":"Naomi Osaka","hand":"R","rank":50,"country":"JPN","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.582,"rtpt_won":0.385,"elo":2030},
        "clay":{"svpt_won":0.558,"rtpt_won":0.365,"elo":1900},
        "grass":{"svpt_won":0.578,"rtpt_won":0.372,"elo":1980}},
    "shnaider": {"full_name":"Diana Shnaider","hand":"L","rank":25,"country":"RUS","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.572,"rtpt_won":0.390,"elo":1990},
        "clay":{"svpt_won":0.568,"rtpt_won":0.388,"elo":1985},
        "grass":{"svpt_won":0.565,"rtpt_won":0.375,"elo":1955}},
    "kalinskaya": {"full_name":"Anna Kalinskaya","hand":"R","rank":35,"country":"RUS","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.565,"rtpt_won":0.382,"elo":1960},
        "clay":{"svpt_won":0.555,"rtpt_won":0.372,"elo":1920},
        "grass":{"svpt_won":0.562,"rtpt_won":0.372,"elo":1935}},
    "andreeva": {"full_name":"Mirra Andreeva","hand":"R","rank":20,"country":"RUS","birth_year":2007,"backhand":"2h",
        "hard":{"svpt_won":0.562,"rtpt_won":0.392,"elo":1995},
        "clay":{"svpt_won":0.565,"rtpt_won":0.395,"elo":2010},
        "grass":{"svpt_won":0.558,"rtpt_won":0.378,"elo":1955}},
    "svitolina": {"full_name":"Elina Svitolina","hand":"R","rank":25,"country":"UKR","birth_year":1994,"backhand":"2h",
        "hard":{"svpt_won":0.562,"rtpt_won":0.390,"elo":1985},
        "clay":{"svpt_won":0.558,"rtpt_won":0.385,"elo":1970},
        "grass":{"svpt_won":0.558,"rtpt_won":0.378,"elo":1975}},
    "samsonova": {"full_name":"Liudmila Samsonova","hand":"R","rank":30,"country":"RUS","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.572,"rtpt_won":0.380,"elo":1970},
        "clay":{"svpt_won":0.558,"rtpt_won":0.368,"elo":1930},
        "grass":{"svpt_won":0.568,"rtpt_won":0.368,"elo":1945}},
    "badosa": {"full_name":"Paula Badosa","hand":"R","rank":25,"country":"ESP","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.568,"rtpt_won":0.385,"elo":1980},
        "clay":{"svpt_won":0.565,"rtpt_won":0.388,"elo":1985},
        "grass":{"svpt_won":0.562,"rtpt_won":0.375,"elo":1950}},
    "potapova": {"full_name":"Anastasia Potapova","hand":"R","rank":35,"country":"RUS","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.568,"rtpt_won":0.378,"elo":1955},
        "clay":{"svpt_won":0.558,"rtpt_won":0.370,"elo":1930},
        "grass":{"svpt_won":0.562,"rtpt_won":0.365,"elo":1920}},
    "jabeur": {"full_name":"Ons Jabeur","hand":"R","rank":30,"country":"TUN","birth_year":1994,"backhand":"1h",
        "hard":{"svpt_won":0.552,"rtpt_won":0.392,"elo":1975},
        "clay":{"svpt_won":0.558,"rtpt_won":0.398,"elo":1995},
        "grass":{"svpt_won":0.558,"rtpt_won":0.390,"elo":1985}},
    "yastremska": {"full_name":"Dayana Yastremska","hand":"R","rank":35,"country":"UKR","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.568,"rtpt_won":0.378,"elo":1955},
        "clay":{"svpt_won":0.555,"rtpt_won":0.368,"elo":1920},
        "grass":{"svpt_won":0.562,"rtpt_won":0.368,"elo":1930}},
    "garcia": {"full_name":"Caroline Garcia","hand":"R","rank":40,"country":"FRA","birth_year":1993,"backhand":"2h",
        "hard":{"svpt_won":0.578,"rtpt_won":0.375,"elo":1970},
        "clay":{"svpt_won":0.565,"rtpt_won":0.365,"elo":1930},
        "grass":{"svpt_won":0.572,"rtpt_won":0.365,"elo":1955}},
    "muchova": {"full_name":"Karolina Muchova","hand":"R","rank":40,"country":"CZE","birth_year":1996,"backhand":"1h",
        "hard":{"svpt_won":0.558,"rtpt_won":0.385,"elo":1955},
        "clay":{"svpt_won":0.558,"rtpt_won":0.390,"elo":1965},
        "grass":{"svpt_won":0.555,"rtpt_won":0.382,"elo":1945}},
    "ostapenko": {"full_name":"Jelena Ostapenko","hand":"R","rank":35,"country":"LAT","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.572,"rtpt_won":0.372,"elo":1960},
        "clay":{"svpt_won":0.568,"rtpt_won":0.368,"elo":1955},
        "grass":{"svpt_won":0.575,"rtpt_won":0.368,"elo":1975}},
    "noskova": {"full_name":"Linda Noskova","hand":"R","rank":35,"country":"CZE","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.568,"rtpt_won":0.375,"elo":1950},
        "clay":{"svpt_won":0.555,"rtpt_won":0.365,"elo":1910},
        "grass":{"svpt_won":0.565,"rtpt_won":0.368,"elo":1935}},
    "frech": {"full_name":"Magdalena Frech","hand":"R","rank":40,"country":"POL","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.555,"rtpt_won":0.372,"elo":1930},
        "clay":{"svpt_won":0.558,"rtpt_won":0.378,"elo":1945},
        "grass":{"svpt_won":0.548,"rtpt_won":0.360,"elo":1900}},
    "cirstea": {"full_name":"Sorana Cirstea","hand":"R","rank":50,"country":"ROU","birth_year":1990,"backhand":"2h",
        "hard":{"svpt_won":0.565,"rtpt_won":0.368,"elo":1940},
        "clay":{"svpt_won":0.558,"rtpt_won":0.362,"elo":1925},
        "grass":{"svpt_won":0.558,"rtpt_won":0.358,"elo":1920}},
    "parry": {"full_name":"Diane Parry","hand":"R","rank":60,"country":"FRA","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.545,"rtpt_won":0.360,"elo":1880},
        "clay":{"svpt_won":0.552,"rtpt_won":0.368,"elo":1910},
        "grass":{"svpt_won":0.538,"rtpt_won":0.345,"elo":1860}},
    "chwalinska": {"full_name":"Maja Chwalinska","hand":"R","rank":60,"country":"POL","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.552,"rtpt_won":0.362,"elo":1890},
        "clay":{"svpt_won":0.558,"rtpt_won":0.368,"elo":1910},
        "grass":{"svpt_won":0.545,"rtpt_won":0.348,"elo":1865}},
    "vekic": {"full_name":"Donna Vekic","hand":"R","rank":40,"country":"CRO","birth_year":1996,"backhand":"2h",
        "hard":{"svpt_won":0.568,"rtpt_won":0.375,"elo":1955},
        "clay":{"svpt_won":0.552,"rtpt_won":0.360,"elo":1905},
        "grass":{"svpt_won":0.568,"rtpt_won":0.370,"elo":1955}},
    "kudermetova": {"full_name":"Veronika Kudermetova","hand":"R","rank":45,"country":"RUS","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.558,"rtpt_won":0.375,"elo":1940},
        "clay":{"svpt_won":0.552,"rtpt_won":0.368,"elo":1910},
        "grass":{"svpt_won":0.552,"rtpt_won":0.360,"elo":1910}},
    "raducanu": {"full_name":"Emma Raducanu","hand":"R","rank":55,"country":"GBR","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.565,"rtpt_won":0.375,"elo":1940},
        "clay":{"svpt_won":0.548,"rtpt_won":0.362,"elo":1895},
        "grass":{"svpt_won":0.562,"rtpt_won":0.368,"elo":1935}},
    "tauson": {"full_name":"Clara Tauson","hand":"R","rank":50,"country":"DEN","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.562,"rtpt_won":0.370,"elo":1925},
        "clay":{"svpt_won":0.548,"rtpt_won":0.360,"elo":1895},
        "grass":{"svpt_won":0.558,"rtpt_won":0.362,"elo":1910}},
    "blinkova": {"full_name":"Anna Blinkova","hand":"R","rank":55,"country":"RUS","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.555,"rtpt_won":0.368,"elo":1920},
        "clay":{"svpt_won":0.550,"rtpt_won":0.362,"elo":1905},
        "grass":{"svpt_won":0.548,"rtpt_won":0.352,"elo":1895}},
    "alexandrova": {"full_name":"Ekaterina Alexandrova","hand":"R","rank":50,"country":"RUS","birth_year":1994,"backhand":"2h",
        "hard":{"svpt_won":0.572,"rtpt_won":0.368,"elo":1940},
        "clay":{"svpt_won":0.552,"rtpt_won":0.355,"elo":1895},
        "grass":{"svpt_won":0.565,"rtpt_won":0.358,"elo":1920}},
    "xinyu_wang": {"full_name":"Xinyu Wang","hand":"R","rank":60,"country":"CHN","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.555,"rtpt_won":0.362,"elo":1900},
        "clay":{"svpt_won":0.548,"rtpt_won":0.355,"elo":1875},
        "grass":{"svpt_won":0.548,"rtpt_won":0.348,"elo":1875}},
    # ── Extended WTA database (ranks 11–130) ──────────────────────────────────
    "elise_mertens":       {"full_name":"Elise Mertens","hand":"R","rank":22,"country":"BEL","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.560,"rtpt_won":0.408,"elo":1916},"clay":{"svpt_won":0.555,"rtpt_won":0.403,"elo":1900},"grass":{"svpt_won":0.552,"rtpt_won":0.392,"elo":1891}},
    "simona_halep":        {"full_name":"Simona Halep","hand":"R","rank":30,"country":"ROU","birth_year":1991,"backhand":"2h",
        "hard":{"svpt_won":0.558,"rtpt_won":0.408,"elo":1895},"clay":{"svpt_won":0.565,"rtpt_won":0.418,"elo":1940},"grass":{"svpt_won":0.550,"rtpt_won":0.395,"elo":1875}},
    "karolina_pliskova":   {"full_name":"Karolina Pliskova","hand":"R","rank":35,"country":"CZE","birth_year":1992,"backhand":"2h",
        "hard":{"svpt_won":0.578,"rtpt_won":0.390,"elo":1888},"clay":{"svpt_won":0.558,"rtpt_won":0.375,"elo":1842},"grass":{"svpt_won":0.582,"rtpt_won":0.385,"elo":1908}},
    "katie_boulter":       {"full_name":"Katie Boulter","hand":"R","rank":32,"country":"GBR","birth_year":1996,"backhand":"2h",
        "hard":{"svpt_won":0.565,"rtpt_won":0.390,"elo":1884},"clay":{"svpt_won":0.548,"rtpt_won":0.372,"elo":1845},"grass":{"svpt_won":0.572,"rtpt_won":0.388,"elo":1920}},
    "celine_naef":         {"full_name":"Celine Naef","hand":"R","rank":38,"country":"SUI","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.562,"rtpt_won":0.401,"elo":1866},"clay":{"svpt_won":0.552,"rtpt_won":0.392,"elo":1846},"grass":{"svpt_won":0.555,"rtpt_won":0.388,"elo":1841}},
    "anett_kontaveit":     {"full_name":"Anett Kontaveit","hand":"R","rank":40,"country":"EST","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.568,"rtpt_won":0.392,"elo":1858},"clay":{"svpt_won":0.555,"rtpt_won":0.382,"elo":1830},"grass":{"svpt_won":0.558,"rtpt_won":0.378,"elo":1833}},
    "linda_fruhvirtova":   {"full_name":"Linda Fruhvirtova","hand":"L","rank":46,"country":"CZE","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.557,"rtpt_won":0.396,"elo":1838},"clay":{"svpt_won":0.549,"rtpt_won":0.388,"elo":1818},"grass":{"svpt_won":0.550,"rtpt_won":0.388,"elo":1813}},
    "leylah_fernandez":    {"full_name":"Leylah Fernandez","hand":"L","rank":48,"country":"CAN","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.555,"rtpt_won":0.395,"elo":1831},"clay":{"svpt_won":0.548,"rtpt_won":0.388,"elo":1811},"grass":{"svpt_won":0.548,"rtpt_won":0.385,"elo":1806}},
    "anastasia_pavlyuchenkova": {"full_name":"Anastasia Pavlyuchenkova","hand":"R","rank":48,"country":"RUS","birth_year":1991,"backhand":"1h",
        "hard":{"svpt_won":0.558,"rtpt_won":0.392,"elo":1831},"clay":{"svpt_won":0.562,"rtpt_won":0.400,"elo":1850},"grass":{"svpt_won":0.548,"rtpt_won":0.378,"elo":1806}},
    "bianca_andreescu":    {"full_name":"Bianca Andreescu","hand":"R","rank":52,"country":"CAN","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.555,"rtpt_won":0.393,"elo":1820},"clay":{"svpt_won":0.545,"rtpt_won":0.382,"elo":1796},"grass":{"svpt_won":0.545,"rtpt_won":0.382,"elo":1791}},
    "lulu_sun":            {"full_name":"Lulu Sun","hand":"R","rank":52,"country":"NZL","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.553,"rtpt_won":0.393,"elo":1816},"clay":{"svpt_won":0.545,"rtpt_won":0.382,"elo":1796},"grass":{"svpt_won":0.552,"rtpt_won":0.388,"elo":1830}},
    "elisabetta_cocciaretto": {"full_name":"Elisabetta Cocciaretto","hand":"R","rank":52,"country":"ITA","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.553,"rtpt_won":0.393,"elo":1816},"clay":{"svpt_won":0.558,"rtpt_won":0.398,"elo":1830},"grass":{"svpt_won":0.545,"rtpt_won":0.382,"elo":1791}},
    "peyton_stearns":      {"full_name":"Peyton Stearns","hand":"R","rank":56,"country":"USA","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.550,"rtpt_won":0.390,"elo":1821},"clay":{"svpt_won":0.542,"rtpt_won":0.380,"elo":1801},"grass":{"svpt_won":0.542,"rtpt_won":0.378,"elo":1796}},
    "victoria_azarenka":   {"full_name":"Victoria Azarenka","hand":"R","rank":55,"country":"BLR","birth_year":1989,"backhand":"2h",
        "hard":{"svpt_won":0.568,"rtpt_won":0.395,"elo":1845},"clay":{"svpt_won":0.555,"rtpt_won":0.382,"elo":1810},"grass":{"svpt_won":0.558,"rtpt_won":0.380,"elo":1820}},
    "yulia_putintseva":    {"full_name":"Yulia Putintseva","hand":"R","rank":55,"country":"KAZ","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.545,"rtpt_won":0.390,"elo":1808},"clay":{"svpt_won":0.548,"rtpt_won":0.395,"elo":1815},"grass":{"svpt_won":0.538,"rtpt_won":0.375,"elo":1783}},
    "anhelina_kalinina":   {"full_name":"Anhelina Kalinina","hand":"R","rank":55,"country":"UKR","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.550,"rtpt_won":0.390,"elo":1808},"clay":{"svpt_won":0.545,"rtpt_won":0.384,"elo":1792},"grass":{"svpt_won":0.542,"rtpt_won":0.375,"elo":1783}},
    "moyuka_uchijima":     {"full_name":"Moyuka Uchijima","hand":"R","rank":62,"country":"JPN","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.546,"rtpt_won":0.387,"elo":1804},"clay":{"svpt_won":0.540,"rtpt_won":0.379,"elo":1784},"grass":{"svpt_won":0.538,"rtpt_won":0.372,"elo":1779}},
    "zhu_lin":             {"full_name":"Zhu Lin","hand":"R","rank":62,"country":"CHN","birth_year":1994,"backhand":"2h",
        "hard":{"svpt_won":0.546,"rtpt_won":0.387,"elo":1804},"clay":{"svpt_won":0.540,"rtpt_won":0.379,"elo":1784},"grass":{"svpt_won":0.538,"rtpt_won":0.372,"elo":1779}},
    "magda_linette":       {"full_name":"Magda Linette","hand":"R","rank":65,"country":"POL","birth_year":1992,"backhand":"2h",
        "hard":{"svpt_won":0.544,"rtpt_won":0.385,"elo":1796},"clay":{"svpt_won":0.546,"rtpt_won":0.388,"elo":1802},"grass":{"svpt_won":0.537,"rtpt_won":0.372,"elo":1771}},
    "alycia_parks":        {"full_name":"Alycia Parks","hand":"R","rank":65,"country":"USA","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.558,"rtpt_won":0.380,"elo":1808},"clay":{"svpt_won":0.545,"rtpt_won":0.370,"elo":1772},"grass":{"svpt_won":0.552,"rtpt_won":0.375,"elo":1793}},
    "elina_avanesyan":     {"full_name":"Elina Avanesyan","hand":"R","rank":65,"country":"RUS","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.544,"rtpt_won":0.385,"elo":1796},"clay":{"svpt_won":0.542,"rtpt_won":0.382,"elo":1789},"grass":{"svpt_won":0.537,"rtpt_won":0.372,"elo":1771}},
    "camila_osorio":       {"full_name":"Camila Osorio","hand":"R","rank":70,"country":"COL","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.540,"rtpt_won":0.382,"elo":1782},"clay":{"svpt_won":0.550,"rtpt_won":0.392,"elo":1808},"grass":{"svpt_won":0.532,"rtpt_won":0.368,"elo":1757}},
    "varvara_gracheva":    {"full_name":"Varvara Gracheva","hand":"R","rank":68,"country":"FRA","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.541,"rtpt_won":0.383,"elo":1788},"clay":{"svpt_won":0.544,"rtpt_won":0.387,"elo":1800},"grass":{"svpt_won":0.534,"rtpt_won":0.370,"elo":1763}},
    "lucia_bronzetti":     {"full_name":"Lucia Bronzetti","hand":"R","rank":68,"country":"ITA","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.541,"rtpt_won":0.383,"elo":1788},"clay":{"svpt_won":0.550,"rtpt_won":0.392,"elo":1808},"grass":{"svpt_won":0.534,"rtpt_won":0.370,"elo":1763}},
    "eva_lys":             {"full_name":"Eva Lys","hand":"R","rank":72,"country":"GER","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.539,"rtpt_won":0.381,"elo":1776},"clay":{"svpt_won":0.535,"rtpt_won":0.376,"elo":1756},"grass":{"svpt_won":0.532,"rtpt_won":0.368,"elo":1751}},
    "harmony_tan":         {"full_name":"Harmony Tan","hand":"R","rank":72,"country":"FRA","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.539,"rtpt_won":0.381,"elo":1776},"clay":{"svpt_won":0.535,"rtpt_won":0.376,"elo":1756},"grass":{"svpt_won":0.540,"rtpt_won":0.380,"elo":1786}},
    "mayar_sherif":        {"full_name":"Mayar Sherif","hand":"R","rank":65,"country":"EGY","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.536,"rtpt_won":0.380,"elo":1778},"clay":{"svpt_won":0.548,"rtpt_won":0.393,"elo":1808},"grass":{"svpt_won":0.528,"rtpt_won":0.365,"elo":1753}},
    "zeynep_sonmez":       {"full_name":"Zeynep Sonmez","hand":"R","rank":75,"country":"TUR","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.537,"rtpt_won":0.379,"elo":1768},"clay":{"svpt_won":0.540,"rtpt_won":0.382,"elo":1778},"grass":{"svpt_won":0.529,"rtpt_won":0.362,"elo":1743}},
    "cristina_bucsa":      {"full_name":"Cristina Bucsa","hand":"R","rank":75,"country":"ESP","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.537,"rtpt_won":0.379,"elo":1768},"clay":{"svpt_won":0.548,"rtpt_won":0.390,"elo":1798},"grass":{"svpt_won":0.528,"rtpt_won":0.362,"elo":1743}},
    "jessica_bouzas_maneiro": {"full_name":"Jessica Bouzas Maneiro","hand":"R","rank":75,"country":"ESP","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.537,"rtpt_won":0.379,"elo":1768},"clay":{"svpt_won":0.548,"rtpt_won":0.390,"elo":1798},"grass":{"svpt_won":0.528,"rtpt_won":0.362,"elo":1743}},
    "laura_siegemund":     {"full_name":"Laura Siegemund","hand":"R","rank":75,"country":"GER","birth_year":1988,"backhand":"2h",
        "hard":{"svpt_won":0.545,"rtpt_won":0.384,"elo":1782},"clay":{"svpt_won":0.550,"rtpt_won":0.390,"elo":1795},"grass":{"svpt_won":0.555,"rtpt_won":0.388,"elo":1808}},
    "katerina_siniakova":  {"full_name":"Katerina Siniakova","hand":"R","rank":78,"country":"CZE","birth_year":1996,"backhand":"2h",
        "hard":{"svpt_won":0.540,"rtpt_won":0.380,"elo":1760},"clay":{"svpt_won":0.548,"rtpt_won":0.388,"elo":1780},"grass":{"svpt_won":0.545,"rtpt_won":0.380,"elo":1770}},
    "claire_liu":          {"full_name":"Claire Liu","hand":"R","rank":80,"country":"USA","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.533,"rtpt_won":0.377,"elo":1754},"clay":{"svpt_won":0.528,"rtpt_won":0.368,"elo":1734},"grass":{"svpt_won":0.526,"rtpt_won":0.362,"elo":1729}},
    "ann_li":              {"full_name":"Ann Li","hand":"R","rank":80,"country":"USA","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.533,"rtpt_won":0.377,"elo":1754},"clay":{"svpt_won":0.528,"rtpt_won":0.368,"elo":1734},"grass":{"svpt_won":0.526,"rtpt_won":0.362,"elo":1729}},
    "marie_bouzkova":      {"full_name":"Marie Bouzkova","hand":"R","rank":80,"country":"CZE","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.533,"rtpt_won":0.377,"elo":1754},"clay":{"svpt_won":0.538,"rtpt_won":0.382,"elo":1770},"grass":{"svpt_won":0.530,"rtpt_won":0.370,"elo":1740}},
    "lesia_tsurenko":      {"full_name":"Lesia Tsurenko","hand":"R","rank":80,"country":"UKR","birth_year":1989,"backhand":"2h",
        "hard":{"svpt_won":0.533,"rtpt_won":0.377,"elo":1754},"clay":{"svpt_won":0.530,"rtpt_won":0.374,"elo":1744},"grass":{"svpt_won":0.526,"rtpt_won":0.362,"elo":1729}},
    "sara_sorribes_tormo": {"full_name":"Sara Sorribes Tormo","hand":"R","rank":80,"country":"ESP","birth_year":1996,"backhand":"2h",
        "hard":{"svpt_won":0.522,"rtpt_won":0.374,"elo":1728},"clay":{"svpt_won":0.548,"rtpt_won":0.400,"elo":1808},"grass":{"svpt_won":0.514,"rtpt_won":0.355,"elo":1703}},
    "bernarda_pera":       {"full_name":"Bernarda Pera","hand":"R","rank":78,"country":"USA","birth_year":1994,"backhand":"2h",
        "hard":{"svpt_won":0.540,"rtpt_won":0.378,"elo":1760},"clay":{"svpt_won":0.530,"rtpt_won":0.368,"elo":1740},"grass":{"svpt_won":0.532,"rtpt_won":0.365,"elo":1735}},
    "lucrezia_stefanini":  {"full_name":"Lucrezia Stefanini","hand":"R","rank":82,"country":"ITA","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.532,"rtpt_won":0.376,"elo":1748},"clay":{"svpt_won":0.540,"rtpt_won":0.384,"elo":1768},"grass":{"svpt_won":0.524,"rtpt_won":0.362,"elo":1723}},
    "leonie_kung":         {"full_name":"Leonie Kung","hand":"R","rank":85,"country":"SUI","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.530,"rtpt_won":0.374,"elo":1740},"clay":{"svpt_won":0.525,"rtpt_won":0.366,"elo":1720},"grass":{"svpt_won":0.525,"rtpt_won":0.360,"elo":1715}},
    "anna_schmiedlova":    {"full_name":"Anna Schmiedlova","hand":"R","rank":85,"country":"SVK","birth_year":1994,"backhand":"2h",
        "hard":{"svpt_won":0.530,"rtpt_won":0.374,"elo":1740},"clay":{"svpt_won":0.538,"rtpt_won":0.382,"elo":1758},"grass":{"svpt_won":0.522,"rtpt_won":0.360,"elo":1715}},
    "aliaksandra_sasnovich": {"full_name":"Aliaksandra Sasnovich","hand":"R","rank":85,"country":"BLR","birth_year":1994,"backhand":"2h",
        "hard":{"svpt_won":0.530,"rtpt_won":0.374,"elo":1740},"clay":{"svpt_won":0.528,"rtpt_won":0.370,"elo":1732},"grass":{"svpt_won":0.526,"rtpt_won":0.362,"elo":1715}},
    "petra_martic":        {"full_name":"Petra Martic","hand":"R","rank":85,"country":"CRO","birth_year":1991,"backhand":"2h",
        "hard":{"svpt_won":0.530,"rtpt_won":0.374,"elo":1740},"clay":{"svpt_won":0.542,"rtpt_won":0.388,"elo":1772},"grass":{"svpt_won":0.522,"rtpt_won":0.360,"elo":1715}},
    "storm_hunter":        {"full_name":"Storm Hunter","hand":"R","rank":85,"country":"AUS","birth_year":1994,"backhand":"2h",
        "hard":{"svpt_won":0.530,"rtpt_won":0.374,"elo":1740},"clay":{"svpt_won":0.525,"rtpt_won":0.366,"elo":1720},"grass":{"svpt_won":0.535,"rtpt_won":0.372,"elo":1758}},
    "greet_minnen":        {"full_name":"Greet Minnen","hand":"R","rank":88,"country":"BEL","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.527,"rtpt_won":0.371,"elo":1732},"clay":{"svpt_won":0.530,"rtpt_won":0.374,"elo":1740},"grass":{"svpt_won":0.522,"rtpt_won":0.358,"elo":1707}},
    "oksana_selekhmeteva": {"full_name":"Oksana Selekhmeteva","hand":"R","rank":88,"country":"RUS","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.527,"rtpt_won":0.371,"elo":1732},"clay":{"svpt_won":0.525,"rtpt_won":0.368,"elo":1722},"grass":{"svpt_won":0.522,"rtpt_won":0.358,"elo":1707}},
    "viktoriya_tomova":    {"full_name":"Viktoriya Tomova","hand":"R","rank":88,"country":"BUL","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.527,"rtpt_won":0.371,"elo":1732},"clay":{"svpt_won":0.532,"rtpt_won":0.377,"elo":1745},"grass":{"svpt_won":0.522,"rtpt_won":0.358,"elo":1707}},
    "daria_snigur":        {"full_name":"Daria Snigur","hand":"R","rank":90,"country":"UKR","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.526,"rtpt_won":0.370,"elo":1726},"clay":{"svpt_won":0.522,"rtpt_won":0.365,"elo":1708},"grass":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1701}},
    "harriet_dart":        {"full_name":"Harriet Dart","hand":"R","rank":90,"country":"GBR","birth_year":1996,"backhand":"2h",
        "hard":{"svpt_won":0.526,"rtpt_won":0.370,"elo":1726},"clay":{"svpt_won":0.520,"rtpt_won":0.362,"elo":1706},"grass":{"svpt_won":0.532,"rtpt_won":0.374,"elo":1748}},
    "elsa_jacquemot":      {"full_name":"Elsa Jacquemot","hand":"R","rank":90,"country":"FRA","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.526,"rtpt_won":0.370,"elo":1726},"clay":{"svpt_won":0.528,"rtpt_won":0.374,"elo":1732},"grass":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1701}},
    "diana_marcinkevica":  {"full_name":"Diana Marcinkevica","hand":"R","rank":90,"country":"LAT","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.526,"rtpt_won":0.370,"elo":1726},"clay":{"svpt_won":0.522,"rtpt_won":0.365,"elo":1708},"grass":{"svpt_won":0.522,"rtpt_won":0.362,"elo":1701}},
    "emma_lene_norsgaard": {"full_name":"Emma Lene Norsgaard","hand":"R","rank":90,"country":"DEN","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.526,"rtpt_won":0.370,"elo":1726},"clay":{"svpt_won":0.525,"rtpt_won":0.368,"elo":1718},"grass":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1701}},
    "tereza_martincova":   {"full_name":"Tereza Martincova","hand":"R","rank":90,"country":"CZE","birth_year":1994,"backhand":"2h",
        "hard":{"svpt_won":0.526,"rtpt_won":0.370,"elo":1726},"clay":{"svpt_won":0.528,"rtpt_won":0.372,"elo":1732},"grass":{"svpt_won":0.526,"rtpt_won":0.368,"elo":1718}},
    "tatjana_maria":       {"full_name":"Tatjana Maria","hand":"R","rank":92,"country":"GER","birth_year":1987,"backhand":"2h",
        "hard":{"svpt_won":0.522,"rtpt_won":0.368,"elo":1720},"clay":{"svpt_won":0.520,"rtpt_won":0.365,"elo":1712},"grass":{"svpt_won":0.530,"rtpt_won":0.375,"elo":1745}},
    "dominika_salkova":    {"full_name":"Dominika Salkova","hand":"R","rank":92,"country":"CZE","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.525,"rtpt_won":0.369,"elo":1720},"clay":{"svpt_won":0.522,"rtpt_won":0.365,"elo":1700},"grass":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1695}},
    "olga_danilovic":      {"full_name":"Olga Danilovic","hand":"R","rank":92,"country":"SRB","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.525,"rtpt_won":0.369,"elo":1720},"clay":{"svpt_won":0.530,"rtpt_won":0.375,"elo":1734},"grass":{"svpt_won":0.518,"rtpt_won":0.355,"elo":1695}},
    "hailey_baptiste":     {"full_name":"Hailey Baptiste","hand":"R","rank":92,"country":"USA","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.548,"rtpt_won":0.368,"elo":1748},"clay":{"svpt_won":0.535,"rtpt_won":0.355,"elo":1718},"grass":{"svpt_won":0.542,"rtpt_won":0.362,"elo":1738}},
    "ocean_dodin":         {"full_name":"Ocean Dodin","hand":"R","rank":95,"country":"FRA","birth_year":1996,"backhand":"2h",
        "hard":{"svpt_won":0.523,"rtpt_won":0.367,"elo":1712},"clay":{"svpt_won":0.520,"rtpt_won":0.362,"elo":1692},"grass":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1687}},
    "suzan_lamens":        {"full_name":"Suzan Lamens","hand":"R","rank":95,"country":"NED","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.523,"rtpt_won":0.367,"elo":1712},"clay":{"svpt_won":0.520,"rtpt_won":0.362,"elo":1692},"grass":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1687}},
    "panna_udvardy":       {"full_name":"Panna Udvardy","hand":"R","rank":95,"country":"HUN","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.523,"rtpt_won":0.367,"elo":1712},"clay":{"svpt_won":0.522,"rtpt_won":0.365,"elo":1698},"grass":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1687}},
    "georgia_masarova":    {"full_name":"Georgia Masarova","hand":"R","rank":95,"country":"ESP","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.523,"rtpt_won":0.367,"elo":1712},"clay":{"svpt_won":0.535,"rtpt_won":0.380,"elo":1742},"grass":{"svpt_won":0.516,"rtpt_won":0.355,"elo":1687}},
    "nuria_parrizas_diaz": {"full_name":"Nuria Parrizas-Diaz","hand":"R","rank":95,"country":"ESP","birth_year":1991,"backhand":"2h",
        "hard":{"svpt_won":0.522,"rtpt_won":0.366,"elo":1708},"clay":{"svpt_won":0.542,"rtpt_won":0.385,"elo":1758},"grass":{"svpt_won":0.514,"rtpt_won":0.352,"elo":1683}},
    "julia_grabher":       {"full_name":"Julia Grabher","hand":"R","rank":95,"country":"AUT","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.523,"rtpt_won":0.367,"elo":1712},"clay":{"svpt_won":0.520,"rtpt_won":0.362,"elo":1692},"grass":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1687}},
    "irina_begu":          {"full_name":"Irina Begu","hand":"R","rank":98,"country":"ROU","birth_year":1990,"backhand":"2h",
        "hard":{"svpt_won":0.521,"rtpt_won":0.365,"elo":1704},"clay":{"svpt_won":0.528,"rtpt_won":0.374,"elo":1722},"grass":{"svpt_won":0.516,"rtpt_won":0.352,"elo":1679}},
    "yafan_wang":          {"full_name":"Yafan Wang","hand":"R","rank":98,"country":"CHN","birth_year":1995,"backhand":"2h",
        "hard":{"svpt_won":0.521,"rtpt_won":0.365,"elo":1704},"clay":{"svpt_won":0.520,"rtpt_won":0.362,"elo":1694},"grass":{"svpt_won":0.515,"rtpt_won":0.350,"elo":1679}},
    "jodie_burrage":       {"full_name":"Jodie Burrage","hand":"R","rank":98,"country":"GBR","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.521,"rtpt_won":0.365,"elo":1704},"clay":{"svpt_won":0.518,"rtpt_won":0.360,"elo":1684},"grass":{"svpt_won":0.528,"rtpt_won":0.372,"elo":1728}},
    "ana_bogdan":          {"full_name":"Ana Bogdan","hand":"R","rank":100,"country":"ROU","birth_year":1992,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.364,"elo":1698},"clay":{"svpt_won":0.528,"rtpt_won":0.372,"elo":1718},"grass":{"svpt_won":0.514,"rtpt_won":0.350,"elo":1673}},
    "qiang_wang":          {"full_name":"Qiang Wang","hand":"R","rank":100,"country":"CHN","birth_year":1992,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.364,"elo":1698},"clay":{"svpt_won":0.518,"rtpt_won":0.360,"elo":1682},"grass":{"svpt_won":0.514,"rtpt_won":0.350,"elo":1673}},
    "nadia_podoroska":     {"full_name":"Nadia Podoroska","hand":"R","rank":100,"country":"ARG","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.364,"elo":1698},"clay":{"svpt_won":0.542,"rtpt_won":0.385,"elo":1748},"grass":{"svpt_won":0.512,"rtpt_won":0.348,"elo":1673}},
    "tamara_zidansek":     {"full_name":"Tamara Zidansek","hand":"R","rank":100,"country":"SLO","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.364,"elo":1698},"clay":{"svpt_won":0.530,"rtpt_won":0.375,"elo":1718},"grass":{"svpt_won":0.512,"rtpt_won":0.348,"elo":1673}},
    "dalma_galfi":         {"full_name":"Dalma Galfi","hand":"R","rank":100,"country":"HUN","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.364,"elo":1698},"clay":{"svpt_won":0.528,"rtpt_won":0.372,"elo":1718},"grass":{"svpt_won":0.514,"rtpt_won":0.350,"elo":1673}},
    "rebecca_sramkova":    {"full_name":"Rebecca Sramkova","hand":"R","rank":102,"country":"SVK","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.363,"elo":1692},"clay":{"svpt_won":0.525,"rtpt_won":0.368,"elo":1706},"grass":{"svpt_won":0.514,"rtpt_won":0.348,"elo":1667}},
    "emina_bektas":        {"full_name":"Emina Bektas","hand":"R","rank":105,"country":"USA","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.361,"elo":1684},"clay":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1672},"grass":{"svpt_won":0.514,"rtpt_won":0.346,"elo":1659}},
    "renata_zarazua":      {"full_name":"Renata Zarazua","hand":"R","rank":105,"country":"MEX","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.361,"elo":1684},"clay":{"svpt_won":0.525,"rtpt_won":0.368,"elo":1698},"grass":{"svpt_won":0.512,"rtpt_won":0.346,"elo":1659}},
    "nao_hibino":          {"full_name":"Nao Hibino","hand":"R","rank":105,"country":"JPN","birth_year":1990,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.361,"elo":1684},"clay":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1672},"grass":{"svpt_won":0.514,"rtpt_won":0.346,"elo":1659}},
    "viktoriya_golubic":   {"full_name":"Viktoriya Golubic","hand":"R","rank":105,"country":"SUI","birth_year":1993,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.361,"elo":1684},"clay":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1672},"grass":{"svpt_won":0.526,"rtpt_won":0.364,"elo":1702}},
    "mia_pohankova":       {"full_name":"Mia Pohankova","hand":"R","rank":110,"country":"CZE","birth_year":2005,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1670},"clay":{"svpt_won":0.520,"rtpt_won":0.355,"elo":1655},"grass":{"svpt_won":0.514,"rtpt_won":0.344,"elo":1645}},
    "sofya_lansere":       {"full_name":"Sofya Lansere","hand":"R","rank":110,"country":"RUS","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1670},"clay":{"svpt_won":0.520,"rtpt_won":0.355,"elo":1655},"grass":{"svpt_won":0.514,"rtpt_won":0.344,"elo":1645}},
    "nadiia_kichenok":     {"full_name":"Nadiia Kichenok","hand":"R","rank":110,"country":"UKR","birth_year":1992,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1670},"clay":{"svpt_won":0.520,"rtpt_won":0.355,"elo":1655},"grass":{"svpt_won":0.514,"rtpt_won":0.344,"elo":1645}},
    "katerina_baindl":     {"full_name":"Katerina Baindl","hand":"R","rank":110,"country":"UKR","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1670},"clay":{"svpt_won":0.520,"rtpt_won":0.355,"elo":1655},"grass":{"svpt_won":0.514,"rtpt_won":0.344,"elo":1645}},
    "fiona_ferro":         {"full_name":"Fiona Ferro","hand":"R","rank":110,"country":"FRA","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1670},"clay":{"svpt_won":0.528,"rtpt_won":0.366,"elo":1688},"grass":{"svpt_won":0.514,"rtpt_won":0.344,"elo":1645}},
    "irene_burillo_escorihuela": {"full_name":"Irene Burillo Escorihuela","hand":"R","rank":110,"country":"ESP","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1670},"clay":{"svpt_won":0.530,"rtpt_won":0.368,"elo":1692},"grass":{"svpt_won":0.512,"rtpt_won":0.344,"elo":1645}},
    "valentini_grammatikopoulou": {"full_name":"Valentini Grammatikopoulou","hand":"R","rank":112,"country":"GRE","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.357,"elo":1664},"clay":{"svpt_won":0.525,"rtpt_won":0.362,"elo":1678},"grass":{"svpt_won":0.514,"rtpt_won":0.342,"elo":1639}},
    "sara_errani":         {"full_name":"Sara Errani","hand":"R","rank":115,"country":"ITA","birth_year":1987,"backhand":"2h",
        "hard":{"svpt_won":0.512,"rtpt_won":0.356,"elo":1645},"clay":{"svpt_won":0.535,"rtpt_won":0.382,"elo":1705},"grass":{"svpt_won":0.506,"rtpt_won":0.342,"elo":1620}},
    "ajla_tomljanovic":    {"full_name":"Ajla Tomljanovic","hand":"R","rank":115,"country":"AUS","birth_year":1993,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.355,"elo":1656},"clay":{"svpt_won":0.518,"rtpt_won":0.350,"elo":1638},"grass":{"svpt_won":0.522,"rtpt_won":0.355,"elo":1662}},
    "yuki_naito":          {"full_name":"Yuki Naito","hand":"R","rank":115,"country":"JPN","birth_year":1996,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.355,"elo":1656},"clay":{"svpt_won":0.520,"rtpt_won":0.352,"elo":1644},"grass":{"svpt_won":0.514,"rtpt_won":0.342,"elo":1631}},
    "sloane_stephens":     {"full_name":"Sloane Stephens","hand":"R","rank":120,"country":"USA","birth_year":1993,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.352,"elo":1642},"clay":{"svpt_won":0.524,"rtpt_won":0.358,"elo":1655},"grass":{"svpt_won":0.515,"rtpt_won":0.342,"elo":1617}},
    "anastasija_sevastova": {"full_name":"Anastasija Sevastova","hand":"R","rank":120,"country":"LAT","birth_year":1990,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.352,"elo":1642},"clay":{"svpt_won":0.520,"rtpt_won":0.348,"elo":1628},"grass":{"svpt_won":0.522,"rtpt_won":0.352,"elo":1648}},
    "aliona_bolsova":      {"full_name":"Aliona Bolsova","hand":"R","rank":120,"country":"ESP","birth_year":1993,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.352,"elo":1642},"clay":{"svpt_won":0.530,"rtpt_won":0.362,"elo":1665},"grass":{"svpt_won":0.514,"rtpt_won":0.340,"elo":1617}},
    "katie_swan":          {"full_name":"Katie Swan","hand":"R","rank":120,"country":"GBR","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.352,"elo":1642},"clay":{"svpt_won":0.518,"rtpt_won":0.346,"elo":1625},"grass":{"svpt_won":0.526,"rtpt_won":0.358,"elo":1662}},
    "kaia_kanepi":         {"full_name":"Kaia Kanepi","hand":"R","rank":120,"country":"EST","birth_year":1985,"backhand":"2h",
        "hard":{"svpt_won":0.556,"rtpt_won":0.360,"elo":1720},"clay":{"svpt_won":0.540,"rtpt_won":0.348,"elo":1680},"grass":{"svpt_won":0.552,"rtpt_won":0.358,"elo":1708}},
    "polona_hercog":       {"full_name":"Polona Hercog","hand":"R","rank":120,"country":"SLO","birth_year":1991,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.352,"elo":1642},"clay":{"svpt_won":0.528,"rtpt_won":0.362,"elo":1660},"grass":{"svpt_won":0.515,"rtpt_won":0.342,"elo":1617}},
    # ── Ranks 88-300 (qualifying / lower-ranked WTA players) ──────────────────
    "heather_watson": {"full_name":"Heather Watson","hand":"R","rank":88,"country":"GBR","birth_year":1992,"backhand":"2h",
        "hard":{"svpt_won":0.527,"rtpt_won":0.371,"elo":1732},"clay":{"svpt_won":0.522,"rtpt_won":0.365,"elo":1712},"grass":{"svpt_won":0.532,"rtpt_won":0.375,"elo":1752}},
    "clervie_ngounoue": {"full_name":"Clervie Ngounoue","hand":"R","rank":90,"country":"FRA","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.526,"rtpt_won":0.370,"elo":1726},"clay":{"svpt_won":0.522,"rtpt_won":0.366,"elo":1706},"grass":{"svpt_won":0.520,"rtpt_won":0.362,"elo":1696}},
    "mai_hontama": {"full_name":"Mai Hontama","hand":"R","rank":100,"country":"JPN","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.364,"elo":1698},"clay":{"svpt_won":0.520,"rtpt_won":0.360,"elo":1678},"grass":{"svpt_won":0.520,"rtpt_won":0.356,"elo":1668}},
    "lucie_havlickova": {"full_name":"Lucie Havlickova","hand":"R","rank":103,"country":"CZE","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.362,"elo":1690},"clay":{"svpt_won":0.520,"rtpt_won":0.362,"elo":1700},"grass":{"svpt_won":0.520,"rtpt_won":0.358,"elo":1660}},
    "marina_bassols_ribera": {"full_name":"Marina Bassols Ribera","hand":"R","rank":108,"country":"ESP","birth_year":1999,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.359,"elo":1676},"clay":{"svpt_won":0.520,"rtpt_won":0.362,"elo":1706},"grass":{"svpt_won":0.520,"rtpt_won":0.355,"elo":1646}},
    "leolia_jeanjean": {"full_name":"Leolia Jeanjean","hand":"R","rank":108,"country":"FRA","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.359,"elo":1676},"clay":{"svpt_won":0.520,"rtpt_won":0.360,"elo":1686},"grass":{"svpt_won":0.520,"rtpt_won":0.355,"elo":1646}},
    "caroline_dolehide": {"full_name":"Caroline Dolehide","hand":"L","rank":122,"country":"USA","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.351,"elo":1636},"clay":{"svpt_won":0.520,"rtpt_won":0.347,"elo":1616},"grass":{"svpt_won":0.520,"rtpt_won":0.347,"elo":1606}},
    "anna_lena_friedsam": {"full_name":"Anna-Lena Friedsam","hand":"L","rank":122,"country":"GER","birth_year":1994,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.351,"elo":1636},"clay":{"svpt_won":0.520,"rtpt_won":0.352,"elo":1646},"grass":{"svpt_won":0.520,"rtpt_won":0.347,"elo":1606}},
    "robin_montgomery": {"full_name":"Robin Montgomery","hand":"R","rank":128,"country":"USA","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.347,"elo":1619},"clay":{"svpt_won":0.520,"rtpt_won":0.343,"elo":1599},"grass":{"svpt_won":0.520,"rtpt_won":0.343,"elo":1589}},
    "viktoria_hruncakova": {"full_name":"Viktoria Hruncakova","hand":"R","rank":130,"country":"SVK","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.346,"elo":1614},"clay":{"svpt_won":0.520,"rtpt_won":0.343,"elo":1594},"grass":{"svpt_won":0.520,"rtpt_won":0.342,"elo":1584}},
    "mariam_bolkvadze": {"full_name":"Mariam Bolkvadze","hand":"R","rank":133,"country":"GEO","birth_year":1998,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.344,"elo":1606},"clay":{"svpt_won":0.520,"rtpt_won":0.341,"elo":1586},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1576}},
    "xiaodi_you": {"full_name":"Xiaodi You","hand":"R","rank":148,"country":"CHN","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1564},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1544},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1534}},
    "maria_lourdes_carle": {"full_name":"Maria Lourdes Carle","hand":"R","rank":150,"country":"ARG","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1558},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1560},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530}},
    "guo_hanyu": {"full_name":"Guo Hanyu","hand":"R","rank":155,"country":"CHN","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1550},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530}},
    "mona_barthel": {"full_name":"Mona Barthel","hand":"R","rank":155,"country":"GER","birth_year":1990,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1550},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530}},
    "yi_zhou": {"full_name":"Yi Zhou","hand":"R","rank":175,"country":"CHN","birth_year":1997,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1550},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530}},
    "whitney_osuigwe": {"full_name":"Whitney Osuigwe","hand":"R","rank":170,"country":"USA","birth_year":2001,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1550},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530}},
    "maria_timofeeva": {"full_name":"Maria Timofeeva","hand":"R","rank":190,"country":"RUS","birth_year":2003,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1550},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530}},
    "lin_zhu": {"full_name":"Lin Zhu","hand":"R","rank":192,"country":"CHN","birth_year":2000,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1550},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530}},
    "akasha_urhobo": {"full_name":"Akasha Urhobo","hand":"R","rank":220,"country":"GBR","birth_year":2005,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1550},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530},"grass":{"svpt_won":0.522,"rtpt_won":0.340,"elo":1565}},
    "lola_radivojevic": {"full_name":"Lola Radivojevic","hand":"R","rank":248,"country":"SRB","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1550},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530}},
    "tiantsoa_sarah_rajaonah": {"full_name":"Tiantsoa Sarah Rajaonah","hand":"R","rank":275,"country":"MAD","birth_year":2002,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1550},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530}},
    "carol_young_suh_lee": {"full_name":"Carol Young Suh Lee","hand":"R","rank":295,"country":"USA","birth_year":2004,"backhand":"2h",
        "hard":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1550},"clay":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530},"grass":{"svpt_won":0.520,"rtpt_won":0.340,"elo":1530}},
}

# ─────────────────────────────────────────────────────────────────────────────
# HEAD-TO-HEAD RECORDS
# ─────────────────────────────────────────────────────────────────────────────
H2H: Dict[Tuple[str, str], Tuple[int, int]] = {
    ("djokovic",  "alcaraz"):   (2,  5),
    ("djokovic",  "sinner"):    (5,  2),
    ("alcaraz",   "sinner"):    (5,  4),
    ("djokovic",  "medvedev"):  (12, 5),
    ("djokovic",  "zverev"):    (10, 4),
    ("alcaraz",   "zverev"):    (4,  5),
    ("sinner",    "medvedev"):  (8,  4),
    ("sinner",    "zverev"):    (5,  4),
    ("swiatek",   "sabalenka"): (16, 9),
    ("swiatek",   "gauff"):     (12, 6),
    ("sabalenka", "gauff"):     (7,  5),
    ("swiatek",   "rybakina"):  (8,  6),
    ("sabalenka", "rybakina"):  (7,  4),
    ("gauff",     "rybakina"):  (4,  5),
    ("swiatek",   "keys"):      (9,  3),
    ("sabalenka", "keys"):      (6,  3),
}

# Surface-specific H2H (takes priority when ≥3 matches)
H2H_SURFACE: Dict[Tuple[str, str, str], Tuple[int, int]] = {
    ("djokovic",  "alcaraz",  "clay"):   (1, 3),
    ("djokovic",  "alcaraz",  "hard"):   (1, 2),
    ("djokovic",  "alcaraz",  "grass"):  (0, 1),
    ("djokovic",  "sinner",   "hard"):   (4, 2),
    ("djokovic",  "sinner",   "clay"):   (1, 0),
    ("alcaraz",   "sinner",   "clay"):   (3, 2),
    ("alcaraz",   "sinner",   "hard"):   (2, 2),
    ("alcaraz",   "sinner",   "grass"):  (1, 0),
    ("sinner",    "medvedev", "hard"):   (6, 3),
    ("swiatek",   "sabalenka","clay"):   (10, 3),
    ("swiatek",   "sabalenka","hard"):   (6,  6),
    ("swiatek",   "rybakina", "clay"):   (5,  1),
    ("swiatek",   "rybakina", "hard"):   (3,  5),
}

# BO5 specialist adjustment (Grand Slam format only)
BO5_SPECIALIST: Dict[str, float] = {
    "djokovic": +0.025,
    "sinner":   +0.018,
    "alcaraz":  +0.012,
    "zverev":   +0.008,
    "medvedev": +0.005,
    "ruud":     -0.012,
    "rublev":   -0.010,
    "tsitsipas":-0.008,
    "fritz":    -0.005,
    "de_minaur":-0.006,
}

KELLY_BY_TIER: Dict[str, float] = {"A": 0.30, "B": 0.25, "C": 0.20}

SURFACE_PT_ADJ: Dict[str, float] = {
    "hard":    0.000,
    "clay":   -0.020,
    "grass":  +0.022,
    "carpet": +0.015,
}

TOUR_META: Dict[str, dict] = {
    "grand_slam":  {"name": "大滿貫",    "best_of": 5},
    "masters1000": {"name": "大師賽",    "best_of": 3},
    "wta1000":     {"name": "WTA千人賽", "best_of": 3},
    "atp500":      {"name": "ATP 500",   "best_of": 3},
    "wta500":      {"name": "WTA 500",   "best_of": 3},
    "atp250":      {"name": "ATP 250",   "best_of": 3},
    "wta250":      {"name": "WTA 250",   "best_of": 3},
    "challenger":  {"name": "挑戰賽",    "best_of": 3},
}

# ─── v4.0 PLAYSTYLE MATRIX ───────────────────────────────────────────────────
PLAYSTYLE: Dict[str, str] = {
    # ATP
    "djokovic": "counter_puncher", "alcaraz": "all_court",
    "sinner": "aggressive_baseliner", "medvedev": "defensive_baseliner",
    "zverev": "aggressive_baseliner", "rublev": "aggressive_baseliner",
    "tsitsipas": "all_court", "fritz": "big_server",
    "de_minaur": "counter_puncher", "hurkacz": "big_server",
    "dimitrov": "all_court", "paul": "aggressive_baseliner",
    "auger_aliassime": "big_server", "musetti": "all_court",
    "tiafoe": "aggressive_baseliner", "berrettini": "big_server",
    "ruud": "defensive_baseliner", "draper": "aggressive_baseliner",
    "shelton": "big_server", "khachanov": "aggressive_baseliner",
    "bublik": "big_server", "humbert": "aggressive_baseliner",
    "jarry": "big_server", "cobolli": "aggressive_baseliner",
    "carreno_busta": "counter_puncher", "tirante": "counter_puncher",
    # WTA
    "swiatek": "aggressive_baseliner", "sabalenka": "aggressive_baseliner",
    "gauff": "all_court", "rybakina": "big_server",
    "pegula": "defensive_baseliner", "keys": "aggressive_baseliner",
    "zheng": "aggressive_baseliner", "paolini": "counter_puncher",
    "navarro": "all_court", "krejcikova": "all_court",
    "sakkari": "aggressive_baseliner", "kasatkina": "all_court",
    "kvitova": "big_server", "haddad_maia": "aggressive_baseliner",
    "kostyuk": "aggressive_baseliner", "bencic": "all_court",
    "collins": "aggressive_baseliner",
    # ATP — additional
    "rune": "aggressive_baseliner", "lehecka": "big_server",
    "korda": "big_server", "fils": "aggressive_baseliner",
    "sonego": "aggressive_baseliner", "arnaldi": "aggressive_baseliner",
    "griekspoor": "big_server", "baez": "counter_puncher",
    "etcheverry": "counter_puncher", "tabilo": "aggressive_baseliner",
    "cerundolo": "aggressive_baseliner", "juan_cerundolo": "aggressive_baseliner",
    "davidovich": "counter_puncher", "navone": "counter_puncher",
    "fonseca": "aggressive_baseliner", "mensik": "big_server",
    "borges": "defensive_baseliner", "kecmanovic": "aggressive_baseliner",
    "marozsan": "defensive_baseliner", "shapovalov": "aggressive_baseliner",
    "norrie": "counter_puncher", "mannarino": "defensive_baseliner",
    "moutet": "counter_puncher", "svajda": "big_server",
    "jodar": "counter_puncher", "coric": "aggressive_baseliner",
    "rinderknech": "big_server", "darderi": "aggressive_baseliner",
    "popyrin": "big_server", "wawrinka": "aggressive_baseliner",
    "gaston": "counter_puncher",
    # WTA — additional
    "osaka": "aggressive_baseliner", "shnaider": "aggressive_baseliner",
    "kalinskaya": "aggressive_baseliner", "andreeva": "counter_puncher",
    "svitolina": "counter_puncher", "samsonova": "aggressive_baseliner",
    "badosa": "aggressive_baseliner", "potapova": "aggressive_baseliner",
    "jabeur": "all_court", "yastremska": "aggressive_baseliner",
    "garcia": "big_server", "muchova": "all_court",
    "ostapenko": "aggressive_baseliner", "noskova": "big_server",
    "frech": "defensive_baseliner", "cirstea": "aggressive_baseliner",
    "parry": "defensive_baseliner", "chwalinska": "aggressive_baseliner",
    "vekic": "big_server", "kudermetova": "aggressive_baseliner",
    "raducanu": "all_court", "tauson": "aggressive_baseliner",
    "blinkova": "aggressive_baseliner", "alexandrova": "big_server",
    "xinyu_wang": "defensive_baseliner",
}

STYLE_MATCHUP_ADJ: Dict[Tuple[str, str], float] = {
    ("big_server",           "counter_puncher"):      +0.018,
    ("big_server",           "defensive_baseliner"):  +0.022,
    ("big_server",           "aggressive_baseliner"): +0.010,
    ("aggressive_baseliner", "defensive_baseliner"):  +0.015,
    ("aggressive_baseliner", "counter_puncher"):      -0.010,
    ("counter_puncher",      "aggressive_baseliner"): +0.010,
    ("counter_puncher",      "big_server"):           -0.018,
    ("counter_puncher",      "defensive_baseliner"):  +0.005,
    ("all_court",            "big_server"):           +0.005,
    ("all_court",            "defensive_baseliner"):  +0.008,
    ("all_court",            "counter_puncher"):      +0.003,
    ("defensive_baseliner",  "aggressive_baseliner"): -0.015,
    ("defensive_baseliner",  "big_server"):           -0.022,
}

STYLE_SURFACE_MOD: Dict[str, float] = {
    "hard": 1.0, "clay": 1.3, "grass": 1.4, "carpet": 1.1,
}

# Dynamic model weights [elo, markov, hb, adv] per surface
DYNAMIC_WEIGHTS: Dict[str, Dict[str, float]] = {
    "hard":   {"elo": 0.25, "markov": 0.25, "hb": 0.20, "adv": 0.30},
    "clay":   {"elo": 0.18, "markov": 0.22, "hb": 0.25, "adv": 0.35},
    "grass":  {"elo": 0.30, "markov": 0.28, "hb": 0.22, "adv": 0.20},
    "carpet": {"elo": 0.28, "markov": 0.27, "hb": 0.22, "adv": 0.23},
}
# WTA: return/rally factor boosted
WTA_WEIGHT_ADJ: Dict[str, float] = {
    "elo": -0.03, "markov": -0.02, "hb": +0.02, "adv": +0.03,
}

SURFACE_TRANSITION_PENALTY: Dict[Tuple[str, str], float] = {
    ("clay",  "grass"):  -0.020,
    ("hard",  "grass"):  -0.012,
    ("grass", "clay"):   -0.010,
    ("grass", "hard"):   -0.006,
    ("clay",  "hard"):   -0.005,
    ("hard",  "clay"):   -0.008,
}
SURFACE_TRANSITION_WINDOW = 21  # days

PUBLIC_BIAS_THRESHOLD = 0.10   # raised: need stronger disagreement before fading
PUBLIC_BIAS_FADE      = 0.015  # reduced: smaller nudge to avoid overriding market

INJURY_SERVE_DROP_THRESH = 0.05
INJURY_STREAK_PENALTY    = -0.015

# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CACHES
# ─────────────────────────────────────────────────────────────────────────────
_LIVE_ELO:          Dict[str, dict]  = {}
_LIVE_FORM:         Dict[str, float] = {}
_RECENT_STATS:      Dict[str, dict]  = {}
_INJURIES:          Dict[str, str]   = {}
_SACKMANN_PROFILES: Dict[str, dict]  = {}
_ODDS_PREV:         Dict[str, dict]  = {}  # previous run odds for movement detection
_DYNAMIC_H2H:         Dict[Tuple[str, str, str], Tuple[int, int]] = {}  # (p1,p2,surf)→(w1,w2)
_DYNAMIC_H2H_OVERALL: Dict[Tuple[str, str],       Tuple[int, int]] = {}  # (p1,p2)→(w1,w2)
_LIVE_RANKS: Dict[str, Tuple[int, str]] = {}  # norm_key → (current_rank, "atp"|"wta")

# ─────────────────────────────────────────────────────────────────────────────
# MARKOV CHAIN TENNIS MODEL
# ─────────────────────────────────────────────────────────────────────────────

def game_win_prob(p: float) -> float:
    q = 1.0 - p
    d = p * p + q * q
    if d < 1e-9:
        return 0.5
    p_win_deuce   = p * p / d
    no_deuce      = p ** 4 * (1.0 + 4.0 * q + 10.0 * q ** 2)
    p_reach_deuce = 20.0 * (p ** 3) * (q ** 3)
    return no_deuce + p_reach_deuce * p_win_deuce


def set_win_prob(p1_sv: float, p2_sv: float,
                first_server: int = 1, tiebreak: bool = True) -> float:
    g1 = game_win_prob(p1_sv)
    g2 = game_win_prob(p2_sv)
    tb = (g1 + 1.0 - g2) / 2.0
    memo: Dict[tuple, float] = {}

    def dp(s1: int, s2: int, srv: int) -> float:
        if s1 >= 6 and s1 - s2 >= 2:
            return 1.0
        if s2 >= 6 and s2 - s1 >= 2:
            return 0.0
        if tiebreak and s1 == 6 and s2 == 6:
            return tb
        key = (s1, s2, srv)
        if key in memo:
            return memo[key]
        p_win = g1 if srv == 1 else (1.0 - g2)
        nxt   = 2 if srv == 1 else 1
        val   = p_win * dp(s1 + 1, s2, nxt) + (1.0 - p_win) * dp(s1, s2 + 1, nxt)
        memo[key] = val
        return val

    return dp(0, 0, first_server)


def match_win_prob(p1_sv: float, p2_sv: float, best_of: int = 3) -> float:
    need = (best_of + 1) // 2
    memo: Dict[tuple, float] = {}

    def dp(w1: int, w2: int, srv: int) -> float:
        if w1 == need:
            return 1.0
        if w2 == need:
            return 0.0
        key = (w1, w2, srv)
        if key in memo:
            return memo[key]
        ps  = set_win_prob(p1_sv, p2_sv, first_server=srv)
        nxt = 2 if srv == 1 else 1
        val = ps * dp(w1 + 1, w2, nxt) + (1.0 - ps) * dp(w1, w2 + 1, nxt)
        memo[key] = val
        return val

    return dp(0, 0, 1)


def expected_total_games(p1_sv: float, p2_sv: float,
                         best_of: int = 3, n: int = 4000) -> float:
    g1 = game_win_prob(p1_sv)
    g2 = game_win_prob(p2_sv)
    tb = (g1 + 1.0 - g2) / 2.0
    need  = (best_of + 1) // 2
    total = 0

    for _ in range(n):
        games = 0
        w1 = w2 = 0
        srv = 1
        while w1 < need and w2 < need:
            s1 = s2 = 0
            while True:
                p_win = g1 if srv == 1 else (1.0 - g2)
                if random.random() < p_win:
                    s1 += 1
                else:
                    s2 += 1
                games += 1
                srv = 2 if srv == 1 else 1
                if (s1 >= 6 and s1 - s2 >= 2) or (s2 >= 6 and s2 - s1 >= 2):
                    break
                if s1 == 6 and s2 == 6:
                    s1 = 7 if random.random() < tb else s1
                    s2 = 7 if s1 != 7 else s2
                    if s1 == 7 or s2 == 7:
                        games += 1
                    break
            if s1 > s2:
                w1 += 1
            else:
                w2 += 1
        total += games

    return total / n


# ─────────────────────────────────────────────────────────────────────────────
# ELO MODEL
# ─────────────────────────────────────────────────────────────────────────────
ELO_SCALE = 400.0


def elo_win_prob(elo1: float, elo2: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo2 - elo1) / ELO_SCALE))


def h2h_adj(p1: str, p2: str, surface: str = "") -> float:
    # 1. Dynamic H2H surface-specific (from Sackmann data, last 5 years)
    if surface and _DYNAMIC_H2H:
        cp1, cp2 = (p1, p2) if p1 <= p2 else (p2, p1)
        flipped  = (cp1 != p1)
        sw1, sw2 = _DYNAMIC_H2H.get((cp1, cp2, surface), (0, 0))
        if flipped:
            sw1, sw2 = sw2, sw1
        if sw1 + sw2 >= 3:
            return max(-0.05, min(0.05, (sw1 / (sw1 + sw2) - 0.5) * 0.10))

    # 2. Static H2H surface-specific
    if surface:
        if (p1, p2, surface) in H2H_SURFACE:
            sw1, sw2 = H2H_SURFACE[(p1, p2, surface)]
        elif (p2, p1, surface) in H2H_SURFACE:
            sw2, sw1 = H2H_SURFACE[(p2, p1, surface)]
        else:
            sw1, sw2 = 0, 0
        if sw1 + sw2 >= 3:
            return max(-0.05, min(0.05, (sw1 / (sw1 + sw2) - 0.5) * 0.10))

    # 3. Dynamic H2H overall
    if _DYNAMIC_H2H_OVERALL:
        cp1, cp2 = (p1, p2) if p1 <= p2 else (p2, p1)
        flipped  = (cp1 != p1)
        w1, w2   = _DYNAMIC_H2H_OVERALL.get((cp1, cp2), (0, 0))
        if flipped:
            w1, w2 = w2, w1
        if w1 + w2 >= 4:
            return max(-0.05, min(0.05, (w1 / (w1 + w2) - 0.5) * 0.10))

    # 4. Static H2H overall
    w1, w2 = 0, 0
    if (p1, p2) in H2H:
        w1, w2 = H2H[(p1, p2)]
    elif (p2, p1) in H2H:
        w2, w1 = H2H[(p2, p1)]
    total = w1 + w2
    if total < 4:
        return 0.0
    return max(-0.05, min(0.05, (w1 / total - 0.5) * 0.10))


# ─────────────────────────────────────────────────────────────────────────────
# FATIGUE & HOLD/BREAK MODELS
# ─────────────────────────────────────────────────────────────────────────────

def fatigue_score(days_rest: int, prev_minutes: float, sets_played: int) -> float:
    """Return 0–10 fatigue score. Higher = more fatigued."""
    score = 0.0
    if days_rest == 0:
        score += 4.0
    elif days_rest == 1:
        score += 2.5
    elif days_rest == 2:
        score += 1.0
    elif days_rest >= 6:
        score -= 1.0
    if prev_minutes > 180:
        score += 3.0
    elif prev_minutes > 120:
        score += 1.5
    if sets_played >= 4:
        score += 2.0
    elif sets_played == 3:
        score += 0.8
    return max(0.0, min(10.0, score))


def hold_break_win_prob(hold1: float, break1: float,
                        hold2: float, break2: float) -> float:
    """P(p1 wins) from dominance ratio of hold+break rates."""
    dom1 = (hold1 + break1) / 2.0
    dom2 = (hold2 + break2) / 2.0
    return dom1 / (dom1 + dom2) if (dom1 + dom2) > 1e-9 else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED ADJUSTMENT FUNCTIONS  (v3)
# ─────────────────────────────────────────────────────────────────────────────

def get_court_speed_adj(sport_key: str, tournament: str = "") -> float:
    """Additive boost/penalty to svpt_won for court speed / indoor."""
    t = (sport_key + " " + tournament).lower()
    base = 0.0
    if any(x in t for x in INDOOR_TOURNAMENTS):
        base += 0.012
    for name, adj in COURT_SPEED_ADJ.items():
        if name in t:
            base += adj
            break
    return max(-0.025, min(0.025, base))


def age_fatigue_mult(player_key: str) -> float:
    """Players over 28 accumulate fatigue faster; returns multiplier >= 1.0."""
    all_players = {**ATP_STATS, **WTA_STATS}
    by = all_players.get(player_key, {}).get("birth_year")
    if not by:
        return 1.0
    age = datetime.datetime.utcnow().year - by
    if age <= 28:
        return 1.0
    return min(1.6, 1.0 + (age - 28) * AGE_FATIGUE_SCALE)


def lefty_matchup_adj(p1_key: str, p2_key: str, surface: str) -> float:
    """Serve point bonus when left-hander serves against right-hander."""
    all_players = {**ATP_STATS, **WTA_STATS}
    h1 = all_players.get(p1_key, {}).get("hand", "R")
    h2 = all_players.get(p2_key, {}).get("hand", "R")
    if h1 == h2:
        return 0.0
    bonus = LEFTY_SERVE_BONUS
    if surface == "grass":
        bonus += LEFTY_GRASS_EXTRA
    return bonus if h1 == "L" else -bonus


def backhand_matchup_adj(p1_key: str, p2_key: str, surface: str) -> float:
    """1h backhand vulnerability vs heavy lefty topspin on clay."""
    if surface != "clay":
        return 0.0
    all_players = {**ATP_STATS, **WTA_STATS}
    bh1 = all_players.get(p1_key, {}).get("backhand", "2h")
    bh2 = all_players.get(p2_key, {}).get("backhand", "2h")
    h1  = all_players.get(p1_key, {}).get("hand", "R")
    h2  = all_players.get(p2_key, {}).get("hand", "R")
    adj = 0.0
    if bh1 == "1h" and h2 == "L":
        adj -= BH_TOPSPIN_VULN
    if bh2 == "1h" and h1 == "L":
        adj += BH_TOPSPIN_VULN
    return adj


def clutch_adj(p1_key: str, p2_key: str) -> float:
    """Tiebreak + break point save + deciding set record."""
    prof1 = _SACKMANN_PROFILES.get(p1_key, {})
    prof2 = _SACKMANN_PROFILES.get(p2_key, {})
    adj   = 0.0
    tb1 = prof1.get("tb_win_pct");  tb2 = prof2.get("tb_win_pct")
    if tb1 is not None and tb2 is not None:
        adj += (tb1 - tb2) * 0.10
    bp1 = prof1.get("bp_save_pct"); bp2 = prof2.get("bp_save_pct")
    if bp1 is not None and bp2 is not None:
        adj += (bp1 - bp2) * 0.06
    dc1 = prof1.get("deciding_pct"); dc2 = prof2.get("deciding_pct")
    if dc1 is not None and dc2 is not None:
        adj += (dc1 - dc2) * 0.06
    return max(-0.07, min(0.07, adj))


def df_penalty_adj(p1_key: str, p2_key: str, is_wta: bool) -> float:
    """Double fault rate differential; more impact in WTA."""
    prof1 = _SACKMANN_PROFILES.get(p1_key, {})
    prof2 = _SACKMANN_PROFILES.get(p2_key, {})
    df1 = prof1.get("df_rate"); df2 = prof2.get("df_rate")
    if df1 is None or df2 is None:
        return 0.0
    scale = 0.35 if is_wta else 0.22
    return max(-0.04, min(0.04, (df2 - df1) * scale))


def surface_form_adj(p1_key: str, p2_key: str, surface: str) -> float:
    """Surface-specific recent win rate differential."""
    prof1 = _SACKMANN_PROFILES.get(p1_key, {})
    prof2 = _SACKMANN_PROFILES.get(p2_key, {})
    sf1 = prof1.get("surface_form", {}).get(surface)
    sf2 = prof2.get("surface_form", {}).get(surface)
    if sf1 is None or sf2 is None:
        return 0.0
    # Clay form is more predictive than other surfaces — higher coeff/cap
    coeff = 0.18 if surface == "clay" else 0.12
    cap   = 0.055 if surface == "clay" else 0.040
    return max(-cap, min(cap, (sf1 - sf2) * coeff))


def ace_serve_adj(p1_key: str, p2_key: str) -> float:
    """Ace rate differential as additional serve dominance signal."""
    prof1 = _SACKMANN_PROFILES.get(p1_key, {})
    prof2 = _SACKMANN_PROFILES.get(p2_key, {})
    a1 = prof1.get("ace_rate"); a2 = prof2.get("ace_rate")
    if a1 is None or a2 is None:
        return 0.0
    return max(-0.025, min(0.025, (a1 - a2) * 0.40))


def altitude_adj(tournament: str, surface: str) -> float:
    """Higher altitude → ball travels faster; especially offsets clay slowness."""
    t = tournament.lower().replace(" ", "_").replace("-", "_")
    for city, alt_m in ALTITUDE_M.items():
        if city in t:
            if alt_m < 400:
                return 0.0
            base = min(0.020, (alt_m - 200) / 500 * 0.005)
            return round(base * (1.5 if surface == "clay" else 1.0), 4)
    return 0.0


def fetch_wind(lat: float, lon: float) -> Tuple[Optional[float], float]:
    """Wind speed km/h via free Open-Meteo API."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "current": "wind_speed_10m,precipitation",
                    "wind_speed_unit": "kmh", "forecast_days": 1},
            timeout=6,
        )
        r.raise_for_status()
        cur = r.json().get("current", {})
        return cur.get("wind_speed_10m"), cur.get("precipitation", 0.0)
    except Exception as e:
        log.debug("fetch_wind (%s,%s): %s", lat, lon, e)
        return None, 0.0


def wind_adj(tournament: str, surface: str, p1_key: str, p2_key: str) -> Tuple[float, float]:
    """
    Wind penalises heavy topspin/clay-court baseline play.
    Returns (prob_adj_for_p1, wind_kmh).
    Only applied for outdoor clay/grass.
    """
    if surface not in ("clay", "grass"):
        return 0.0, 0.0
    coords = None
    t = tournament.lower().replace(" ", "_").replace("-", "_")
    for name, (lat, lon) in TOURNAMENT_COORDS.items():
        if name in t or t in name:
            coords = (lat, lon)
            break
    if not coords:
        return 0.0, 0.0
    wind, rain = fetch_wind(*coords)
    if wind is None or wind < 15:
        return 0.0, wind or 0.0
    # Aggressive servers benefit more in wind; topspin players suffer
    a1 = _SACKMANN_PROFILES.get(p1_key, {}).get("ace_rate") or 0.06
    a2 = _SACKMANN_PROFILES.get(p2_key, {}).get("ace_rate") or 0.06
    factor = min(0.030, (wind - 15) * 0.0012)
    adj = (a1 - a2) * factor * 4.0
    return max(-0.025, min(0.025, adj)), round(wind, 1)


def win_streak_adj(p1_key: str, p2_key: str) -> float:
    """Recent win/loss streak as momentum signal. ±3% max."""
    prof1 = _SACKMANN_PROFILES.get(p1_key, {})
    prof2 = _SACKMANN_PROFILES.get(p2_key, {})
    s1 = prof1.get("win_streak", 0)
    s2 = prof2.get("win_streak", 0)

    def bonus(s: int) -> float:
        if s >= 5:  return 0.030
        if s >= 4:  return 0.022
        if s >= 3:  return 0.015
        if s <= -4: return -0.025
        if s <= -3: return -0.015
        return 0.0

    return max(-0.04, min(0.04, bonus(s1) - bonus(s2)))


def bo5_adj(p1_key: str, p2_key: str, best_of: int) -> float:
    """Grand Slam specialist advantage (best-of-5 only)."""
    if best_of != 5:
        return 0.0
    sp1 = BO5_SPECIALIST.get(p1_key, 0.0)
    sp2 = BO5_SPECIALIST.get(p2_key, 0.0)
    return max(-0.05, min(0.05, sp1 - sp2))


def first_serve_adj(p1_key: str, p2_key: str) -> float:
    """1st serve % differential → consistent server advantage (±1.5%)."""
    p1 = _SACKMANN_PROFILES.get(p1_key, {})
    p2 = _SACKMANN_PROFILES.get(p2_key, {})
    f1 = p1.get("first_serve_pct")
    f2 = p2.get("first_serve_pct")
    if f1 is None or f2 is None:
        return 0.0
    return max(-0.015, min(0.015, (f1 - f2) * 0.15))


def bp_attack_adj(p1_key: str, p2_key: str) -> float:
    """Break point conversion rate differential (±3%)."""
    p1 = _SACKMANN_PROFILES.get(p1_key, {})
    p2 = _SACKMANN_PROFILES.get(p2_key, {})
    c1 = p1.get("bp_conv_pct")
    c2 = p2.get("bp_conv_pct")
    if c1 is None or c2 is None:
        return 0.0
    return max(-0.03, min(0.03, (c1 - c2) * 0.12))


def conditioning_adj(p1_key: str, p2_key: str) -> float:
    """Heavy recent match load penalty (>6 matches in 14 days → fatigue flag)."""
    p1 = _SACKMANN_PROFILES.get(p1_key, {})
    p2 = _SACKMANN_PROFILES.get(p2_key, {})
    m1 = p1.get("matches_last_14d", 3)
    m2 = p2.get("matches_last_14d", 3)

    def penalty(m: int) -> float:
        if m >= 8:  return -0.030
        if m >= 7:  return -0.020
        if m >= 6:  return -0.010
        return 0.0

    return max(-0.04, min(0.04, penalty(m2) - penalty(m1)))


def _dynamic_weights(surface: str, is_wta: bool) -> Dict[str, float]:
    w = dict(DYNAMIC_WEIGHTS.get(surface, DYNAMIC_WEIGHTS["hard"]))
    if is_wta:
        for k, adj in WTA_WEIGHT_ADJ.items():
            w[k] = round(w.get(k, 0) + adj, 4)
    # 正規化使總和 = 1.0
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def playstyle_adj(p1_key: str, p2_key: str, surface: str) -> float:
    s1 = PLAYSTYLE.get(p1_key, "all_court")
    s2 = PLAYSTYLE.get(p2_key, "all_court")
    if s1 == s2:
        return 0.0
    base = STYLE_MATCHUP_ADJ.get((s1, s2), 0.0)
    if base == 0.0:
        base = -STYLE_MATCHUP_ADJ.get((s2, s1), 0.0)
    mod = STYLE_SURFACE_MOD.get(surface, 1.0)
    return max(-0.035, min(0.035, base * mod))


def surface_transition_adj(p1_key: str, p2_key: str,
                            prof1: dict, prof2: dict,
                            current_surface: str) -> float:
    adj = 0.0
    for pkey, prof, sign in ((p1_key, prof1, +1), (p2_key, prof2, -1)):
        last_surf = prof.get("last_surface")
        last_days = prof.get("last_surf_days", 999)
        if not last_surf or last_surf == current_surface:
            continue
        if last_days > SURFACE_TRANSITION_WINDOW:
            continue
        pen = SURFACE_TRANSITION_PENALTY.get((last_surf, current_surface), 0.0)
        adj += sign * pen
    return max(-0.030, min(0.030, adj))


def return_depth_adj(p1_key: str, p2_key: str, surface: str, is_wta: bool) -> float:
    p1 = _SACKMANN_PROFILES.get(p1_key, {})
    p2 = _SACKMANN_PROFILES.get(p2_key, {})
    ss1 = p1.get("second_serve_win")
    ss2 = p2.get("second_serve_win")
    if ss1 is None or ss2 is None:
        return 0.0
    scale = 0.30 if is_wta else 0.20
    if surface == "clay":
        scale *= 1.2
    return max(-0.030, min(0.030, (ss2 - ss1) * scale))


def injury_risk_adj(p1_key: str, p2_key: str) -> float:
    adj = 0.0
    all_db = {**ATP_STATS, **WTA_STATS}
    for pkey, sign in ((p1_key, +1), (p2_key, -1)):
        prof = _SACKMANN_PROFILES.get(pkey, {})
        if pkey in _INJURIES:
            adj += sign * (-0.030)
            continue
        # form collapse
        if prof.get("win_streak", 0) <= -3:
            adj += sign * INJURY_STREAK_PENALTY
        # serve efficiency drop vs season average
        db_sv = all_db.get(pkey, {}).get("hard", {}).get("svpt_won")
        recent_sv = prof.get("svpt_won")
        if db_sv and recent_sv and (db_sv - recent_sv) > INJURY_SERVE_DROP_THRESH:
            adj += sign * (-0.020)
    return max(-0.040, min(0.040, adj))


def public_bias_adj_fn(model_home, dv_home, model_away, dv_away):
    fade = 0.0
    if dv_home - model_home > PUBLIC_BIAS_THRESHOLD:
        fade -= PUBLIC_BIAS_FADE
    if dv_away - model_away > PUBLIC_BIAS_THRESHOLD:
        fade += PUBLIC_BIAS_FADE
    return fade


def compute_elo_from_sackmann(all_matches: List[dict]) -> None:
    """
    Derive surface-specific ELO from Sackmann match history and store in _LIVE_ELO.
    v4.0: Recency-weighted K factor — recent matches carry more weight.
    K=48/40/32 base, scaled by recency: ×1.0 (≤6mo), ×0.70 (6–18mo), ×0.45 (18mo+)
    """
    all_db = {**ATP_STATS, **WTA_STATS}
    elos: Dict[str, Dict[str, float]] = {}
    now_utc = datetime.datetime.utcnow()

    for row in sorted(all_matches, key=lambda r: r.get("tourney_date", "19000101")):
        wname = (row.get("winner_name") or "").lower()
        lname = (row.get("loser_name") or "").lower()
        wkey  = norm_player(wname)
        lkey  = norm_player(lname)
        if not wkey or not lkey:
            continue

        surf_raw = (row.get("surface") or "hard").lower()
        surf = surf_raw if surf_raw in ("hard", "clay", "grass") else "hard"

        for key in (wkey, lkey):
            if key not in elos:
                base = float(all_db.get(key, {}).get(surf, {}).get("elo", 1500))
                elos[key] = {"hard": base, "clay": base, "grass": base}

        ew = elos[wkey][surf]
        el = elos[lkey][surf]
        exp_w = 1.0 / (1.0 + 10.0 ** ((el - ew) / 400.0))

        lvl = (row.get("tourney_level") or "").upper()
        k   = 48 if lvl == "G" else 40 if lvl in ("M", "F") else 32

        # Recency-weighted K: more recent matches → higher impact
        days_ago = 730
        td_str = row.get("tourney_date", "")
        if len(td_str) == 8:
            try:
                match_dt = datetime.datetime(
                    int(td_str[:4]), int(td_str[4:6]), int(td_str[6:8]))
                days_ago = max(0, (now_utc - match_dt).days)
            except ValueError:
                pass
        if days_ago <= 180:
            k_mult = 1.00
        elif days_ago <= 540:
            k_mult = 0.70
        else:
            k_mult = ELO_RECENCY_FLOOR
        k = k * k_mult

        elos[wkey][surf] = ew + k * (1.0 - exp_w)
        elos[lkey][surf] = el - k * (1.0 - exp_w)

    # Store ELO for ALL computed players (not just those in static dict)
    for key, surf_elos in elos.items():
        _LIVE_ELO[key] = {s: round(v, 1) for s, v in surf_elos.items()}
    log.info("compute_elo_from_sackmann: %d players (recency-weighted K)", len(elos))


def compute_dynamic_h2h(all_matches: List[dict]) -> None:
    """
    v4.0: Build dynamic H2H records from Sackmann match data (last 5 years).
    Stored in _DYNAMIC_H2H (surface-specific) and _DYNAMIC_H2H_OVERALL.
    Used by h2h_adj() as primary source, falling back to static H2H tables.
    """
    global _DYNAMIC_H2H, _DYNAMIC_H2H_OVERALL
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=5 * 365)
    surf_counts:    Dict[Tuple[str, str, str], List[int]] = {}
    overall_counts: Dict[Tuple[str, str],       List[int]] = {}

    for row in all_matches:
        wname = (row.get("winner_name") or "").lower()
        lname = (row.get("loser_name") or "").lower()
        wkey  = norm_player(wname)
        lkey  = norm_player(lname)
        if not wkey or not lkey or wkey == lkey:
            continue

        td_str = row.get("tourney_date", "")
        if len(td_str) == 8:
            try:
                md = datetime.datetime(int(td_str[:4]), int(td_str[4:6]), int(td_str[6:8]))
                if md < cutoff:
                    continue
            except ValueError:
                continue

        surf_raw = (row.get("surface") or "hard").lower()
        surf = surf_raw if surf_raw in ("hard", "clay", "grass") else "hard"

        # Canonical order: alphabetical so (a,b) and (b,a) map to same key
        if wkey <= lkey:
            cp1, cp2, win_p1 = wkey, lkey, True
        else:
            cp1, cp2, win_p1 = lkey, wkey, False

        sk = (cp1, cp2, surf)
        if sk not in surf_counts:
            surf_counts[sk] = [0, 0]
        surf_counts[sk][0 if win_p1 else 1] += 1

        ok = (cp1, cp2)
        if ok not in overall_counts:
            overall_counts[ok] = [0, 0]
        overall_counts[ok][0 if win_p1 else 1] += 1

    _DYNAMIC_H2H         = {k: (v[0], v[1]) for k, v in surf_counts.items()}
    _DYNAMIC_H2H_OVERALL = {k: (v[0], v[1]) for k, v in overall_counts.items()}
    log.info("compute_dynamic_h2h: %d surface pairs, %d overall pairs",
             len(_DYNAMIC_H2H), len(_DYNAMIC_H2H_OVERALL))


def load_odds_prev() -> Dict[str, dict]:
    try:
        with open(ODDS_PREV_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_odds_prev(odds_map: Dict[str, dict]) -> None:
    os.makedirs("docs", exist_ok=True)
    try:
        snapshot = {k: {"best_home": v["best_home"], "best_away": v["best_away"]}
                    for k, v in odds_map.items()}
        with open(ODDS_PREV_PATH, "w") as f:
            json.dump(snapshot, f)
    except Exception as e:
        log.warning("save_odds_prev: %s", e)


def odds_move_signal(key: str, odds_info: dict,
                     prev: Dict[str, dict]) -> Tuple[float, str]:
    """
    Detect significant odds movement (sharp money indicator).
    Returns (prob_adj_for_home, label).
    """
    if key not in prev:
        return 0.0, ""
    ph = prev[key].get("best_home", odds_info["best_home"])
    pa = prev[key].get("best_away", odds_info["best_away"])
    ch = odds_info["best_home"]
    ca = odds_info["best_away"]

    def to_p(o: float) -> float:
        return 1.0 / o if o > 1.0 else 0.0

    shift = to_p(ch) - to_p(ph)  # positive = home shorted (steam on home)
    if abs(shift) < 0.025:
        return 0.0, ""

    label = "steam_home" if shift > 0 else "steam_away"
    adj   = max(-0.04, min(0.04, shift * 0.40))
    log.info("odds_move %s: shift=%+.3f -> %s adj=%+.4f", key, shift, label, adj)
    return adj, label


def detect_injuries(all_matches: List[dict]) -> set:
    """
    Scan last 14 days for RET/W/O results to flag potentially injured players.
    """
    cutoff  = datetime.datetime.utcnow() - datetime.timedelta(days=14)
    injured: set = set()
    for row in all_matches:
        try:
            md = datetime.datetime.strptime(
                str(row.get("tourney_date", "19000101")), "%Y%m%d")
        except ValueError:
            continue
        if md < cutoff:
            continue
        score = (row.get("score") or "").upper()
        if "RET" not in score and "W/O" not in score:
            continue
        loser = (row.get("loser_name") or "").lower()
        if loser:
            key = norm_player(loser)
            all_db = {**ATP_STATS, **WTA_STATS}
            if key in all_db:
                injured.add(key)
                log.info("auto-injury: %s (%s)", key, row.get("loser_name"))
    return injured


def extract_tournament(sport_key: str, game: dict) -> str:
    """Derive a normalised tournament name from sport_key / sport_title."""
    sk = sport_key.lower()
    for token in ("french_open", "wimbledon", "us_open", "australian_open"):
        if token in sk:
            return token
    title = game.get("sport_title", "").lower()
    return title.replace(" ", "_").replace("-", "_")


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
_ALIASES: Dict[str, str] = {
    "novak djokovic":        "djokovic",
    "carlos alcaraz":        "alcaraz",
    "jannik sinner":         "sinner",
    "daniil medvedev":       "medvedev",
    "alexander zverev":      "zverev",
    "andrey rublev":         "rublev",
    "stefanos tsitsipas":    "tsitsipas",
    "taylor fritz":          "fritz",
    "alex de minaur":        "de_minaur",
    "hubert hurkacz":        "hurkacz",
    "grigor dimitrov":       "dimitrov",
    "tommy paul":            "paul",
    "felix auger-aliassime": "auger_aliassime",
    "felix auger aliassime": "auger_aliassime",
    "lorenzo musetti":       "musetti",
    "frances tiafoe":        "tiafoe",
    "matteo berrettini":     "berrettini",
    "casper ruud":           "ruud",
    "jack draper":           "draper",
    "karen khachanov":       "khachanov",
    "ben shelton":           "shelton",
    "alexander bublik":      "bublik",
    "ugo humbert":           "humbert",
    "nicolas jarry":         "jarry",
    "flavio cobolli":        "cobolli",
    "pablo carreno busta":   "carreno_busta",
    "thiago agustin tirante":"tirante",
    "holger rune":                "rune",
    "jiri lehecka":               "lehecka",
    "sebastian korda":            "korda",
    "arthur fils":                "fils",
    "lorenzo sonego":             "sonego",
    "matteo arnaldi":             "arnaldi",
    "tallon griekspoor":          "griekspoor",
    "sebastian baez":             "baez",
    "tomas martin etcheverry":    "etcheverry",
    "tomas etcheverry":           "etcheverry",
    "alejandro tabilo":           "tabilo",
    "francisco cerundolo":        "cerundolo",
    "juan manuel cerundolo":      "juan_cerundolo",
    "alejandro davidovich fokina":"davidovich",
    "alejandro davidovich":       "davidovich",
    "mariano navone":             "navone",
    "joao fonseca":               "fonseca",
    "jakub mensik":               "mensik",
    "nuno borges":                "borges",
    "miomir kecmanovic":          "kecmanovic",
    "fabian marozsan":            "marozsan",
    "denis shapovalov":           "shapovalov",
    "cameron norrie":             "norrie",
    "adrian mannarino":           "mannarino",
    "corentin moutet":            "moutet",
    "zachary svajda":             "svajda",
    "rafael jodar":               "jodar",
    "borna coric":                "coric",
    "arthur rinderknech":         "rinderknech",
    "luciano darderi":            "darderi",
    "alexei popyrin":             "popyrin",
    "stan wawrinka":              "wawrinka",
    "hugo gaston":                "gaston",
    "iga swiatek":                "swiatek",
    "aryna sabalenka":       "sabalenka",
    "coco gauff":            "gauff",
    "elena rybakina":        "rybakina",
    "jessica pegula":        "pegula",
    "madison keys":          "keys",
    "qinwen zheng":          "zheng",
    "jasmine paolini":       "paolini",
    "emma navarro":          "navarro",
    "barbora krejcikova":    "krejcikova",
    "maria sakkari":         "sakkari",
    "daria kasatkina":       "kasatkina",
    "petra kvitova":         "kvitova",
    "beatriz haddad maia":   "haddad_maia",
    "marta kostyuk":         "kostyuk",
    "belinda bencic":        "bencic",
    "danielle collins":      "collins",
    "naomi osaka":                "osaka",
    "diana shnaider":             "shnaider",
    "anna kalinskaya":            "kalinskaya",
    "mirra andreeva":             "andreeva",
    "elina svitolina":            "svitolina",
    "liudmila samsonova":         "samsonova",
    "paula badosa":               "badosa",
    "anastasia potapova":         "potapova",
    "ons jabeur":                 "jabeur",
    "dayana yastremska":          "yastremska",
    "caroline garcia":            "garcia",
    "karolina muchova":           "muchova",
    "jelena ostapenko":           "ostapenko",
    "linda noskova":              "noskova",
    "magdalena frech":            "frech",
    "sorana cirstea":             "cirstea",
    "diane parry":                "parry",
    "maja chwalinska":            "chwalinska",
    "donna vekic":                "vekic",
    "veronika kudermetova":       "kudermetova",
    "emma raducanu":              "raducanu",
    "clara tauson":               "tauson",
    "anna blinkova":              "blinkova",
    "ekaterina alexandrova":      "alexandrova",
    "xinyu wang":                 "xinyu_wang",
    # common abbreviated / alternate forms
    "de minaur":                  "de_minaur",
    "alex de minaur":             "de_minaur",
    "carreno":                    "carreno_busta",
    "carreno busta":              "carreno_busta",
    "p. carreno busta":           "carreno_busta",
    "auger-aliassime":            "auger_aliassime",
    "haddad maia":                "haddad_maia",
    "b. haddad maia":             "haddad_maia",
    "davidovich fokina":          "davidovich",
    "a. davidovich fokina":       "davidovich",
    "t. etcheverry":              "etcheverry",
    "j. cerundolo":               "cerundolo",
    "j.m. cerundolo":             "juan_cerundolo",
    "f. cerundolo":               "cerundolo",
    "van de zandschulp":          "van_de_zandschulp",
    "mpetshi perricard":          "mpetshi_perricard",
    "giovanni mpetshi perricard": "mpetshi_perricard",
    # ── WTA extended aliases ──────────────────────────────────────────────────
    "simona halep":               "simona_halep",
    "karolina pliskova":          "karolina_pliskova",
    "karolína plíšková":          "karolina_pliskova",
    "elise mertens":              "elise_mertens",
    "katie boulter":              "katie_boulter",
    "celine naef":                "celine_naef",
    "anett kontaveit":            "anett_kontaveit",
    "linda fruhvirtova":          "linda_fruhvirtova",
    "leylah fernandez":           "leylah_fernandez",
    "anastasia pavlyuchenkova":   "anastasia_pavlyuchenkova",
    "bianca andreescu":           "bianca_andreescu",
    "lulu sun":                   "lulu_sun",
    "elisabetta cocciaretto":     "elisabetta_cocciaretto",
    "peyton stearns":             "peyton_stearns",
    "victoria azarenka":          "victoria_azarenka",
    "yulia putintseva":           "yulia_putintseva",
    "anhelina kalinina":          "anhelina_kalinina",
    "moyuka uchijima":            "moyuka_uchijima",
    "zhu lin":                    "zhu_lin",
    "magda linette":              "magda_linette",
    "alycia parks":               "alycia_parks",
    "elina avanesyan":            "elina_avanesyan",
    "camila osorio":              "camila_osorio",
    "varvara gracheva":           "varvara_gracheva",
    "lucia bronzetti":            "lucia_bronzetti",
    "eva lys":                    "eva_lys",
    "harmony tan":                "harmony_tan",
    "mayar sherif":               "mayar_sherif",
    "zeynep sonmez":              "zeynep_sonmez",
    "cristina bucsa":             "cristina_bucsa",
    "jessica bouzas maneiro":     "jessica_bouzas_maneiro",
    "laura siegemund":            "laura_siegemund",
    "katerina siniakova":         "katerina_siniakova",
    "claire liu":                 "claire_liu",
    "ann li":                     "ann_li",
    "marie bouzkova":             "marie_bouzkova",
    "lesia tsurenko":             "lesia_tsurenko",
    "sara sorribes tormo":        "sara_sorribes_tormo",
    "bernarda pera":              "bernarda_pera",
    "lucrezia stefanini":         "lucrezia_stefanini",
    "leonie kung":                "leonie_kung",
    "anna schmiedlova":           "anna_schmiedlova",
    "aliaksandra sasnovich":      "aliaksandra_sasnovich",
    "petra martic":               "petra_martic",
    "storm hunter":               "storm_hunter",
    "greet minnen":               "greet_minnen",
    "oksana selekhmeteva":        "oksana_selekhmeteva",
    "viktoriya tomova":           "viktoriya_tomova",
    "daria snigur":               "daria_snigur",
    "harriet dart":               "harriet_dart",
    "elsa jacquemot":             "elsa_jacquemot",
    "diana marcinkevica":         "diana_marcinkevica",
    "emma lene norsgaard":        "emma_lene_norsgaard",
    "tereza martincova":          "tereza_martincova",
    "tatjana maria":              "tatjana_maria",
    "dominika salkova":           "dominika_salkova",
    "dominika šalková":           "dominika_salkova",
    "dominika šalkova":           "dominika_salkova",
    "olga danilovic":             "olga_danilovic",
    "hailey baptiste":            "hailey_baptiste",
    "ocean dodin":                "ocean_dodin",
    "suzan lamens":               "suzan_lamens",
    "panna udvardy":              "panna_udvardy",
    "georgia masarova":           "georgia_masarova",
    "nuria parrizas diaz":        "nuria_parrizas_diaz",
    "nuria parrizas-diaz":        "nuria_parrizas_diaz",
    "julia grabher":              "julia_grabher",
    "irina begu":                 "irina_begu",
    "yafan wang":                 "yafan_wang",
    "jodie burrage":              "jodie_burrage",
    "ana bogdan":                 "ana_bogdan",
    "qiang wang":                 "qiang_wang",
    "nadia podoroska":            "nadia_podoroska",
    "tamara zidansek":            "tamara_zidansek",
    "dalma galfi":                "dalma_galfi",
    "rebecca sramkova":           "rebecca_sramkova",
    "emina bektas":               "emina_bektas",
    "renata zarazua":             "renata_zarazua",
    "nao hibino":                 "nao_hibino",
    "viktoriya golubic":          "viktoriya_golubic",
    "mia pohankova":              "mia_pohankova",
    "mia pohánková":              "mia_pohankova",
    "mia pohánkova":              "mia_pohankova",
    "sofya lansere":              "sofya_lansere",
    "nadiia kichenok":            "nadiia_kichenok",
    "katerina baindl":            "katerina_baindl",
    "fiona ferro":                "fiona_ferro",
    "irene burillo escorihuela":  "irene_burillo_escorihuela",
    "valentini grammatikopoulou": "valentini_grammatikopoulou",
    "sara errani":                "sara_errani",
    "ajla tomljanovic":           "ajla_tomljanovic",
    "yuki naito":                 "yuki_naito",
    "sloane stephens":            "sloane_stephens",
    "anastasija sevastova":       "anastasija_sevastova",
    "aliona bolsova":             "aliona_bolsova",
    "katie swan":                 "katie_swan",
    "kaia kanepi":                "kaia_kanepi",
    "polona hercog":              "polona_hercog",
    "bouzas maneiro":             "jessica_bouzas_maneiro",
    "sorribes tormo":             "sara_sorribes_tormo",
    "haddad maia":                "haddad_maia",
    # ── New ATP qualifying / lower-ranked players (ranks 85-300) ────────────────
    "michael zheng":              "michael_zheng",
    "m. zheng":                   "michael_zheng",
    "christopher o'connell":      "christopher_o'connell",
    "c. o'connell":               "christopher_o'connell",
    "marcelo tomas barrios vera": "marcelo_tomas_barrios_vera",
    "barrios vera":               "marcelo_tomas_barrios_vera",
    "m. barrios vera":            "marcelo_tomas_barrios_vera",
    "roberto carballes baena":    "roberto_carballes_baena",
    "carballes baena":            "roberto_carballes_baena",
    "daniel evans":               "daniel_evans",
    "mackenzie mcdonald":         "mackenzie_mcdonald",
    "soonwoo kwon":               "soonwoo_kwon",
    "harold mayot":               "harold_mayot",
    "bu yunchaokete":             "bu_yunchaokete",
    "laslo djere":                "laslo_djere",
    "luca nardi":                 "luca_nardi",
    "darwin blanch":              "darwin_blanch",
    "timofey skatov":             "timofey_skatov",
    "dusan lajovic":              "dusan_lajovic",
    "alejandro moro canas":       "alejandro_moro_canas",
    "moro canas":                 "alejandro_moro_canas",
    "alexis galarneau":           "alexis_galarneau",
    "otto virtanen":              "otto_virtanen",
    "roman safiullin":            "roman_safiullin",
    "zsombor piros":              "zsombor_piros",
    "clement tabur":              "clement_tabur",
    "billy harris":               "billy_harris",
    "borna gojo":                 "borna_gojo",
    "tristan boyer":              "tristan_boyer",
    "aziz dougaz":                "aziz_dougaz",
    "dane sweeny":                "dane_sweeny",
    "elias ymer":                 "elias_ymer",
    "august holmgren":            "august_holmgren",
    "jaime faria":                "jaime_faria",
    "andrea pellegrino":          "andrea_pellegrino",
    "arthur gea":                 "arthur_gea",
    "gustavo heide":              "gustavo_heide",
    "oliver tarvet":              "oliver_tarvet",
    "luka pavlovic":              "luka_pavlovic",
    "tristan schoolkate":         "tristan_schoolkate",
    "shintaro mochizuki":         "shintaro_mochizuki",
    "jerome kym":                 "jerome_kym",
    "nicolas mejia":              "nicolas_mejia",
    "paul jubb":                  "paul_jubb",
    "kimmer coppejans":           "kimmer_coppejans",
    "chris rodesch":              "chris_rodesch",
    "henry searle":               "henry_searle",
    "stefano travaglia":          "stefano_travaglia",
    "rei sakamoto":               "rei_sakamoto",
    "vilius gaubas":              "vilius_gaubas",
    "gauthier onclin":            "gauthier_onclin",
    "colton smith":               "colton_smith",
    "bernard tomic":              "bernard_tomic",
    "pol martin tiffon":          "pol_martin_tiffon",
    "martin tiffon":              "pol_martin_tiffon",
    "pablo llamas ruiz":          "pablo_llamas_ruiz",
    "llamas ruiz":                "pablo_llamas_ruiz",
    "federico cina":              "federico_cina",
    "federico cinà":              "federico_cina",
    "federico ciná":              "federico_cina",
    # ── New WTA qualifying / lower-ranked players (ranks 88-300) ────────────────
    "heather watson":             "heather_watson",
    "clervie ngounoue":           "clervie_ngounoue",
    "mai hontama":                "mai_hontama",
    "lucie havlickova":           "lucie_havlickova",
    "marina bassols ribera":      "marina_bassols_ribera",
    "bassols ribera":             "marina_bassols_ribera",
    "leolia jeanjean":            "leolia_jeanjean",
    "caroline dolehide":          "caroline_dolehide",
    "anna lena friedsam":         "anna_lena_friedsam",
    "anna-lena friedsam":         "anna_lena_friedsam",
    "robin montgomery":           "robin_montgomery",
    "viktoria hruncakova":        "viktoria_hruncakova",
    "mariam bolkvadze":           "mariam_bolkvadze",
    "xiaodi you":                 "xiaodi_you",
    "maria lourdes carle":        "maria_lourdes_carle",
    "lourdes carle":              "maria_lourdes_carle",
    "guo hanyu":                  "guo_hanyu",
    "mona barthel":               "mona_barthel",
    "yi zhou":                    "yi_zhou",
    "whitney osuigwe":            "whitney_osuigwe",
    "maria timofeeva":            "maria_timofeeva",
    "lin zhu":                    "lin_zhu",
    "akasha urhobo":              "akasha_urhobo",
    "lola radivojevic":           "lola_radivojevic",
    "tiantsoa sarah rajaonah":    "tiantsoa_sarah_rajaonah",
    "sarah rajaonah":             "tiantsoa_sarah_rajaonah",
    "carol young suh lee":        "carol_young_suh_lee",
    "carol suh lee":              "carol_young_suh_lee",
}


CHINESE_NAMES: Dict[str, str] = {
    # ATP Top players
    "Novak Djokovic": "德約科維奇", "Carlos Alcaraz": "阿爾卡拉斯",
    "Jannik Sinner": "辛納", "Daniil Medvedev": "梅德韋杰夫",
    "Alexander Zverev": "茲韋列夫", "Andrey Rublev": "魯布列夫",
    "Stefanos Tsitsipas": "西西帕斯", "Taylor Fritz": "弗里茨",
    "Alex de Minaur": "德米諾爾", "Hubert Hurkacz": "胡卡茲",
    "Grigor Dimitrov": "季米特洛夫", "Tommy Paul": "保羅",
    "Felix Auger-Aliassime": "奧熱-阿利亞西姆", "Lorenzo Musetti": "穆塞蒂",
    "Frances Tiafoe": "蒂亞福", "Matteo Berrettini": "貝雷蒂尼",
    "Casper Ruud": "魯德", "Jack Draper": "德雷珀",
    "Karen Khachanov": "哈恰諾夫", "Ben Shelton": "謝爾頓",
    "Alexander Bublik": "布布利克", "Ugo Humbert": "翁貝爾",
    "Nicolas Jarry": "哈里", "Flavio Cobolli": "科博利",
    "Holger Rune": "魯內", "Jiri Lehecka": "萊赫卡",
    "Sebastian Korda": "科爾達", "Alexei Popyrin": "波普林",
    "Lorenzo Sonego": "索內戈", "Arthur Fils": "費爾斯",
    "Tallon Griekspoor": "格里克斯普爾", "Matteo Arnaldi": "阿納爾迪",
    "Hugo Gaston": "加斯頓", "Gael Monfils": "孟菲爾斯",
    "Ethan Quinn": "奎恩", "Francisco Comesana": "科梅薩尼亞",
    "Sebastian Baez": "巴耶斯", "Tomas Etcheverry": "埃切維里",
    "Luciano Darderi": "達德里", "Alejandro Tabilo": "塔比羅",
    "Alejandro Davidovich Fokina": "達維多維奇", "Francisco Cerundolo": "切倫杜羅",
    "Adrian Mannarino": "曼納里諾", "Corentin Moutet": "穆泰",
    "Giovanni Mpetshi Perricard": "佩里卡爾", "Roberto Bautista Agut": "鮑蒂斯塔",
    "David Goffin": "高芬", "Borna Coric": "科里奇",
    "Stan Wawrinka": "瓦林卡", "Rafael Nadal": "納達爾",
    # WTA Top players
    "Iga Swiatek": "斯維亞泰克", "Aryna Sabalenka": "莎巴蘭卡",
    "Coco Gauff": "高芙", "Elena Rybakina": "里巴金娜",
    "Jessica Pegula": "佩古拉", "Madison Keys": "基斯",
    "Qinwen Zheng": "鄭欽文", "Jasmine Paolini": "保利尼",
    "Emma Navarro": "納瓦羅", "Barbora Krejcikova": "克雷奇科娃",
    "Maria Sakkari": "薩卡里", "Daria Kasatkina": "卡薩特金娜",
    "Petra Kvitova": "科維托娃", "Beatriz Haddad Maia": "阿達德·瑪雅",
    "Marta Kostyuk": "科斯秋克", "Belinda Bencic": "本西奇",
    "Danielle Collins": "柯林斯", "Dayana Yastremska": "雅斯特雷姆斯卡",
    "Ons Jabeur": "賈比爾", "Caroline Garcia": "加西亞",
    "Paula Badosa": "巴多薩", "Elina Svitolina": "斯維托利娜",
    "Mirra Andreeva": "安德烈耶娃", "Diana Shnaider": "施奈德",
    "Liudmila Samsonova": "薩姆索諾娃", "Victoria Azarenka": "阿紮倫卡",
    "Simona Halep": "哈勒普", "Anhelina Kalinina": "卡利尼娜",
    "Elena-Gabriela Ruse": "魯塞", "Clara Burel": "比里爾",
    "Ekaterina Alexandrova": "亞歷山德羅娃", "Anna Kalinskaya": "卡林斯卡婭",
    "Veronika Kudermetova": "庫德梅托娃", "Anastasia Pavlyuchenkova": "帕夫柳琴科娃",
    # ATP — additional
    "Miomir Kecmanovic": "科查諾維奇", "Fabian Marozsan": "馬羅森",
    "Vit Kopriva": "科普里瓦", "Brandon Nakashima": "中島",
    "Cameron Norrie": "諾里", "Denis Shapovalov": "夏波瓦洛夫",
    "Joao Fonseca": "方塞卡", "Sebastian Ofner": "奧夫納",
    "Jaume Munar": "穆納爾", "Rinky Hijikata": "土方里奇",
    "Tomas Machac": "馬哈奇", "Botic van de Zandschulp": "桑德舒爾普",
    "Alexander Shevchenko": "謝甫琴科", "Alexandre Muller": "穆勒",
    "Arthur Rinderknech": "林德克內希", "Hamad Medjedovic": "梅傑多維奇",
    "Hugo Dellien": "德連", "Jacob Fearnley": "費恩利",
    "Jaime Faria": "法里亞", "Jakub Mensik": "門西克",
    "Jan-Lennard Struff": "施特魯夫", "Luca Van Assche": "范阿什",
    "Mariano Navone": "納沃內", "Marin Cilic": "西里奇",
    "Marton Fucsovics": "富科維奇", "Nuno Borges": "博爾赫斯",
    "Pablo Carreno Busta": "卡雷尼奧-布斯塔", "Quentin Halys": "阿利斯",
    "Reilly Opelka": "奧佩爾卡", "Terence Atmane": "阿特曼",
    "Thanasi Kokkinakis": "科基納基斯", "Thomas Faurel": "福雷爾",
    "Titouan Droguet": "德魯蓋", "Tomas Martin Etcheverry": "埃切維里",
    "Valentin Vacherot": "瓦謝羅", "Valentin Royer": "羅耶爾",
    "Wu Yibing": "吳易昺", "Yannick Hanfmann": "漢夫曼",
    "Zachary Svajda": "斯瓦伊達", "Zizou Bergs": "貝爾格斯",
    "Adam Walton": "沃爾頓", "Emilio Nava": "納瓦",
    "Kyrian Jacquet": "雅凱", "Pablo Llamas Ruiz": "亞馬斯·魯伊斯",
    "Rafael Jodar": "霍達爾", "Roman Safiullin": "薩菲烏林",
    "Sebastian Baez": "拜斯", "Thiago Agustin Tirante": "蒂蘭特",
    "Alejandro Tabilo": "塔比羅", "Aleksandar Vukic": "武基奇",
    "Alexander Blockx": "布洛克斯", "Arthur Gea": "熱亞",
    "Benjamin Bonzi": "邦齊", "Dino Prizmic": "普里茲米奇",
    "Eliot Spizzirri": "斯皮扎里", "Ethan Quinn": "奎恩",
    "Facundo Diaz Acosta": "迪亞斯·阿科斯塔", "Ignacio Buse": "布塞",
    "Juan Manuel Cerundolo": "塞倫杜洛", "Karen Khachanov": "哈恰諾夫",
    "Luca Van Assche": "范阿什", "Martin Landaluce": "蘭達盧斯",
    "Raphael Collignon": "科利尼翁", "Roman Andres Burruchaga": "布魯查加",
    "Toby Samuel": "薩繆爾", "Hugo Gaston": "加斯頓",
    "Chak Lam Coleman Wong": "黃澤霖", "Luciano Darderi": "達德里",
    "James Duckworth": "達克沃斯", "Moise Kouame": "庫阿梅",
    "Valentin Vacherot": "瓦謝羅", "Adolfo Daniel Vallejo": "巴列霍",
    "Michael Zheng": "鄭麒泰", "Daniel Merida Aguilar": "梅里達",
    # WTA — additional
    "Ajla Tomljanovic": "托姆利亞諾維奇", "Akasha Urhobo": "烏魯霍博",
    "Alina Korneeva": "科爾尼娃", "Alycia Parks": "帕克斯",
    "Amanda Anisimova": "阿尼西莫娃", "Anastasia Potapova": "波塔波娃",
    "Anastasia Zakharova": "紮哈羅娃", "Anna Blinkova": "布林科娃",
    "Anna Bondár": "邦達爾", "Antonia Ruzic": "魯日奇",
    "Ashlyn Krueger": "克魯格", "Caty McNally": "麥克納利",
    "Clara Tauson": "陶森", "Claire Liu": "劉清漪",
    "Cristina Bucsa": "布克薩", "Dalma Galfi": "加爾菲",
    "Danka Kovinic": "科維尼奇", "Daria Snigur": "斯尼古爾",
    "Diane Parry": "帕里", "Donna Vekic": "維基奇",
    "Ella Seidel": "塞德爾", "Elsa Jacquemot": "雅克莫",
    "Emiliana Arango": "阿蘭戈", "Emma Raducanu": "拉杜卡努",
    "Fiona Ferro": "費羅", "Francesca Jones": "瓊斯",
    "Guo Hanyu": "郭涵宇", "Hailey Baptiste": "巴普蒂斯特",
    "Hanne Vandewinkel": "范德溫克爾", "Iva Jović": "約維奇",
    "Jaqueline Cristian": "克里斯蒂安", "Janice Tjen": "田佳妮",
    "Jelena Ostapenko": "奧斯塔潘科", "Jessica Bouzas Maneiro": "布扎斯·馬內羅",
    "Jil Teichmann": "泰希曼", "Julia Grabher": "格拉伯",
    "Kaitlin Quevedo": "克維多", "Kamilla Rakhimova": "拉希莫娃",
    "Karolina Muchova": "穆霍娃", "Katerina Siniakova": "西尼亞科娃",
    "Katie Boulter": "博爾特", "Katie Volynets": "沃利涅茨",
    "Kimberly Birrell": "比雷爾", "Ksenia Efremova": "葉夫列莫娃",
    "Laura Siegemund": "西格蒙德", "Leolia Jeanjean": "讓讓",
    "Leylah Fernandez": "費爾南德斯", "Lilli Tagger": "塔格爾",
    "Linda Fruhvirtova": "弗魯赫維爾托娃", "Linda Noskova": "諾斯科娃",
    "Lois Boisson": "博瓦松", "Lucia Bronzetti": "布隆澤蒂",
    "Magda Linette": "利內特", "Magdalena Frech": "弗雷赫",
    "Maja Chwalinska": "瓦林斯卡", "Marie Bouzkova": "布茲科娃",
    "Marina Bassols Ribera": "巴索爾斯·里貝拉", "Maya Joint": "喬恩特",
    "Mayar Sherif": "謝里夫", "Naomi Osaka": "大坂直美",
    "Nikola Bartunkova": "巴爾圖恩科娃", "Oksana Selekhmeteva": "塞萊赫梅捷娃",
    "Oleksandra Oliynykova": "奧利尼科娃", "Panna Udvardy": "烏德瓦爾迪",
    "Peyton Stearns": "斯特恩斯", "Rebecca Sramkova": "斯拉姆科娃",
    "Renata Zarazua": "薩拉蘇亞", "Sara Bejlek": "貝耶克",
    "Sara Sorribes Tormo": "索里貝斯·托爾莫", "Shuai Zhang": "張帥",
    "Simona Waltert": "瓦爾特", "Sinja Kraus": "克勞斯",
    "Sloane Stephens": "斯蒂芬斯", "Sofia Kenin": "科寧",
    "Solana Sierra": "西耶拉", "Sorana Cirstea": "奇爾斯蒂亞",
    "Susan Bandecchi": "班德基", "Talia Gibson": "吉布森",
    "Tamara Korpatsch": "科爾帕奇", "Tatjana Maria": "瑪麗亞",
    "Taylor Townsend": "湯森德", "Tiantsoa Sarah Rajaonah": "拉喬阿納",
    "Victoria Mboko": "姆博科", "Viktorija Golubic": "戈盧比奇",
    "Veronika Erjavec": "葉爾亞維茨", "Xiyu Wang": "王欣宇",
    "Xinyu Wang": "王欣瑜", "Yulia Putintseva": "普京採娃",
    "Yuliia Starodubtseva": "斯塔羅杜勃采娃", "Zeynep Sonmez": "松梅茲",
    "Alexandra Eala": "埃阿拉", "Alice Tubello": "圖貝洛",
    "Elina Svitolina": "斯維托利娜", "Elena Pridankina": "普里丹金娜",
    "Petra Marcinko": "馬爾欽科", "Eva Lys": "呂斯",
    "McCartney Kessler": "凱斯勒",
}


def cn_name(full_name: str) -> str:
    """Return Chinese name if known, else the last word of the English name."""
    if not full_name:
        return full_name
    if full_name in CHINESE_NAMES:
        return CHINESE_NAMES[full_name]
    nl = full_name.lower()
    for en, cn in CHINESE_NAMES.items():
        if en.lower() == nl:
            return cn
    return full_name.split()[-1] if full_name.split() else full_name


def norm_player(name: str, tour: str = "") -> str:
    """Normalise a player display name to a database key.

    tour: "atp" or "wta" hint to avoid cross-tour last-name collisions
    (e.g. "Zheng" in ATP context must not resolve to WTA Qinwen Zheng).
    """
    n = name.lower().strip()
    if n in _ALIASES:
        key = _ALIASES[n]
        # Skip if the key belongs only to the opposite tour
        if tour == "atp" and key in WTA_STATS and key not in ATP_STATS:
            pass
        elif tour == "wta" and key in ATP_STATS and key not in WTA_STATS:
            pass
        else:
            return key
    # Last-name shortcut: only for single-word inputs (avoids "Michael Zheng" → WTA Zheng)
    if " " not in n and "-" not in n:
        for alias, key in _ALIASES.items():
            if alias.split()[-1] == n:
                if tour == "atp" and key in WTA_STATS and key not in ATP_STATS:
                    continue
                if tour == "wta" and key in ATP_STATS and key not in WTA_STATS:
                    continue
                return key
    return n.replace(" ", "_").replace("-", "_")


def get_surface_stats(key: str, surface: str, tour: str = "") -> dict:
    players = {**ATP_STATS, **WTA_STATS}
    surf = surface if surface in ("hard", "clay", "grass") else "hard"
    has_static = key in players

    # If not in static database, check live rankings for rank-based stats
    if not has_static and key in _LIVE_RANKS:
        rank, live_tour = _LIVE_RANKS[key]
        base = _rank_based_stats(rank, surf, live_tour)
    else:
        if not has_static:
            # Defaults must be consistent: elo=1500 ≈ WTA rank 170+ / ATP rank 180+
            # Match the floor values from _rank_based_stats to avoid ELO/MC divergence
            is_wta_player = (tour == "wta") or (key in WTA_STATS)
            if is_wta_player:
                default = {"svpt_won": 0.520, "rtpt_won": 0.340, "elo": 1500}
            else:
                default = {"svpt_won": 0.590, "rtpt_won": 0.315, "elo": 1500}
        else:
            # has_static=True but surface entry may be missing — keep tour-aware fallback
            is_wta_player = (tour == "wta") or (key in WTA_STATS)
            if is_wta_player:
                default = {"svpt_won": 0.520, "rtpt_won": 0.340, "elo": 1500}
            else:
                default = {"svpt_won": 0.590, "rtpt_won": 0.315, "elo": 1500}
        base = dict(players.get(key, {}).get(surf, default))

    live_elo = _LIVE_ELO.get(key, {}).get(surf)
    if live_elo:
        base["elo"] = live_elo
    prof = _SACKMANN_PROFILES.get(key, {})
    n = prof.get("n_matches", 0)
    # Prefer surface-specific Sackmann stats when available (≥3 matches on that surface)
    surf_sv = prof.get("surf_svpt_won", {}).get(surf)
    surf_rt = prof.get("surf_rtpt_won", {}).get(surf)
    if surf_sv is not None:
        # Surface-specific data: high trust, scale with sample size
        w_sack = min(0.85, 0.55 + n * 0.015) if not has_static else min(0.75, 0.45 + n * 0.012)
        base["svpt_won"] = base["svpt_won"] * (1 - w_sack) + surf_sv * w_sack
        base["rtpt_won"] = base["rtpt_won"] * (1 - w_sack) + (surf_rt or base["rtpt_won"]) * w_sack
    else:
        rec = _RECENT_STATS.get(key, {})
        if rec.get("svpt_won"):
            # All-surface average: lower trust than surface-specific
            w_rec = min(0.70, 0.40 + n * 0.010) if not has_static else min(0.55, 0.30 + n * 0.008)
            base["svpt_won"] = base["svpt_won"] * (1 - w_rec) + rec["svpt_won"] * w_rec
            base["rtpt_won"] = base["rtpt_won"] * (1 - w_rec) + rec.get("rtpt_won", base["rtpt_won"]) * w_rec
    return base


def infer_surface(sport_key: str, tournament: str = "") -> str:
    t = (sport_key + " " + tournament).lower()
    if any(x in t for x in ["clay", "french", "roland", "madrid", "rome",
                              "barcelona", "monte_carlo", "monte-carlo",
                              "hamburg", "kitzbuhel", "bastad", "gstaad",
                              "lyon", "geneva", "estoril", "umag",
                              "bucharest", "marrakech", "belgrade", "zagreb",
                              "istanbul", "munich", "budapest"]):
        return "clay"
    if any(x in t for x in ["grass", "wimbledon", "queens", "halle",
                              "eastbourne", "s-hertogenbosch"]):
        return "grass"
    return "hard"


def infer_tour_level(sport_key: str, tournament: str = "") -> str:
    t = (sport_key + " " + tournament).lower()
    is_wta = "wta" in t
    if any(x in t for x in ["australian", "french", "wimbledon", "us_open",
                              "us open", "roland", "grand_slam"]):
        return "grand_slam"
    if any(x in t for x in ["masters", "1000", "indian_wells", "miami",
                              "montreal", "toronto", "cincinnati", "shanghai",
                              "paris", "rome"]):
        return "wta1000" if is_wta else "masters1000"
    if any(x in t for x in ["500", "dubai", "acapulco", "barcelona",
                              "washington", "hamburg"]):
        return "wta500" if is_wta else "atp500"
    return "wta250" if is_wta else "atp250"


def data_quality_score(key: str) -> float:
    """0.2=generic fallback only, 0.65=live rank only, 0.72=static only, 1.0=full static+Sackmann"""
    static = {**ATP_STATS, **WTA_STATS}
    has_static = key in static
    has_live_rank = key in _LIVE_RANKS
    prof = _SACKMANN_PROFILES.get(key, {})
    n = prof.get("n_matches", 0)
    if has_static and n >= 10:
        return 1.0
    if has_static and n >= 5:
        return 0.90
    if has_static and n >= 1:
        return 0.80
    if has_static:
        return 0.72
    if has_live_rank and n >= 10:
        return 0.75
    if has_live_rank and n >= 5:
        return 0.68
    if has_live_rank and n >= 1:
        return 0.65
    if has_live_rank:
        return 0.62  # live rank, no match data — still better than pure fallback
    if n >= 10:
        return 0.65
    if n >= 5:
        return 0.50
    if n > 0:
        return 0.35
    return 0.20


# ─────────────────────────────────────────────────────────────────────────────
# JEFF SACKMANN DATA — ROLLING FORM + FATIGUE + ADVANCED STATS
# ─────────────────────────────────────────────────────────────────────────────

def _calc_svpt_won(row: dict, prefix: str = "w") -> Optional[float]:
    """Derive serve point win % from a Sackmann match CSV row."""
    try:
        svpt = float(row.get(f"{prefix}_svpt") or 0)
        in1  = float(row.get(f"{prefix}_1stIn") or 0)
        won1 = float(row.get(f"{prefix}_1stWon") or 0)
        won2 = float(row.get(f"{prefix}_2ndWon") or 0)
        if svpt < 20:
            return None
        fsp = in1 / svpt
        fsw = won1 / in1 if in1 > 0 else 0.68
        ssw = won2 / max(1.0, svpt - in1)
        return fsp * fsw + (1.0 - fsp) * ssw
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def _name_matches(csv_name: str, full_name: str) -> bool:
    """Check if a Sackmann CSV 'First Last' name matches our full_name."""
    cl    = csv_name.lower().strip()
    parts = full_name.lower().split()
    if not parts or not cl:
        return False
    last = parts[-1]
    if last not in cl:
        return False
    if len(last) < 6:
        first     = parts[0][0] if parts[0] else ""
        csv_parts = cl.split()
        csv_first = csv_parts[0][0] if csv_parts and csv_parts[0] else ""
        return first == csv_first
    return True


def _rank_based_stats(rank: int, surface: str, tour: str) -> dict:
    """Compute estimated ELO/serve/return stats from current ranking."""
    surf = surface if surface in ("hard", "clay", "grass") else "hard"
    r = max(1, rank)
    if tour == "wta":
        hard_elo = max(1550, 1950 - max(0, r - 10) * 2.8)
        svpt_won = max(0.520, 0.582 - r * 0.0007)
        rtpt_won = max(0.340, 0.418 - r * 0.0006)
        elo_offsets = {"hard": 0, "clay": -15, "grass": -20}
    else:
        hard_elo = max(1600, 1950 - max(0, r - 10) * 2.5)
        svpt_won = max(0.590, 0.655 - r * 0.0006)
        rtpt_won = max(0.315, 0.365 - r * 0.0005)
        elo_offsets = {"hard": 0, "clay": -20, "grass": -25}
    elo = hard_elo + elo_offsets.get(surf, 0)
    return {"svpt_won": round(svpt_won, 4), "rtpt_won": round(rtpt_won, 4), "elo": round(elo, 1)}


def fetch_current_rankings() -> None:
    """Load current ATP/WTA rankings from local cloned Sackmann repo or HTTP fallback."""
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    http_headers: dict = {"Authorization": f"token {gh_token}"} if gh_token else {}
    local_paths = {
        "atp": os.environ.get("SACKMANN_ATP_PATH", ""),
        "wta": os.environ.get("SACKMANN_WTA_PATH", ""),
    }

    def _read_csv(local_base: str, filename: str) -> Optional[str]:
        """Return CSV text from local file, or HTTP fallback, or None."""
        if local_base:
            fp = os.path.join(local_base, filename)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp)
                if sz > 100:
                    with open(fp, "r", encoding="utf-8") as f:
                        return f.read()
                else:
                    log.info("_read_csv: %s is only %d bytes, trying HTTP fallback", filename, sz)
        tour_name = filename.split("_")[0]
        url = (f"https://raw.githubusercontent.com/mikahon/tennis_{tour_name}"
               f"/master/{filename}")
        try:
            rp = requests.get(url, timeout=30, headers=http_headers)
            if rp.status_code == 200:
                log.info("_read_csv: fetched %s via HTTP (%d bytes)", filename, len(rp.content))
                return rp.text
            else:
                log.info("_read_csv: HTTP %d for %s (no fallback data)", rp.status_code, url)
        except Exception as exc:
            log.info("_read_csv: request error for %s: %s", filename, exc)
        return None

    for tour in ("atp", "wta"):
        try:
            players_text = _read_csv(local_paths[tour], f"{tour}_players.csv")
            if not players_text:
                log.info("fetch_current_rankings: %s players CSV not found, using static data", tour)
                continue
            players: Dict[str, str] = {}
            for row in csv.DictReader(io.StringIO(players_text)):
                pid   = row.get("player_id", "").strip()
                first = (row.get("name_first") or "").strip()
                last  = (row.get("name_last")  or "").strip()
                if pid and last:
                    players[pid] = f"{first} {last}".strip()

            rank_text = _read_csv(local_paths[tour], f"{tour}_rankings_current.csv")
            if not rank_text:
                log.info("fetch_current_rankings: %s rankings CSV not found, using static data", tour)
                continue

            count = 0
            for row in csv.DictReader(io.StringIO(rank_text)):
                pid      = (row.get("player") or row.get("player_id") or "").strip()
                rank_str = (row.get("rank") or "").strip()
                if not pid or not rank_str:
                    continue
                try:
                    rank = int(rank_str)
                except ValueError:
                    continue
                if rank > 600:
                    continue
                full_name = players.get(pid, "")
                if full_name:
                    key = norm_player(full_name, tour=tour)
                    _LIVE_RANKS[key] = (rank, tour)
                    count += 1
            log.info("fetch_current_rankings: %s → %d players (top 600)", tour, count)
        except Exception as e:
            log.info("fetch_current_rankings %s: %s", tour, e)


def fetch_sackmann_matches(year: int = None) -> List[dict]:
    """
    Load ATP + WTA match CSVs.
    Local path: checks year, year-1 through year-4 (current/recent data when available locally).
    HTTP mirror: only year-2 through year-4 (mikahon mirror lag is ~1-2 years).
    """
    if year is None:
        year = datetime.datetime.utcnow().year
    local_paths = {
        "atp": os.environ.get("SACKMANN_ATP_PATH", ""),
        "wta": os.environ.get("SACKMANN_WTA_PATH", ""),
    }
    rows: List[dict] = []
    # HTTP mirror only has data up to ~year-2; local clone may have current/recent year
    mirror_years = [year - 2, year - 3, year - 4]
    local_only_years = [year, year - 1]
    years_seen: set = set()

    for y in local_only_years + mirror_years:
        for tour in ("atp", "wta"):
            if (y, tour) in years_seen:
                continue
            fetched = False
            # 1. Try local file (any year)
            local_file = os.path.join(local_paths[tour], f"{tour}_matches_{y}.csv") if local_paths[tour] else ""
            if local_file and os.path.isfile(local_file):
                try:
                    with open(local_file, "r", encoding="utf-8") as f:
                        batch = list(csv.DictReader(f))
                    rows.extend(batch)
                    log.info("fetch_sackmann: %s_%d (local) → %d rows", tour, y, len(batch))
                    fetched = True
                except Exception as e:
                    log.info("fetch_sackmann %s_%d local read error: %s", tour, y, e)
            if fetched:
                years_seen.add((y, tour))
                continue
            # 2. HTTP fallback only for mirror years
            if y not in mirror_years:
                continue
            url = (f"https://raw.githubusercontent.com/mikahon/tennis_{tour}"
                   f"/master/{tour}_matches_{y}.csv")
            try:
                r = requests.get(url, timeout=25)
                if r.status_code == 200:
                    batch = list(csv.DictReader(io.StringIO(r.text)))
                    rows.extend(batch)
                    log.info("fetch_sackmann: %s_%d (HTTP) → %d rows", tour, y, len(batch))
                    fetched = True
                else:
                    log.info("fetch_sackmann %s_%d: HTTP %d (no data for this year)", tour, y, r.status_code)
            except Exception as e:
                log.info("fetch_sackmann %s_%d request error: %s", tour, y, e)
            years_seen.add((y, tour))
    log.info("fetch_sackmann total: %d rows", len(rows))
    return rows


def build_player_profile(all_matches: List[dict], full_name: str,
                         n: int = 20) -> Optional[dict]:
    """Compute rolling serve/return/form/fatigue + advanced stats from last n matches."""
    player_rows: List[Tuple[dict, bool]] = []
    for row in all_matches:
        wname = row.get("winner_name", "")
        lname = row.get("loser_name", "")
        if _name_matches(wname, full_name):
            player_rows.append((row, True))
        elif _name_matches(lname, full_name):
            player_rows.append((row, False))

    if not player_rows:
        return None

    player_rows.sort(key=lambda x: x[0].get("tourney_date", "0"), reverse=True)

    last_surface  = "hard"
    last_surf_days = 999
    if player_rows:
        last_row = player_rows[0][0]
        ls_raw = (last_row.get("surface") or "hard").lower()
        last_surface = ls_raw if ls_raw in ("hard", "clay", "grass") else "hard"
        ls_date = last_row.get("tourney_date", "")
        if len(ls_date) == 8:
            try:
                ld2 = datetime.datetime(int(ls_date[:4]), int(ls_date[4:6]), int(ls_date[6:8]))
                last_surf_days = max(0, (datetime.datetime.utcnow() - ld2).days)
            except ValueError:
                pass

    recent = player_rows[:n]

    sv_wons, rt_wons, results, mins_list, sets_list = [], [], [], [], []
    df_list:   List[float] = []
    ace_list:  List[float] = []
    fs_pct_list:  List[float] = []  # 1st serve %
    ss_win_list:  List[float] = []  # 2nd serve win %
    bp_conv_num,  bp_conv_den = 0, 0  # BP conversion (attack)
    tb_won, tb_total        = 0, 0
    bp_saved, bp_faced      = 0, 0
    dec_won, dec_total      = 0, 0
    # surface-specific serve/return tracking
    surf_sv: Dict[str, List[float]] = {"hard": [], "clay": [], "grass": []}
    surf_rt: Dict[str, List[float]] = {"hard": [], "clay": [], "grass": []}
    now_utc = datetime.datetime.utcnow()
    cutoff14 = now_utc - datetime.timedelta(days=14)
    matches_14d = 0
    surface_res: Dict[str, List[int]] = {"hard": [], "clay": [], "grass": []}

    def _sf(val) -> Optional[float]:
        try:
            v = float(val or 0)
            return v if v > 0 else None
        except (ValueError, TypeError):
            return None

    for row, is_winner in recent:
        prefix     = "w" if is_winner else "l"
        opp_prefix = "l" if is_winner else "w"

        sv     = _calc_svpt_won(row, prefix)
        rt_opp = _calc_svpt_won(row, opp_prefix)
        if sv is not None:
            sv_wons.append(sv)
        if rt_opp is not None:
            rt_wons.append(1.0 - rt_opp)

        results.append(1 if is_winner else 0)

        try:
            m = float(row.get("minutes") or 0)
            if m > 0:
                mins_list.append(m)
        except (ValueError, TypeError):
            pass

        score = row.get("score", "") or ""
        sets  = [s for s in score.split() if "-" in s and not s.startswith("RET")]
        sets_list.append(max(1, len(sets)))

        svpt  = _sf(row.get(f"{prefix}_svpt"))
        in1st = _sf(row.get(f"{prefix}_1stIn"))
        w2nd  = _sf(row.get(f"{prefix}_2ndWon"))
        df    = _sf(row.get(f"{prefix}_df"))
        ace   = _sf(row.get(f"{prefix}_ace"))

        if svpt and svpt >= 20:
            if df is not None:
                df_list.append(df / svpt)
            if ace is not None:
                ace_list.append(ace / svpt)
            if in1st is not None:
                fs_pct_list.append(in1st / svpt)
                sec_svpt = svpt - in1st
                if w2nd is not None and sec_svpt > 0:
                    ss_win_list.append(w2nd / sec_svpt)

        # BP conversion (attacking): opp prefix tells us our chances
        opp = "l" if is_winner else "w"
        try:
            opp_bpf = int(row.get(f"{opp}_bpFaced") or 0)
            opp_bps = int(row.get(f"{opp}_bpSaved") or 0)
            if opp_bpf > 0:
                bp_conv_num += opp_bpf - opp_bps  # BPs we converted
                bp_conv_den += opp_bpf
        except (ValueError, TypeError):
            pass

        # Match load in last 14 days
        td_str = row.get("tourney_date", "")
        if len(td_str) == 8:
            try:
                md = datetime.datetime(int(td_str[:4]), int(td_str[4:6]), int(td_str[6:8]))
                if md >= cutoff14:
                    matches_14d += 1
            except ValueError:
                pass

        for s in sets:
            base = s.split("(")[0]
            if base == "7-6":
                tb_total += 1
                if is_winner:
                    tb_won += 1
            elif base == "6-7":
                tb_total += 1
                if not is_winner:
                    tb_won += 1

        try:
            bpf = int(row.get(f"{prefix}_bpFaced") or 0)
            bps = int(row.get(f"{prefix}_bpSaved") or 0)
            if bpf > 0:
                bp_faced += bpf
                bp_saved += bps
        except (ValueError, TypeError):
            pass

        if len(sets) >= 3:
            dec_total += 1
            if is_winner:
                dec_won += 1

        surf_raw = (row.get("surface") or "hard").lower()
        surf_key = surf_raw if surf_raw in surface_res else "hard"
        surface_res[surf_key].append(1 if is_winner else 0)
        # surface-specific serve/return
        sv_val = _calc_svpt_won(row, prefix)
        rt_val = _calc_svpt_won(row, opp_prefix)
        if sv_val is not None:
            surf_sv[surf_key].append(sv_val)
        if rt_val is not None:
            surf_rt[surf_key].append(1.0 - rt_val)

    # v4.0: exponential decay (λ=FORM_DECAY_LAMBDA) — recent matches weighted higher
    weights   = [math.exp(-FORM_DECAY_LAMBDA * i) for i in range(len(results))]
    total_w   = sum(weights)
    form_rate = sum(r * w for r, w in zip(results, weights)) / total_w if total_w > 0 else 0.5

    last_date_str = recent[0][0].get("tourney_date", "") or ""
    days_rest = 3
    if len(last_date_str) == 8:
        try:
            ld = datetime.datetime(
                int(last_date_str[:4]),
                int(last_date_str[4:6]),
                int(last_date_str[6:8]),
            )
            days_rest = max(0, (datetime.datetime.utcnow() - ld).days)
        except ValueError:
            pass

    surf_form = {
        s: round(sum(v) / len(v), 4)
        for s, v in surface_res.items() if len(v) >= 3
    }

    # Win streak: positive = consecutive wins, negative = consecutive losses
    win_streak = 0
    if results:
        direction = results[0]  # 1=win, 0=loss
        for r in results:
            if r == direction:
                win_streak += 1 if direction else -1
            else:
                break

    return {
        "svpt_won":     round(sum(sv_wons) / len(sv_wons), 4) if sv_wons else None,
        "rtpt_won":     round(sum(rt_wons) / len(rt_wons), 4) if rt_wons else None,
        "form_rate":    round(form_rate, 4),
        "n_matches":    len(recent),
        "days_rest":    days_rest,
        "last_minutes": mins_list[0] if mins_list else 90.0,
        "avg_minutes":  round(sum(mins_list) / len(mins_list), 1) if mins_list else 90.0,
        "last_sets":    sets_list[0] if sets_list else 3,
        "df_rate":      round(sum(df_list)  / len(df_list),  5) if df_list  else None,
        "ace_rate":     round(sum(ace_list) / len(ace_list), 5) if ace_list else None,
        "tb_win_pct":   round(tb_won / tb_total, 4) if tb_total >= 3 else None,
        "bp_save_pct":  round(bp_saved / bp_faced, 4) if bp_faced >= 5 else None,
        "deciding_pct": round(dec_won / dec_total, 4) if dec_total >= 3 else None,
        "surface_form":       surf_form,
        "win_streak":         win_streak,
        "first_serve_pct":    round(sum(fs_pct_list) / len(fs_pct_list), 4) if fs_pct_list  else None,
        "second_serve_win":   round(sum(ss_win_list) / len(ss_win_list), 4) if ss_win_list  else None,
        "bp_conv_pct":        round(bp_conv_num / bp_conv_den, 4) if bp_conv_den >= 5 else None,
        "matches_last_14d":   matches_14d,
        "last_surface":       last_surface,
        "last_surf_days":     last_surf_days,
        "surf_svpt_won": {s: round(sum(v)/len(v), 4) for s, v in surf_sv.items() if len(v) >= 3},
        "surf_rtpt_won": {s: round(sum(v)/len(v), 4) for s, v in surf_rt.items() if len(v) >= 3},
    }


def load_sackmann_data(all_matches: Optional[List[dict]] = None) -> None:
    """Populate _SACKMANN_PROFILES + _RECENT_STATS from Sackmann CSV data."""
    if all_matches is None:
        all_matches = fetch_sackmann_matches()
    if not all_matches:
        log.warning("load_sackmann_data: no match data — using static stats only")
        return
    all_players = {**ATP_STATS, **WTA_STATS}

    # Also collect players from Sackmann data who are NOT in the static list
    # (e.g. qualifiers, lower-ranked players currently in ATP/WTA draws)
    extra_players: Dict[str, str] = {}   # key → full_name
    for row in all_matches:
        for field in ("winner_name", "loser_name"):
            name = (row.get(field) or "").strip()
            if not name:
                continue
            key = norm_player(name.lower())
            if key not in all_players and key not in extra_players:
                extra_players[key] = name
    ok = 0
    for key, pdata in all_players.items():
        full_name = pdata.get("full_name", "")
        if not full_name:
            continue
        profile = build_player_profile(all_matches, full_name, n=20)
        if profile:
            _SACKMANN_PROFILES[key] = profile
            if profile.get("svpt_won"):
                rec: dict = {"svpt_won": profile["svpt_won"]}
                if profile.get("rtpt_won"):
                    rec["rtpt_won"] = profile["rtpt_won"]
                _RECENT_STATS[key] = rec
            ok += 1
    log.info("load_sackmann_data: %d/%d static players profiled", ok, len(all_players))

    # Profile extra players (qualifiers / lower-ranked not in static list)
    extra_ok = 0
    for key, full_name in extra_players.items():
        profile = build_player_profile(all_matches, full_name, n=20)
        if profile:
            _SACKMANN_PROFILES[key] = profile
            if profile.get("svpt_won"):
                rec2: dict = {"svpt_won": profile["svpt_won"]}
                if profile.get("rtpt_won"):
                    rec2["rtpt_won"] = profile["rtpt_won"]
                _RECENT_STATS[key] = rec2
            extra_ok += 1
    log.info("load_sackmann_data: +%d/%d extra players profiled", extra_ok, len(extra_players))

    # Compute live surface ELO from match history (replaces static fetch_ta_elo)
    compute_elo_from_sackmann(all_matches)

    # v4.0: Build dynamic H2H records from the same match history
    compute_dynamic_h2h(all_matches)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE  (v4.0 — 15-factor + Power-Devig + Calibration)
# ─────────────────────────────────────────────────────────────────────────────

def predict(p1_key: str, p2_key: str, surface: str,
            tour_level: str = "atp250", best_of: int = 3,
            sport_key: str = "", tournament: str = "") -> dict:
    """
    Base: 25% Surface ELO + 25% Markov + 20% H/B + 30% Advanced (DF-adjusted H/B)
    Adj:  fatigue(age-weighted) | surface-form | H2H(dynamic) | clutch(BO5-boosted)
          DF | lefty | backhand | altitude | wind | win-streak | BO5 specialist
    Post: probability calibration (α=0.88); WTA extra dampening (α×0.96)
    """
    surf_adj = SURFACE_PT_ADJ.get(surface, 0.0)
    is_wta   = "wta" in sport_key.lower() or "wta" in tour_level.lower()
    tour_str = "wta" if is_wta else "atp"

    s1 = get_surface_stats(p1_key, surface, tour=tour_str)
    s2 = get_surface_stats(p2_key, surface, tour=tour_str)

    prof1 = _SACKMANN_PROFILES.get(p1_key, {})
    prof2 = _SACKMANN_PROFILES.get(p2_key, {})

    cs_adj   = get_court_speed_adj(sport_key, tournament)
    lefty_sv = lefty_matchup_adj(p1_key, p2_key, surface)

    p1_sv = max(0.50, min(0.78,
        0.5 * (s1["svpt_won"] + 1.0 - s2["rtpt_won"]) + surf_adj + cs_adj + lefty_sv))
    p2_sv = max(0.50, min(0.78,
        0.5 * (s2["svpt_won"] + 1.0 - s1["rtpt_won"]) + surf_adj + cs_adj - lefty_sv))

    markov_p1 = match_win_prob(p1_sv, p2_sv, best_of=best_of)
    elo_p1    = elo_win_prob(s1.get("elo", 1800), s2.get("elo", 1800))

    hold1  = game_win_prob(max(0.50, min(0.80, s1["svpt_won"] + surf_adj + cs_adj)))
    hold2  = game_win_prob(max(0.50, min(0.80, s2["svpt_won"] + surf_adj + cs_adj)))
    break1 = game_win_prob(max(0.30, min(0.65, s1["rtpt_won"] - surf_adj)))
    break2 = game_win_prob(max(0.30, min(0.65, s2["rtpt_won"] - surf_adj)))
    hb_p1  = hold_break_win_prob(hold1, break1, hold2, break2)

    df1 = prof1.get("df_rate") or 0.04
    df2 = prof2.get("df_rate") or 0.04
    hold1_df = game_win_prob(max(0.50, min(0.80,
        s1["svpt_won"] * (1.0 - df1 * 1.5) + surf_adj + cs_adj)))
    hold2_df = game_win_prob(max(0.50, min(0.80,
        s2["svpt_won"] * (1.0 - df2 * 1.5) + surf_adj + cs_adj)))
    adv_p1 = hold_break_win_prob(hold1_df, break1, hold2_df, break2)

    w = _dynamic_weights(surface, is_wta)
    raw_prob = w["elo"] * elo_p1 + w["markov"] * markov_p1 + w["hb"] * hb_p1 + w["adv"] * adv_p1

    fat1 = fatigue_score(
        prof1.get("days_rest", 7),
        float(prof1.get("last_minutes", 90)),
        int(prof1.get("last_sets", 3)),
    ) * age_fatigue_mult(p1_key)
    fat2 = fatigue_score(
        prof2.get("days_rest", 7),
        float(prof2.get("last_minutes", 90)),
        int(prof2.get("last_sets", 3)),
    ) * age_fatigue_mult(p2_key)
    fat_adj_val = max(-0.10, min(0.10, (fat2 - fat1) * 0.015))

    form1_rate = prof1.get("form_rate", 0.5)
    form2_rate = prof2.get("form_rate", 0.5)
    global_form = (form1_rate - form2_rate) * 0.15
    surf_form   = surface_form_adj(p1_key, p2_key, surface)
    form_adj_val = max(-0.05, min(0.05, global_form * 0.6 + surf_form * 0.4))

    h2h_val     = h2h_adj(p1_key, p2_key, surface)
    clutch_val  = clutch_adj(p1_key, p2_key)
    df_val      = df_penalty_adj(p1_key, p2_key, is_wta)
    bh_val      = backhand_matchup_adj(p1_key, p2_key, surface)
    streak_val  = win_streak_adj(p1_key, p2_key)
    bo5_val     = bo5_adj(p1_key, p2_key, best_of)
    fs_val      = first_serve_adj(p1_key, p2_key)
    bp_atk_val  = bp_attack_adj(p1_key, p2_key)
    cond_val    = conditioning_adj(p1_key, p2_key)
    style_val      = playstyle_adj(p1_key, p2_key, surface)
    surf_trans_val = surface_transition_adj(p1_key, p2_key, prof1, prof2, surface)
    ret_depth_val  = return_depth_adj(p1_key, p2_key, surface, is_wta)
    inj_val        = injury_risk_adj(p1_key, p2_key)

    alt_adj    = altitude_adj(tournament, surface)
    # altitude shifts serve probability directly (same direction for both)
    p1_sv = max(0.50, min(0.78, p1_sv + alt_adj))
    p2_sv = max(0.50, min(0.78, p2_sv + alt_adj))

    wind_val, wind_kmh = wind_adj(tournament, surface, p1_key, p2_key)

    # v4.0: In BO5 (Grand Slams) mental/clutch game matters significantly more
    effective_clutch = clutch_val * (1.40 if best_of == 5 else 1.0)
    effective_clutch = max(-0.10, min(0.10, effective_clutch))

    blend_raw = (
        raw_prob + fat_adj_val + form_adj_val + h2h_val + effective_clutch
        + df_val + bh_val + streak_val + wind_val
        + bo5_val + fs_val + bp_atk_val + cond_val
        + style_val + surf_trans_val + ret_depth_val + inj_val
    )

    # v4.0: Probability calibration — regress extreme predictions toward 0.5
    # WTA gets extra dampening (α×0.96) due to higher match volatility
    calib_alpha = PROB_CALIB_ALPHA * (0.96 if is_wta else 1.0)
    blend_cal   = 0.5 + (blend_raw - 0.5) * calib_alpha
    blend = max(0.05, min(0.95, blend_cal))

    exp_g = expected_total_games(p1_sv, p2_sv, best_of=best_of)

    log.info(
        "predict %s vs %s [%s%s] ELO=%.3f MC=%.3f HB=%.3f ADV=%.3f raw=%.3f "
        "fat=%+.3f frm=%+.3f h2h=%+.3f clch=%+.3f df=%+.3f bh=%+.3f "
        "streak=%+.3f wind=%+.3f alt=%.3f bo5=%+.3f fs=%+.3f bp=%+.3f cond=%+.3f uncalib=%.3f -> %.3f",
        p1_key, p2_key, surface, " indoor" if cs_adj > 0.01 else "",
        elo_p1, markov_p1, hb_p1, adv_p1, raw_prob,
        fat_adj_val, form_adj_val, h2h_val, effective_clutch, df_val, bh_val,
        streak_val, wind_val, alt_adj, bo5_val, fs_val, bp_atk_val, cond_val,
        blend_raw, blend,
    )

    return {
        "blend_p1":        round(blend, 4),
        "model_p1":        round(markov_p1, 4),
        "elo_p1":          round(elo_p1, 4),
        "hb_p1":           round(hb_p1, 4),
        "adv_p1":          round(adv_p1, 4),
        "h2h_adj":         round(h2h_val, 4),
        "fat_adj":         round(fat_adj_val, 4),
        "form_adj":        round(form_adj_val, 4),
        "clutch_adj":      round(effective_clutch, 4),
        "df_adj":          round(df_val, 4),
        "lefty_adj":       round(lefty_sv, 4),
        "backhand_adj":    round(bh_val, 4),
        "p1_sv":           round(p1_sv, 4),
        "p2_sv":           round(p2_sv, 4),
        "elo1":            s1.get("elo", 1800),
        "elo2":            s2.get("elo", 1800),
        "fatigue1":        round(fat1, 1),
        "fatigue2":        round(fat2, 1),
        "form1":           round(form1_rate, 3),
        "form2":           round(form2_rate, 3),
        "tb_win1":         round(prof1.get("tb_win_pct") or 0.5, 3),
        "tb_win2":         round(prof2.get("tb_win_pct") or 0.5, 3),
        "bp_save1":        round(prof1.get("bp_save_pct") or 0.60, 3),
        "bp_save2":        round(prof2.get("bp_save_pct") or 0.60, 3),
        "df_rate1":        round(df1, 4),
        "df_rate2":        round(df2, 4),
        "ace_rate1":       round(prof1.get("ace_rate") or 0.06, 4),
        "ace_rate2":       round(prof2.get("ace_rate") or 0.06, 4),
        "expected_games":  round(exp_g, 1),
        "best_of":         best_of,
        "surface":         surface,
        "court_speed_adj": round(cs_adj, 4),
        "altitude_adj":    round(alt_adj, 4),
        "wind_adj":        round(wind_val, 4),
        "wind_kmh":        round(wind_kmh, 1),
        "streak_adj":      round(streak_val, 4),
        "win_streak1":     prof1.get("win_streak", 0),
        "win_streak2":     prof2.get("win_streak", 0),
        "bo5_adj":         round(bo5_val, 4),
        "fs_adj":          round(fs_val, 4),
        "bp_atk_adj":      round(bp_atk_val, 4),
        "cond_adj":        round(cond_val, 4),
        "bp_conv1":        round(prof1.get("bp_conv_pct") or 0.40, 3),
        "bp_conv2":        round(prof2.get("bp_conv_pct") or 0.40, 3),
        "fs_pct1":         round(prof1.get("first_serve_pct") or 0.60, 3),
        "fs_pct2":         round(prof2.get("first_serve_pct") or 0.60, 3),
        "load1":           prof1.get("matches_last_14d", 0),
        "load2":           prof2.get("matches_last_14d", 0),
        "is_wta":          is_wta,
        "style_adj":        round(style_val, 4),
        "surf_trans_adj":   round(surf_trans_val, 4),
        "ret_depth_adj":    round(ret_depth_val, 4),
        "inj_adj":          round(inj_val, 4),
        "model_weights":    {k: round(v, 3) for k, v in w.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# ODDS FETCHING & PARSING
# ─────────────────────────────────────────────────────────────────────────────
ODDS_SPORTS_BASE = ["tennis_atp", "tennis_wta"]
ODDS_SPORTS_SLAMS = [
    "tennis_atp_french_open", "tennis_wta_french_open",
    "tennis_atp_wimbledon",   "tennis_wta_wimbledon",
    "tennis_atp_us_open",     "tennis_wta_us_open",
    "tennis_atp_australian_open", "tennis_wta_australian_open",
]


def safe_get(url: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 404:
            log.debug("safe_get %s: 404 (inactive)", url.split("?")[0])
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("safe_get %s: %s", url, e)
        return None


def fetch_active_sports() -> set:
    """Call /v4/sports/ once to find which tennis sports are currently active."""
    data = safe_get(
        "https://api.the-odds-api.com/v4/sports/",
        params={"apiKey": ODDS_API_KEY},
    )
    if not data:
        return set()
    return {s["key"] for s in data if s.get("active") and "tennis" in s["key"]}


def fetch_odds() -> List[dict]:
    if not ODDS_API_KEY:
        log.warning("ODDS_API_KEY not set")
        return []
    active = fetch_active_sports()
    if not active:
        log.warning("fetch_active_sports returned empty — falling back to base sports")
        active = set(ODDS_SPORTS_BASE)
    # always include base; add slams only if active
    sports = list(ODDS_SPORTS_BASE) + [s for s in ODDS_SPORTS_SLAMS if s in active]
    log.info("fetch_odds: querying %d sport(s): %s", len(sports), sports)
    results = []
    for sport in sports:
        data = safe_get(
            "https://api.the-odds-api.com/v4/sports/%s/odds/" % sport,
            params={"apiKey": ODDS_API_KEY, "regions": "eu",
                    "markets": "h2h", "oddsFormat": "decimal"},
        )
        if data:
            for g in data:
                g["_sport"] = sport
            results.extend(data)
            log.info("  %s: %d games", sport, len(data))
    return results


def devigge(p1_raw: float, p2_raw: float) -> float:
    """
    v4.0: Power-method devig (multiplicative) — more accurate than additive for
    asymmetric tennis markets.  Finds exponent k s.t. p1^(1/k)+p2^(1/k)=1.
    Falls back to additive when margin is near-zero.
    """
    if p1_raw < 1e-6 or p2_raw < 1e-6:
        total = p1_raw + p2_raw
        return p1_raw / total if total > 1e-9 else 0.5
    margin = p1_raw + p2_raw - 1.0
    if margin < 0.005:   # near-zero overround: additive ≈ power
        return p1_raw / (p1_raw + p2_raw)
    # Binary search for power k such that p1^(1/k) + p2^(1/k) == 1
    lo, hi = 0.3, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if p1_raw ** (1.0 / mid) + p2_raw ** (1.0 / mid) > 1.0:
            lo = mid
        else:
            hi = mid
    k   = (lo + hi) / 2
    dp1 = p1_raw ** (1.0 / k)
    dp2 = p2_raw ** (1.0 / k)
    total = dp1 + dp2
    return dp1 / total if total > 1e-9 else 0.5


def parse_odds(raw: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for game in raw:
        home  = game.get("home_team", "")
        away  = game.get("away_team", "")
        books = game.get("bookmakers", [])
        if not home or not away or not books:
            continue
        hp, ap = [], []
        for bk in books:
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "h2h":
                    continue
                for oc in mkt.get("outcomes", []):
                    pr = float(oc.get("price", 1.0))
                    if oc.get("name") == home:
                        hp.append(pr)
                    elif oc.get("name") == away:
                        ap.append(pr)
        if len(hp) < MIN_BOOKS or len(ap) < MIN_BOOKS:
            continue
        best_h = max(hp); best_a = max(ap)
        cons_h = sum(hp) / len(hp); cons_a = sum(ap) / len(ap)
        dv_h   = devigge(1.0 / cons_h, 1.0 / cons_a)
        key    = "%s|%s" % (home.lower(), away.lower())
        out[key] = {
            "home": home, "away": away,
            "best_home": round(best_h, 3), "best_away": round(best_a, 3),
            "dv_p_home": round(dv_h, 4),  "dv_p_away": round(1.0 - dv_h, 4),
            "n_books":     len(hp),
            "sport":       game.get("_sport", ""),
            "sport_title": game.get("sport_title", ""),
            "commence":    game.get("commence_time", ""),
        }
    log.info("Parsed odds for %d matches", len(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LIVE ELO FROM TENNIS ABSTRACT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ta_elo_page(html: str) -> int:
    """Parse Tennis Abstract ELO HTML page, populate _LIVE_ELO. Returns count of players added."""
    import re as _re
    updated = 0
    # Match table rows: name link + 4 ELO columns (overall, hard, clay, grass)
    row_pat = _re.compile(
        r'<a[^>]+>([^<]{3,40})</a>'          # player name
        r'(?:.*?<td[^>]*>(\d{3,4})</td>){4}',  # 4 ELO numbers
        _re.DOTALL,
    )
    # Simpler pattern: find all anchor text + following td numbers
    cell_pat = _re.compile(r'<td[^>]*>(\d{3,4})</td>')
    name_pat  = _re.compile(r'<a[^>]+>([A-Z][a-z]+(?:[\s\-][A-Za-z]+){1,3})</a>')

    # Find all player rows by scanning for name links followed by numeric cells
    pos = 0
    while pos < len(html):
        nm = name_pat.search(html, pos)
        if nm is None:
            break
        name_end = nm.end()
        # Look for 4 ELO numbers in the next 300 chars
        chunk = html[name_end: name_end + 300]
        nums = cell_pat.findall(chunk)
        if len(nums) >= 4:
            elo_vals = [int(x) for x in nums[:4]]
            overall, hard_elo, clay_elo, grass_elo = elo_vals
            player_name = nm.group(1).strip()
            key = norm_player(player_name)
            if key and 1400 <= overall <= 3000:
                _LIVE_ELO[key] = {
                    "hard":  float(hard_elo),
                    "clay":  float(clay_elo),
                    "grass": float(grass_elo),
                }
                updated += 1
        pos = name_end
    return updated


def fetch_ta_elo() -> None:
    """Fetch current surface ELO for all ATP + WTA players from Tennis Abstract."""
    total = 0
    for tour_slug in ("atp_elo_ratings", "wta_elo_ratings"):
        url = f"https://tennisabstract.com/reports/{tour_slug}.html"
        try:
            r = requests.get(url, timeout=30,
                             headers={"User-Agent": "Mozilla/5.0 TennisBotResearch/1.0"})
            r.raise_for_status()
            n = _parse_ta_elo_page(r.text)
            log.info("fetch_ta_elo: %s → %d players", tour_slug, n)
            total += n
        except Exception as e:
            log.warning("fetch_ta_elo %s: %s", tour_slug, e)
    log.info("fetch_ta_elo total: %d players updated in _LIVE_ELO", total)


# ─────────────────────────────────────────────────────────────────────────────
# KELLY CRITERION & PICK GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def kelly_stake(model_p: float, price: float, conf: float = 1.0) -> float:
    if price <= 1.0 or model_p <= 0.0:
        return 0.0
    ev = model_p * price - 1.0
    if ev <= 0:
        return 0.0
    kf    = (model_p * price - 1.0) / (price - 1.0)
    stake = BANKROLL * kf * KELLY * conf
    return max(KELLY_FLOOR, min(KELLY_MAX, stake))


def generate_picks(matches: List[dict],
                   odds_prev: Optional[Dict[str, dict]] = None) -> List[dict]:
    picks     = []
    daily_exp = 0.0
    if odds_prev is None:
        odds_prev = {}

    for m in matches:
        if daily_exp >= MAX_DAILY_EXP:
            break
        pred      = m["pred"]
        odds_info = m["odds_info"]
        p1_key    = m["p1_key"]
        p2_key    = m["p2_key"]

        blend_p1 = pred["blend_p1"]
        blend_p2 = 1.0 - blend_p1
        dv_p1    = odds_info["dv_p_home"]
        dv_p2    = odds_info["dv_p_away"]

        # Data quality: shrink model toward market when player data is sparse
        dq = min(data_quality_score(p1_key), data_quality_score(p2_key))
        if dq < 0.85:
            shrink = (0.85 - dq) * 0.35  # up to ~23% shrink at dq=0.2
            blend_p1 = blend_p1 * (1 - shrink) + dv_p1 * shrink
            blend_p2 = 1.0 - blend_p1

        # 大冷門市場錨：推薦方市場機率 < 40% 且最佳賠率 > 3.0 時，額外向市場靠攏
        # 防止 Markov Chain 過度信任近期數據而無視市場共識
        _tmp_best = max(odds_info.get("best_home", 1.5), odds_info.get("best_away", 1.5))
        if dv_p1 < 0.40 and _tmp_best > 3.0:
            ug_shrink = min(0.30, (0.40 - dv_p1) * 1.5)
            blend_p1 = blend_p1 * (1 - ug_shrink) + dv_p1 * ug_shrink
            blend_p2 = 1.0 - blend_p1
        elif dv_p2 < 0.40 and _tmp_best > 3.0:
            ug_shrink = min(0.30, (0.40 - dv_p2) * 1.5)
            blend_p2 = blend_p2 * (1 - ug_shrink) + dv_p2 * ug_shrink
            blend_p1 = 1.0 - blend_p2

        # Odds movement signal (smart-money boost)
        ok = "%s|%s" % (odds_info["home"].lower(), odds_info["away"].lower())
        steam_adj, steam_label = odds_move_signal(ok, odds_info, odds_prev)
        blend_p1 = max(0.05, min(0.95, blend_p1 + steam_adj))
        blend_p2 = 1.0 - blend_p1

        # Market public bias fade
        bias_adj = public_bias_adj_fn(blend_p1, odds_info["dv_p_home"],
                                      blend_p2, odds_info["dv_p_away"])
        blend_p1 = max(0.05, min(0.95, blend_p1 + bias_adj))
        blend_p2 = 1.0 - blend_p1

        edge1 = blend_p1 - dv_p1
        edge2 = blend_p2 - dv_p2

        if edge1 >= edge2:
            edge, model_p, dv_p = edge1, blend_p1, dv_p1
            best_price, bet_name = odds_info["best_home"], odds_info["home"]
        else:
            edge, model_p, dv_p = edge2, blend_p2, dv_p2
            best_price, bet_name = odds_info["best_away"], odds_info["away"]

        # Market efficiency: more bookmakers → tighter market → require higher edge
        n_books = odds_info.get("n_books", MIN_BOOKS)
        eff_min_edge = MIN_EDGE_ML + max(0.0, (n_books - 6) * 0.003)
        # 高賠率安全閥：推薦的賠率 > 2.20 時要求更高 edge，避免模型噪音放大
        if best_price > 2.20:
            eff_min_edge = max(eff_min_edge, 0.10)
        # 市場強烈看空推薦方：市場給 < 42% 且對手市場 > 55%，需更高 edge 才信任模型
        opp_dv = dv_p1 if bet_name == odds_info["away"] else dv_p2
        if dv_p < 0.42 and opp_dv > 0.55:
            eff_min_edge = max(eff_min_edge, 0.14)
        # 有傷病訊號時提高 edge 門檻（傷病資訊不確定性高）
        has_inj = abs(pred.get("inj_adj", 0.0)) >= 0.01
        if has_inj:
            eff_min_edge = max(eff_min_edge, 0.10)
        # 數據品質低時提高 edge 門檻
        if dq < 0.65:
            eff_min_edge = max(eff_min_edge, 0.12)
        elif dq < 0.80:
            eff_min_edge = max(eff_min_edge, 0.09)
        # 市場機率底線：市場認為我們推薦的選手勝率 < 33% → 模型可能誤判，跳過
        if dv_p < 0.33:
            continue

        if edge < eff_min_edge:
            continue
        if model_p < MIN_CONF_ML:
            continue
        if p1_key in _INJURIES or p2_key in _INJURIES:
            continue

        # Tier classification before Kelly so we can use tier-specific fraction
        if edge >= 0.12:
            star = "\U0001f48e"; tier = "A"
        elif edge >= 0.09:
            star = "⭐"; tier = "B"
        else:
            star = "•"; tier = "C"

        conf         = min(1.0, (model_p - MIN_CONF_ML) * 2.0 + 0.70)
        kelly_frac   = KELLY_BY_TIER.get(tier, KELLY)
        kf_raw       = (model_p * best_price - 1.0) / (best_price - 1.0) if best_price > 1.0 else 0.0
        stake        = max(KELLY_FLOOR, min(KELLY_MAX, BANKROLL * kf_raw * kelly_frac * conf))
        stake        = min(stake, MAX_DAILY_EXP - daily_exp)
        daily_exp   += stake

        surface_emoji = {"clay": "\U0001f7e4", "grass": "\U0001f7e2", "hard": "\U0001f535"}.get(m["surface"], "⚪")

        picks.append({
            "tier":           tier,
            "star":           star,
            "surface_emoji":  surface_emoji,
            "tour":           TOUR_META.get(m["tour_level"], {}).get("name", m["tour_level"]),
            "tour_level":     m["tour_level"],
            "tour_type":      "WTA" if pred.get("is_wta") else "ATP",
            "tournament":     m.get("tournament", ""),
            "sport_title":    m.get("sport_title", ""),
            "surface":        m["surface"],
            "p1":             odds_info["home"],
            "p2":             odds_info["away"],
            "p1_cn":          cn_name(odds_info["home"]),
            "p2_cn":          cn_name(odds_info["away"]),
            "p1_key":         p1_key,
            "p2_key":         p2_key,
            "bet_on":         bet_name,
            "bet_on_cn":      cn_name(bet_name),
            "best_price":     round(best_price, 3),
            "model_p":        round(model_p * 100, 1),
            "dv_p":           round(dv_p * 100, 1),
            "edge":           round(edge * 100, 1),
            "conf":           round(conf * 100, 1),
            "stake":          round(stake, 0),
            "p1_sv_pct":      round(pred["p1_sv"] * 100, 1),
            "p2_sv_pct":      round(pred["p2_sv"] * 100, 1),
            "elo1":           pred["elo1"],
            "elo2":           pred["elo2"],
            "expected_games": pred["expected_games"],
            "best_of":        pred["best_of"],
            "h2h_adj":        round(pred["h2h_adj"] * 100, 1),
            "hb_p1":          round(pred.get("hb_p1", 0.5) * 100, 1),
            "form_adj":       round(pred.get("form_adj", 0.0) * 100, 1),
            "fat_adj":        round(pred.get("fat_adj", 0.0) * 100, 1),
            "fatigue1":       pred.get("fatigue1", 0.0),
            "fatigue2":       pred.get("fatigue2", 0.0),
            "form1":          round(pred.get("form1", 0.5) * 100, 1),
            "form2":          round(pred.get("form2", 0.5) * 100, 1),
            "commence":       odds_info.get("commence", ""),
            "adv_p1":         round(pred.get("adv_p1", 0.5) * 100, 1),
            "clutch_adj":     round(pred.get("clutch_adj", 0.0) * 100, 1),
            "df_adj":         round(pred.get("df_adj", 0.0) * 100, 1),
            "lefty_adj":      round(pred.get("lefty_adj", 0.0) * 100, 1),
            "backhand_adj":   round(pred.get("backhand_adj", 0.0) * 100, 1),
            "tb_win1":        round(pred.get("tb_win1", 0.5) * 100, 1),
            "tb_win2":        round(pred.get("tb_win2", 0.5) * 100, 1),
            "bp_save1":       round(pred.get("bp_save1", 0.6) * 100, 1),
            "bp_save2":       round(pred.get("bp_save2", 0.6) * 100, 1),
            "df_rate1":       round(pred.get("df_rate1", 0.04) * 100, 2),
            "df_rate2":       round(pred.get("df_rate2", 0.04) * 100, 2),
            "ace_rate1":      round(pred.get("ace_rate1", 0.06) * 100, 2),
            "ace_rate2":      round(pred.get("ace_rate2", 0.06) * 100, 2),
            "court_speed":    round(pred.get("court_speed_adj", 0.0) * 100, 2),
            "altitude_adj":   round(pred.get("altitude_adj", 0.0) * 100, 2),
            "wind_adj":       round(pred.get("wind_adj", 0.0) * 100, 2),
            "wind_kmh":       pred.get("wind_kmh", 0.0),
            "streak_adj":     round(pred.get("streak_adj", 0.0) * 100, 2),
            "win_streak1":    pred.get("win_streak1", 0),
            "win_streak2":    pred.get("win_streak2", 0),
            "bo5_adj":        round(pred.get("bo5_adj", 0.0) * 100, 2),
            "fs_adj":         round(pred.get("fs_adj", 0.0) * 100, 2),
            "bp_atk_adj":     round(pred.get("bp_atk_adj", 0.0) * 100, 2),
            "cond_adj":       round(pred.get("cond_adj", 0.0) * 100, 2),
            "bp_conv1":       round(pred.get("bp_conv1", 0.40) * 100, 1),
            "bp_conv2":       round(pred.get("bp_conv2", 0.40) * 100, 1),
            "fs_pct1":        round(pred.get("fs_pct1", 0.60) * 100, 1),
            "fs_pct2":        round(pred.get("fs_pct2", 0.60) * 100, 1),
            "load1":          pred.get("load1", 0),
            "load2":          pred.get("load2", 0),
            "steam":          steam_label,
            "n_books":        n_books,
            "eff_min_edge":   round(eff_min_edge * 100, 1),
            "style_adj":       round(pred.get("style_adj", 0.0) * 100, 2),
            "surf_trans_adj":  round(pred.get("surf_trans_adj", 0.0) * 100, 2),
            "ret_depth_adj":   round(pred.get("ret_depth_adj", 0.0) * 100, 2),
            "inj_adj":         round(pred.get("inj_adj", 0.0) * 100, 2),
            "public_bias_adj": round(bias_adj * 100, 2),
            "opening_dv_p":    round(dv_p * 100, 1),  # CLV tracking
        })
        log.info("  PICK %s %s vs %s -> %s @%.2f model=%.1f%% edge=+%.1f%% $%.0f",
                 star, odds_info["home"], odds_info["away"],
                 bet_name, best_price, model_p * 100, edge * 100, stake)

    picks.sort(key=lambda x: -x["edge"])
    return picks[:MAX_PICKS]


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY (GitHub Gist)
# ─────────────────────────────────────────────────────────────────────────────

def load_history() -> dict:
    if not GIST_TOKEN or not GIST_ID:
        return {"bets": []}
    try:
        r = requests.get(
            "https://api.github.com/gists/%s" % GIST_ID,
            headers={"Authorization": "token %s" % GIST_TOKEN},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("load_history: %s", e)
        return {"bets": []}
    for fname, fd in data.get("files", {}).items():
        if fname.endswith(".json"):
            try:
                return json.loads(fd.get("content", "{}"))
            except json.JSONDecodeError:
                pass
    return {"bets": []}


def save_history(hist: dict) -> None:
    if not GIST_TOKEN or not GIST_ID:
        return
    try:
        requests.patch(
            "https://api.github.com/gists/%s" % GIST_ID,
            headers={"Authorization": "token %s" % GIST_TOKEN},
            json={"files": {"tennis_hist.json": {
                "content": json.dumps(hist, ensure_ascii=False, indent=2)}}},
            timeout=15,
        )
    except Exception as e:
        log.warning("save_history: %s", e)


def picks_starting_soon(picks: List[dict], now_utc: datetime.datetime,
                         hours: float = 2.0) -> List[dict]:
    """Return picks whose match commence time is within `hours` from now (UTC)
    AND whose match date is today in Taiwan time (UTC+8)."""
    now_tw_date = (now_utc + datetime.timedelta(hours=8)).date()
    result = []
    for p in picks:
        commence = p.get("commence", "")
        if not commence:
            continue
        try:
            mt = datetime.datetime.fromisoformat(commence.replace("Z", "+00:00"))
            mt_naive = mt.replace(tzinfo=None) if mt.tzinfo else mt
            diff_h = (mt_naive - now_utc).total_seconds() / 3600.0
            mt_tw_date = (mt_naive + datetime.timedelta(hours=8)).date()
            if 0.0 <= diff_h <= hours and mt_tw_date == now_tw_date:
                result.append(p)
        except Exception:
            pass
    return result


_SLAM_KEYS = {"french_open", "wimbledon", "us_open", "australian_open"}


def filter_slam_picks(picks: List[dict]) -> List[dict]:
    """Return only picks belonging to the currently active Grand Slam."""
    slam_picks = [p for p in picks if p.get("tournament", "") in _SLAM_KEYS]
    if not slam_picks:
        return []
    from collections import Counter
    dominant = Counter(p["tournament"] for p in slam_picks).most_common(1)[0][0]
    return [p for p in slam_picks if p["tournament"] == dominant]


def record_picks_to_history(picks: List[dict], hist: dict,
                             now_tw: datetime.datetime) -> None:
    """Append today's picks as pending bets if not already recorded."""
    bets   = hist.setdefault("bets", [])
    today  = now_tw.strftime("%Y-%m-%d")
    existing = {
        (b["p1"], b["p2"], b["date"])
        for b in bets
        if "p1" in b and "p2" in b
    }
    added = 0
    for p in picks:
        key = (p["p1"], p["p2"], today)
        if key in existing:
            continue
        bets.append({
            "date":         today,
            "p1":           p["p1"],
            "p2":           p["p2"],
            "bet_on":       p["bet_on"],
            "price":        p["best_price"],
            "stake":        p["stake"],
            "edge":         p["edge"],
            "tier":         p["tier"],
            "surface":      p["surface"],
            "tour":         p["tour"],
            "result":       "P",
            "opening_dv_p": p.get("opening_dv_p"),
            "model_p":      p.get("model_p"),
        })
        existing.add(key)
        added += 1
    log.info("record_picks_to_history: +%d new bets (total %d)", added, len(bets))


def compute_stats(hist: dict) -> dict:
    bets = [b for b in hist.get("bets", []) if b.get("result") in ("W", "L")]
    if not bets:
        return {"settled": 0, "wins": 0, "win_rate": 0.0, "roi": 0.0, "pnl": 0.0}
    wins     = sum(1 for b in bets if b["result"] == "W")
    total_in = sum(b.get("stake", 100) for b in bets)
    pnl      = sum(
        b.get("stake", 100) * (b.get("price", 2.0) - 1) if b["result"] == "W"
        else -b.get("stake", 100)
        for b in bets
    )
    return {
        "settled":  len(bets), "wins": wins,
        "win_rate": round(wins / len(bets) * 100, 1),
        "pnl":      round(pnl, 1),
        "roi":      round(pnl / total_in * 100, 1) if total_in > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────

def send_ntfy(title: str, message: str) -> None:
    if not NTFY_TOPIC:
        return
    try:
        requests.post("https://ntfy.sh",
                      json={"topic": NTFY_TOPIC, "title": title,
                            "message": message, "priority": 4, "tags": ["tennis"]},
                      timeout=10)
    except Exception as e:
        log.warning("ntfy: %s", e)


def send_discord(picks: List[dict], stats: dict, is_recording: bool = False) -> None:
    if not DISCORD_HOOK:
        return
    now   = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    label  = "🟢 紀錄時間" if is_recording else "⚪ 非紀錄時間"
    lines = ["**\U0001f3be ATP/WTA 每日預測 — %s**" % now.strftime("%Y-%m-%d %H:%M"), "```"]
    if not picks:
        lines.append("今日無符合條件的推薦")
    else:
        for p in picks:
            p1d = p.get("p1_cn") or p["p1"]
            p2d = p.get("p2_cn") or p["p2"]
            bnd = p.get("bet_on_cn") or p["bet_on"]
            commence_str = ""
            if p.get("commence"):
                try:
                    ct = datetime.datetime.fromisoformat(p["commence"].replace("Z", "+00:00"))
                    ct_tw = ct.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
                    commence_str = "  🕐 %s (台灣時間)" % ct_tw.strftime("%m/%d %H:%M")
                except Exception:
                    pass
            lines.append("%s %s [%s] %s vs %s%s" % (
                p["star"], p["surface_emoji"], p.get("tour_type", "ATP"), p1d, p2d, commence_str))
            lines.append("  推薦: %s @%.2f  模型:%.1f%%  edge:+%.1f%%  $%.0f" % (
                bnd, p["best_price"], p["model_p"], p["edge"], p["stake"]))
            adj_parts = []
            if p.get("fat_adj"):      adj_parts.append("體能:%+.1f%%" % p["fat_adj"])
            if p.get("form_adj"):     adj_parts.append("狀態:%+.1f%%" % p["form_adj"])
            if p.get("clutch_adj"):   adj_parts.append("心理:%+.1f%%" % p["clutch_adj"])
            if p.get("df_adj"):       adj_parts.append("雙誤:%+.1f%%" % p["df_adj"])
            if p.get("lefty_adj"):    adj_parts.append("左手:%+.1f%%" % p["lefty_adj"])
            if p.get("streak_adj"):   adj_parts.append("連勝:%+.1f%%" % p["streak_adj"])
            if p.get("bo5_adj"):      adj_parts.append("BO5:%+.1f%%" % p["bo5_adj"])
            if p.get("bp_atk_adj"):   adj_parts.append("破發攻:%+.1f%%" % p["bp_atk_adj"])
            if p.get("cond_adj"):     adj_parts.append("負荷:%+.1f%%" % p["cond_adj"])
            if p.get("wind_kmh", 0) > 15: adj_parts.append("風:%.0fkm/h" % p["wind_kmh"])
            if p.get("steam"):        adj_parts.append("💰%s" % p["steam"].replace("_"," "))
            if p.get("style_adj"):     adj_parts.append("風格:%+.1f%%" % p["style_adj"])
            if p.get("surf_trans_adj"):adj_parts.append("場轉:%+.1f%%" % p["surf_trans_adj"])
            if p.get("inj_adj"):       adj_parts.append("傷病:%+.1f%%" % p["inj_adj"])
            if p.get("public_bias_adj"):adj_parts.append("市偏:%+.1f%%" % p["public_bias_adj"])
            if adj_parts:
                lines.append("  " + "  ".join(adj_parts))
            lines.append("  一發:%.0f%%/%.0f%%  破發轉換:%.0f%%/%.0f%%  負荷:%d/%d場" % (
                p.get("fs_pct1", 60), p.get("fs_pct2", 60),
                p.get("bp_conv1", 40), p.get("bp_conv2", 40),
                p.get("load1", 0),    p.get("load2", 0)))
    lines.append("```")
    if stats.get("settled", 0):
        lines.append("戰績: %d/%d (%.1f%%)  ROI: %.1f%%" % (
            stats["wins"], stats["settled"], stats["win_rate"], stats["roi"]))
    try:
        requests.post(DISCORD_HOOK, json={"content": "\n".join(lines)}, timeout=10)
    except Exception as e:
        log.warning("discord: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def write_json(picks: List[dict], stats: dict, history: dict,
               game_preds: dict, now: datetime.datetime) -> None:
    os.makedirs("docs", exist_ok=True)
    payload = {
        "generated_at":     now.strftime("%Y-%m-%d %H:%M") + " (台灣時間)",
        "generated_at_iso": now.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "date":             now.strftime("%Y-%m-%d"),
        "model_version": "v4.0 — PowerDevig+DynH2H+ExpForm+RecencyELO+ProbCalib+15factor",
        "stats":         stats,
        "picks":         picks,
        "recent_history": list(reversed(history.get("bets", [])[-10:])),
        "live_matches":  [],
        "game_preds":    game_preds,
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s (%d picks)", JSON_PATH, len(picks))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    now_tw = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    log.info("=== Tennis Bot v4.0 start %s ===", now_tw.strftime("%Y-%m-%d %H:%M"))

    # Fetch current ATP/WTA rankings for dynamic rank-based stats on any player
    fetch_current_rankings()
    log.info("Live ranks loaded: %d players", len(_LIVE_RANKS))

    all_matches_raw = fetch_sackmann_matches()

    load_sackmann_data(all_matches_raw)

    # If Sackmann data unavailable, fall back to Tennis Abstract live ELO
    if not all_matches_raw:
        log.info("Sackmann data unavailable — fetching Tennis Abstract live ELO as fallback")
        fetch_ta_elo()

    auto_inj = detect_injuries(all_matches_raw) if all_matches_raw else set()
    _INJURIES.update(auto_inj)
    if auto_inj:
        log.info("Auto-flagged injuries/retirements: %s", auto_inj)

    odds_prev = load_odds_prev()

    raw_odds = fetch_odds()
    odds_map = parse_odds(raw_odds)

    save_odds_prev(odds_map)

    matches:    List[dict]      = []
    game_preds: Dict[str, dict] = {}

    for key, odds_info in odds_map.items():
        sport      = odds_info.get("sport", "tennis_atp")
        surface    = infer_surface(sport)
        t_lvl      = infer_tour_level(sport)
        best_of    = TOUR_META.get(t_lvl, {}).get("best_of", 3)
        tournament = extract_tournament(sport, odds_info)

        _tour_hint = "wta" if "wta" in sport else "atp"
        p1_key = norm_player(odds_info["home"], tour=_tour_hint)
        p2_key = norm_player(odds_info["away"], tour=_tour_hint)

        pred = predict(p1_key, p2_key, surface, t_lvl, best_of,
                       sport_key=sport, tournament=tournament)

        game_preds[key] = {
            "p1": odds_info["home"], "p2": odds_info["away"],
            "p1_key": p1_key, "p2_key": p2_key,
            "model_p1":   pred["blend_p1"],
            "surface":    surface,
            "tour_level": t_lvl,
            "best_of":    best_of,
            "exp_games":  pred["expected_games"],
        }
        matches.append({
            "p1_key": p1_key, "p2_key": p2_key,
            "surface": surface, "tour_level": t_lvl, "best_of": best_of,
            "odds_info": odds_info, "pred": pred,
            "tournament": tournament,
            "sport_title": odds_info.get("sport_title", ""),
        })

    log.info("Processed %d matches", len(matches))

    picks   = generate_picks(matches, odds_prev=odds_prev)
    history = load_history()

    now_utc = datetime.datetime.utcnow()
    soon_picks = picks_starting_soon(picks, now_utc, hours=2.0)
    if soon_picks:
        log.info("Recording %d picks starting within 2h to Gist", len(soon_picks))
        record_picks_to_history(soon_picks, history, now_tw)
        save_history(history)
    else:
        log.info("No picks starting within 2h — skip Gist write")

    stats   = compute_stats(history)
    write_json(picks, stats, history, game_preds, now_tw)

    if picks:
        send_ntfy(
            "\U0001f3be Tennis Picks — %s" % now_tw.strftime("%m/%d"),
            "%d 個推薦\n" % len(picks) +
            "\n".join("• %s vs %s → %s @%.2f (+%.1f%%)" % (
                p["p1"], p["p2"], p["bet_on"], p["best_price"], p["edge"]
            ) for p in picks),
        )
    send_discord(picks, stats, bool(soon_picks))
    log.info("=== Done — %d picks ===", len(picks))


if __name__ == "__main__":
    run()
