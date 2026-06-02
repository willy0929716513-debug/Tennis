#!/usr/bin/env python3
"""
Tennis Bot v3.2 — ATP/WTA 巡迴賽預測系統
9因子模型：Surface ELO 2.0 + H2H + Fatigue + Form + 戰術適配 + 賠率逆推
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("tennis_bot")

# ══════════════════════════════════════════════════════════════════════════════
# 0. 環境變數 / 常數
# ══════════════════════════════════════════════════════════════════════════════
TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ODDS_KEY = os.environ.get("ODDS_API_KEY", "")
SPORTS_KEY = os.environ.get("SPORTS_DATA_KEY", "")

# The-Odds-API
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_TENNIS_ATP = "tennis_atp"
SPORT_TENNIS_WTA = "tennis_wta"

# Rapid / SportsData (fallback)
RAPID_BASE = "https://tennis-live-data.p.rapidapi.com"

# 本地持久化
DATA_FILE = "tennis_data.json"

# ── 模型超參 ─────────────────────────────────────────────────────────────────
K_FACTOR = 32          # ELO K 值
INITIAL_ELO = 1500
DECAY_DAYS = 180       # 不活躍衰減週期
FATIGUE_DECAY = 0.85   # 每場疲勞衰減
FORM_WINDOW = 10       # 近期戰績窗口

# ── 信心閾值 ─────────────────────────────────────────────────────────────────
MIN_CONFIDENCE = 55.0  # 低於此不推薦
EDGE_THRESHOLD = 5.0   # 模型 vs 賠率隱含機率差

# ══════════════════════════════════════════════════════════════════════════════
# 1. 資料結構
# ══════════════════════════════════════════════════════════════════════════════

SURFACES = ("hard", "clay", "grass", "carpet", "indoor_hard")

PLAYSTYLE = {
    # ── Top ATP (active / recent) ──────────────────────────────────────────
    "Carlos Alcaraz":        "all_court",
    "Jannik Sinner":         "baseliner",
    "Novak Djokovic":        "all_court",
    "Rafael Nadal":          "clay_specialist",
    "Roger Federer":         "all_court",
    "Daniil Medvedev":       "baseliner",
    "Alexander Zverev":      "baseliner",
    "Andrey Rublev":         "baseliner",
    "Casper Ruud":           "clay_specialist",
    "Stefanos Tsitsipas":    "all_court",
    "Holger Rune":           "all_court",
    "Hubert Hurkacz":        "serve_volleyer",
    "Taylor Fritz":          "baseliner",
    "Ben Shelton":           "serve_volleyer",
    "Frances Tiafoe":        "all_court",
    "Tommy Paul":            "baseliner",
    "Sebastian Korda":       "all_court",
    "Alex de Minaur":        "baseliner",
    "Felix Auger-Aliassime": "all_court",
    "Denis Shapovalov":      "all_court",
    "Grigor Dimitrov":       "all_court",
    "Lorenzo Musetti":       "all_court",
    "Matteo Berrettini":     "serve_volleyer",
    "Gael Monfils":          "baseliner",
    "Stan Wawrinka":         "all_court",
    "Dominic Thiem":         "baseliner",
    "David Goffin":          "baseliner",
    "Marin Cilic":           "serve_volleyer",
    "Roberto Bautista Agut": "baseliner",
    "Pablo Carreno Busta":   "baseliner",
    "Albert Ramos-Vinolas":  "clay_specialist",
    "Diego Schwartzman":     "clay_specialist",
    "Federico Delbonis":     "clay_specialist",
    "Guido Pella":           "clay_specialist",
    "Nicolas Jarry":         "serve_volleyer",
    "Ugo Humbert":           "serve_volleyer",
    "Adrian Mannarino":      "baseliner",
    "Gilles Simon":          "baseliner",
    "Jo-Wilfried Tsonga":    "all_court",
    "Richard Gasquet":       "all_court",
    "Jeremy Chardy":         "serve_volleyer",
    "Lucas Pouille":         "baseliner",
    "Pierre-Hugues Herbert": "serve_volleyer",
    "Benoit Paire":          "baseliner",
    "Nicolas Mahut":         "serve_volleyer",
    "Edouard Roger-Vasselin":"serve_volleyer",
    "Fabrice Martin":        "serve_volleyer",
    "Jonathan Eysseric":     "baseliner",
    "Quentin Halys":         "baseliner",
    "Corentin Moutet":       "all_court",
    "Arthur Rinderknech":    "serve_volleyer",
    "Hugo Gaston":           "all_court",
    "Luca Van Assche":       "baseliner",
    "Giovanni Mpetshi Perricard": "serve_volleyer",
    "Harold Mayot":          "baseliner",
    "Maxime Janvier":        "baseliner",
    "Clement Chidekh":       "serve_volleyer",
    "Manuel Guinard":        "clay_specialist",
    "Terence Atmane":        "serve_volleyer",
    "Matteo Arnaldi":        "all_court",
    "Flavio Cobolli":        "baseliner",
    "Luciano Darderi":       "clay_specialist",
    "Francesco Passaro":     "clay_specialist",
    "Giulio Zeppieri":       "clay_specialist",
    "Luca Nardi":            "baseliner",
    "Andrea Vavassori":      "serve_volleyer",
    "Jannik Sinner":         "baseliner",   # duplicate handled by last-wins
    "Fabio Fognini":         "baseliner",
    "Marco Cecchinato":      "clay_specialist",
    "Gianluca Mager":        "baseliner",
    "Thomas Fabbiano":       "baseliner",
    "Stefano Travaglia":     "baseliner",
    "Andreas Seppi":         "baseliner",
    "Paolo Lorenzi":         "clay_specialist",
    "Filippo Volandri":      "clay_specialist",
    "Camilo Ugo Carabelli":  "clay_specialist",
    "Pedro Cachin":          "clay_specialist",
    "Facundo Bagnis":        "clay_specialist",
    "Federico Coria":        "clay_specialist",
    "Tomas Martin Etcheverry": "clay_specialist",
    "Sebastian Baez":        "clay_specialist",
    "Francisco Cerundolo":   "clay_specialist",
    "Juan Manuel Cerundolo": "clay_specialist",
    "Mariano Navone":        "clay_specialist",
    "Thiago Agustin Tirante":"clay_specialist",
    "Renzo Olivo":           "clay_specialist",
    "Juan Ignacio Londero":  "clay_specialist",
    "Leonardo Mayer":        "clay_specialist",
    "Horacio Zeballos":      "serve_volleyer",
    "Maximilian Marterer":   "baseliner",
    "Yannick Hanfmann":      "baseliner",
    "Oscar Otte":            "baseliner",
    "Daniel Altmaier":       "clay_specialist",
    "Dominik Koepfer":       "clay_specialist",
    "Peter Gojowczyk":       "baseliner",
    "Philipp Kohlschreiber":  "all_court",
    "Florian Mayer":         "all_court",
    "Tommy Haas":            "all_court",
    "Michael Stich":         "serve_volleyer",
    "Boris Becker":          "serve_volleyer",
    "Nikoloz Basilashvili":  "baseliner",
    "Botic van de Zandschulp": "baseliner",
    "Tallon Griekspoor":     "serve_volleyer",
    "Tim van Rijthoven":     "serve_volleyer",
    "Gijs Brouwer":          "baseliner",
    "Jesper de Jong":        "baseliner",
    "Robin Haase":           "baseliner",
    "Thiemo de Bakker":      "baseliner",
    "Igor Sijsling":         "baseliner",
    "Jiri Vesely":           "serve_volleyer",
    "Jiri Lehecka":          "all_court",
    "Tomas Machac":          "all_court",
    "Jakub Mensik":          "all_court",
    "Adam Pavlasek":         "serve_volleyer",
    "Lukas Rosol":           "serve_volleyer",
    "Radek Stepanek":        "serve_volleyer",
    "Tomas Berdych":         "baseliner",
    "Lukas Lacko":           "baseliner",
    "Martin Klizan":         "baseliner",
    "Norbert Gombos":        "baseliner",
    "Andrej Martin":         "clay_specialist",
    "Jozef Kovalik":         "clay_specialist",
    "Alexei Popyrin":        "serve_volleyer",
    "Jordan Thompson":       "baseliner",
    "John Millman":          "baseliner",
    "James Duckworth":       "baseliner",
    "Nick Kyrgios":          "serve_volleyer",
    "Thanasi Kokkinakis":     "serve_volleyer",
    "Christopher O'Connell": "baseliner",
    "Rinky Hijikata":        "baseliner",
    "Max Purcell":           "serve_volleyer",
    "Marc Polmans":          "baseliner",
    "Lleyton Hewitt":        "baseliner",
    "Patrick Rafter":        "serve_volleyer",
    "Mark Philippoussis":    "serve_volleyer",
    "Kei Nishikori":         "all_court",
    "Yoshihito Nishioka":    "baseliner",
    "Taro Daniel":           "baseliner",
    "Sho Shimabukuro":       "baseliner",
    "Yosuke Watanuki":       "baseliner",
    "Zsolt Borsos":          "baseliner",
    "Marton Fucsovics":      "baseliner",
    "Attila Balazs":         "baseliner",
    "Fabian Marozsan":       "clay_specialist",
    "Mate Valkusz":          "baseliner",
    "Laslo Djere":           "clay_specialist",
    "Dusan Lajovic":         "clay_specialist",
    "Hamad Medjedovic":      "clay_specialist",
    "Miomir Kecmanovic":     "baseliner",
    "Filip Krajinovic":      "baseliner",
    "Viktor Troicki":        "serve_volleyer",
    "Janko Tipsarevic":      "baseliner",
    "Nenad Zimonjic":        "serve_volleyer",
    "Mikhail Youzhny":       "baseliner",
    "Mikhail Kukushkin":     "baseliner",
    "Evgeny Donskoy":        "baseliner",
    "Karen Khachanov":       "baseliner",
    "Aslan Karatsev":        "baseliner",
    "Roman Safiullin":       "baseliner",
    "Pavel Kotov":           "baseliner",
    "Alexey Vatutin":        "baseliner",
    "Teymuraz Gabashvili":   "baseliner",
    "Dmitry Tursunov":       "baseliner",
    "Igor Andreev":          "baseliner",
    "Mikhail Elgin":         "baseliner",
    "Andrei Golubev":        "baseliner",
    "Illya Marchenko":       "baseliner",
    "Sergiy Stakhovsky":     "serve_volleyer",
    "Alexandr Dolgopolov":   "all_court",
    "Damir Dzumhur":         "baseliner",
    "Mirza Basic":           "baseliner",
    "Tomislav Brkic":        "baseliner",
    "Nerman Fatic":          "baseliner",
    "Constant Lestienne":    "all_court",
    "Gregoire Barrere":      "baseliner",
    "Maxime Cressy":         "serve_volleyer",
    "John Isner":            "serve_volleyer",
    "Jack Sock":             "all_court",
    "Steve Johnson":         "serve_volleyer",
    "Sam Querrey":           "serve_volleyer",
    "Reilly Opelka":         "serve_volleyer",
    "Brandon Nakashima":     "baseliner",
    "Jenson Brooksby":       "baseliner",
    "Marcos Giron":          "baseliner",
    "Mackenzie McDonald":    "baseliner",
    "Stefan Kozlov":         "baseliner",
    "Michael Mmoh":          "baseliner",
    "J.J. Wolf":             "serve_volleyer",
    "Christopher Eubanks":   "serve_volleyer",
    "Emilio Nava":           "baseliner",
    "Learner Tien":          "baseliner",
    "Alex Michelsen":        "baseliner",
    "Ethan Quinn":           "baseliner",
    "Aleksandar Kovacevic":  "serve_volleyer",
    "Denis Kudla":           "baseliner",
    "Ryan Harrison":         "serve_volleyer",
    "Tennys Sandgren":       "baseliner",
    "Tim Smyczek":           "baseliner",
    "Donald Young":          "all_court",
    "James Blake":           "baseliner",
    "Andy Roddick":          "serve_volleyer",
    "Pete Sampras":          "serve_volleyer",
    "Andre Agassi":          "baseliner",
    "Jim Courier":           "baseliner",
    "Michael Chang":         "baseliner",
    "Andre Sa":              "serve_volleyer",
    "Pedro Sousa":           "clay_specialist",
    "Gastao Elias":          "clay_specialist",
    "Joao Sousa":            "clay_specialist",
    "Nuno Borges":           "baseliner",
    "Jaime Faria":           "baseliner",
    "Daniel Elahi Galan":    "clay_specialist",
    "Nicolas Barrientos":    "clay_specialist",
    "Santiago Giraldo":      "clay_specialist",
    "Alejandro Gonzalez":    "clay_specialist",
    "Harold Arango":         "clay_specialist",
    "Emilio Gomez":          "clay_specialist",
    "Roberto Quiroz":        "baseliner",
    "Gonzalo Lama":          "clay_specialist",
    "Cristian Garin":        "clay_specialist",
    "Nicolas Jarry":         "serve_volleyer",  # already above — last wins
    "Alejandro Tabilo":      "all_court",
    "Tomas Barrios Vera":    "baseliner",
    "Hans Podlipnik-Castillo": "clay_specialist",
    "Marcelo Tomas Barrios": "clay_specialist",
    "Pablo Andujar":         "clay_specialist",
    "Jaume Munar":           "clay_specialist",
    "Bernabe Zapata Miralles":"clay_specialist",
    "Pedro Martinez":        "clay_specialist",
    "Carlos Taberner":       "clay_specialist",
    "Roberto Carballes Baena":"clay_specialist",
    "Fernando Verdasco":     "baseliner",
    "Nicolas Almagro":       "clay_specialist",
    "David Ferrer":          "clay_specialist",
    "Tommy Robredo":         "clay_specialist",
    "Albert Costa":          "clay_specialist",
    "Alex Corretja":         "clay_specialist",
    "Arantxa Rus":           "baseliner",
    "Qualifier":             "baseliner",   # generic fallback
    "Lucky Loser":           "baseliner",
    # ── Top WTA (active) ─────────────────────────────────────────────────────
    "Aryna Sabalenka":       "baseliner",
    "Iga Swiatek":           "clay_specialist",
    "Coco Gauff":            "all_court",
    "Elena Rybakina":        "serve_volleyer",
    "Jessica Pegula":        "baseliner",
    "Mirra Andreeva":        "baseliner",
    "Daria Kasatkina":       "all_court",
    "Madison Keys":          "baseliner",
    "Jasmine Paolini":       "all_court",
    "Emma Navarro":          "baseliner",
    "Paula Badosa":          "all_court",
    "Beatriz Haddad Maia":   "clay_specialist",
    "Liudmila Samsonova":    "baseliner",
    "Anna Kalinskaya":       "baseliner",
    "Karolina Pliskova":     "serve_volleyer",
    "Maria Sakkari":         "baseliner",
    "Barbora Krejcikova":    "all_court",
    "Marketa Vondrousova":   "all_court",
    "Elina Svitolina":       "baseliner",
    "Belinda Bencic":        "all_court",
    "Victoria Azarenka":     "baseliner",
    "Caroline Wozniacki":    "baseliner",
    "Angelique Kerber":      "all_court",
    "Simona Halep":          "all_court",
    "Petra Kvitova":         "serve_volleyer",
    "Garbine Muguruza":      "all_court",
    "Sloane Stephens":       "baseliner",
    "Sofia Kenin":           "all_court",
    "Bianca Andreescu":      "all_court",
    "Naomi Osaka":           "baseliner",
    "Serena Williams":       "baseliner",
    "Venus Williams":        "serve_volleyer",
    "Martina Navratilova":   "serve_volleyer",
    "Steffi Graf":           "baseliner",
    "Monica Seles":          "baseliner",
    "Alycia Parks":          "serve_volleyer",
    "Bernarda Pera":         "baseliner",
    "Claire Liu":            "baseliner",
    "Peyton Stearns":        "baseliner",
    "Amanda Anisimova":      "baseliner",
    "Sofia Kenin":           "all_court",   # dup
    "Katie Boulter":         "serve_volleyer",
    "Harriet Dart":          "baseliner",
    "Heather Watson":        "baseliner",
    "Johanna Konta":         "baseliner",
    "Anne Keothavong":       "baseliner",
    "Laura Robson":          "baseliner",
    "Emma Raducanu":         "baseliner",
    "Leylah Fernandez":      "all_court",
    "Bianca Andreescu":      "all_court",   # dup
    "Eugenie Bouchard":      "baseliner",
    "Rebecca Marino":        "baseliner",
    "Carol Zhao":            "baseliner",
    "Gabriela Dabrowski":    "serve_volleyer",
    "Sharon Fichman":        "serve_volleyer",
    "Kristina Mladenovic":   "baseliner",
    "Caroline Garcia":       "baseliner",
    "Oceane Dodin":          "serve_volleyer",
    "Fiona Ferro":           "all_court",
    "Alize Cornet":          "all_court",
    "Pauline Parmentier":    "baseliner",
    "Virginie Razzano":      "clay_specialist",
    "Marion Bartoli":        "all_court",
    "Amelie Mauresmo":       "all_court",
    "Tatiana Golovin":       "baseliner",
    "Mary Pierce":           "baseliner",
    "Nathalie Tauziat":      "serve_volleyer",
    "Camille Pin":           "baseliner",
    "Stephanie Cohen-Aloro": "clay_specialist",
    "Lucie Safarova":        "all_court",
    "Karolina Muchova":      "all_court",
    "Linda Noskova":         "baseliner",
    "Marie Bouzkova":        "baseliner",
    "Tereza Martincova":     "baseliner",
    "Petra Martic":          "clay_specialist",
    "Donna Vekic":           "all_court",
    "Ana Bogdan":            "clay_specialist",
    "Irina-Camelia Begu":    "clay_specialist",
    "Sorana Cirstea":        "baseliner",
    "Monica Niculescu":      "all_court",
    "Simona Halep":          "all_court",   # dup
    "Danka Kovinic":         "baseliner",
    "Olga Danilovic":        "baseliner",
    "Aleksandra Krunic":     "baseliner",
    "Nina Stojanovic":       "baseliner",
    "Natalija Stevanovic":   "baseliner",
    "Ana Konjuh":            "serve_volleyer",
    "Ajla Tomljanovic":      "all_court",
    "Tamara Zidansek":       "clay_specialist",
    "Polona Hercog":         "clay_specialist",
    "Kaja Juvan":            "baseliner",
    "Lesia Tsurenko":        "baseliner",
    "Dayana Yastremska":     "baseliner",
    "Marta Kostyuk":         "baseliner",
    "Anhelina Kalinina":     "clay_specialist",
    "Katarina Zavatska":     "baseliner",
    "Yuliia Starodubtseva":  "baseliner",
    "Veronika Kudermetova":  "baseliner",
    "Anastasia Pavlyuchenkova": "baseliner",
    "Nadia Podoroska":       "clay_specialist",
    "Ekaterina Alexandrova": "baseliner",
    "Vera Zvonareva":        "baseliner",
    "Elena Vesnina":         "all_court",
    "Svetlana Kuznetsova":   "all_court",
    "Dinara Safina":         "baseliner",
    "Vera Dushevina":        "baseliner",
    "Galina Voskoboeva":     "baseliner",
    "Vitalia Diatchenko":    "serve_volleyer",
    "Oksana Kalashnikova":   "baseliner",
    "Nadiia Kichenok":       "serve_volleyer",
    "Viktoriya Tomova":      "baseliner",
    "Tsvetana Pironkova":    "baseliner",
    "Sesil Karatantcheva":   "baseliner",
    "Elitsa Kostova":        "baseliner",
    "Aranxta Rus":           "baseliner",
    "Richel Hogenkamp":      "baseliner",
    "Eva Wacanno":           "baseliner",
    "Quirine Lemoine":       "baseliner",
    "Leonie Kung":           "baseliner",
    "Viktorija Golubic":     "all_court",
    "Stefanie Voegele":      "baseliner",
    "Timea Bacsinszky":      "clay_specialist",
    "Belinda Bencic":        "all_court",   # dup
    "Xin-yu Wang":           "baseliner",
    "Shu-ai Peng":           "baseliner",
    "Qiang Wang":            "baseliner",
    "Zheng Qinwen":          "baseliner",
    "Xinyu Wang":            "baseliner",
    "Lin Zhu":               "baseliner",
    "Yafan Wang":            "clay_specialist",
    "Zhu Lin":               "baseliner",
    "Peng Shuai":            "baseliner",
    "Jie Zheng":             "baseliner",
    "Saisai Zheng":          "baseliner",
    "Lizette Cabrera":       "baseliner",
    "Storm Hunter":          "serve_volleyer",
    "Priscilla Hon":         "baseliner",
    "Daria Gavrilova":       "baseliner",
    "Ash Barty":             "all_court",
    "Sam Stosur":            "clay_specialist",
    "Casey Dellacqua":       "all_court",
    "Anastasia Grymalska":   "baseliner",
    "Anett Kontaveit":       "baseliner",
    "Kaia Kanepi":           "baseliner",
    "Mayar Sherif":          "clay_specialist",
    "Lina Gjorcheska":       "baseliner",
    "Kamilla Rakhimova":     "baseliner",
    "Tatiana Prozorova":     "baseliner",
    "Diana Shnaider":        "baseliner",
    "Mirra Andreeva":        "baseliner",   # dup
    "Erika Andreeva":        "baseliner",
    "Clara Burel":           "clay_specialist",
    "Elsa Jacquemot":        "clay_specialist",
    "Alice Caudan":          "clay_specialist",
    "Pauline Chirico":       "baseliner",
    "Chloé Paquet":          "clay_specialist",
    "Harmony Tan":           "all_court",
    "Magda Linette":         "all_court",
    "Katarzyna Piter":       "baseliner",
    "Urszula Radwanska":     "baseliner",
    "Agnieszka Radwanska":   "all_court",
    "Maria Kirilenko":       "all_court",
    "Alisa Kleybanova":      "baseliner",
    "Anna Chakvetadze":      "baseliner",
    "Anastasia Myskina":     "clay_specialist",
    "Nadia Petrova":         "baseliner",
    "Elena Dementieva":      "baseliner",
    "Flavia Pennetta":       "all_court",
    "Roberta Vinci":         "all_court",
    "Sara Errani":           "clay_specialist",
    "Camila Giorgi":         "serve_volleyer",
    "Elisabetta Cocciaretto":"clay_specialist",
    "Martina Trevisan":      "clay_specialist",
    "Lucia Bronzetti":       "clay_specialist",
    "Jasmine Paolini":       "all_court",   # dup
    "Lisa Pigato":           "clay_specialist",
    "Angelica Moratelli":    "clay_specialist",
    "Nuria Parrizas Diaz":   "clay_specialist",
    "Aliona Bolsova":        "clay_specialist",
    "Sara Sorribes Tormo":   "clay_specialist",
    "Cristina Bucsa":        "clay_specialist",
    "Rebeka Masarova":       "clay_specialist",
    "Maria Jose Martinez Sanchez": "all_court",
    "Carla Suarez Navarro":  "clay_specialist",
    "Arantxa Sanchez":       "clay_specialist",
    "Conchita Martinez":     "clay_specialist",
    "Virginia Ruzici":       "clay_specialist",
    "Billie Jean King":      "serve_volleyer",
    "Chris Evert":           "baseliner",
    "Evonne Goolagong":      "all_court",
    "Margaret Court":        "all_court",
    "Maria Bueno":           "serve_volleyer",
    "Hana Mandlikova":       "all_court",
    "Mats Wilander":         "clay_specialist",   # legend; cross-gender row OK
    "Ilkay Cengiz":          "baseliner",
    "Ipek Soylu":            "baseliner",
    "Cagla Buyukakcay":      "baseliner",
    "Maria Lourdes Carle":   "clay_specialist",
    "Julia Riera":           "clay_specialist",
    "Solana Sierra":         "clay_specialist",
    "Nadia Podoroska":       "clay_specialist",   # dup
    "Aoi Ito":               "baseliner",
    "Moyuka Uchijima":       "baseliner",
    "Ena Shibahara":         "baseliner",
    "Shuko Aoyama":          "serve_volleyer",
    "Misaki Doi":            "baseliner",
    "Kurumi Nara":           "baseliner",
    "Risa Ozaki":            "baseliner",
    "Kimiko Date":           "baseliner",
    "Akiko Morigami":        "baseliner",
    "Ayumi Morita":          "baseliner",
    "Sania Mirza":           "all_court",
    "Ankita Raina":          "clay_specialist",
    "Prarthana Thombare":    "baseliner",
    "Mihail Popescu":        "baseliner",
    "Renata Voracova":       "baseliner",
    "Klara Zakopalova":      "all_court",
    "Iveta Benesova":        "all_court",
    "Kveta Peschke":         "serve_volleyer",
    "Andrea Hlavackova":     "serve_volleyer",
    "Michaella Krajicek":    "serve_volleyer",
    "Arantxa Rus":           "baseliner",   # dup
    "Bibiane Schoofs":       "baseliner",
    "Cindy Burger":          "baseliner",
    "Maja Chwalinska":       "clay_specialist",
    "Weronika Falkowska":    "baseliner",
    "Julia Niemeier":        "baseliner",
    "Eva Lys":               "baseliner",
    "Nastasja Schunk":       "baseliner",
    "Ella Seidel":           "clay_specialist",
    "Tamara Korpatsch":      "clay_specialist",
    "Laura Siegemund":       "all_court",
    "Sabine Lisicki":        "serve_volleyer",
    "Andrea Petkovic":       "baseliner",
    "Anna-Lena Groenefeld":  "serve_volleyer",
    "Julia Goerges":         "serve_volleyer",
    "Martina Hingis":        "all_court",
    "Lindsay Davenport":     "baseliner",
    "Jennifer Capriati":     "baseliner",
    "Mary Joe Fernandez":    "baseliner",
    "Alycia Parks":          "serve_volleyer",   # dup
}

# Surface-advantage multiplier per style
STYLE_SURFACE: dict[str, dict[str, float]] = {
    "baseliner":       {"hard": 1.00, "clay": 1.05, "grass": 0.93, "carpet": 0.98, "indoor_hard": 1.01},
    "clay_specialist": {"hard": 0.93, "clay": 1.18, "grass": 0.82, "carpet": 0.90, "indoor_hard": 0.94},
    "serve_volleyer":  {"hard": 1.03, "clay": 0.88, "grass": 1.15, "carpet": 1.08, "indoor_hard": 1.04},
    "all_court":       {"hard": 1.01, "clay": 1.02, "grass": 1.02, "carpet": 1.00, "indoor_hard": 1.01},
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. 持久化資料管理
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlayerStats:
    name: str
    elo: dict[str, float] = field(default_factory=lambda: {s: float(INITIAL_ELO) for s in SURFACES})
    overall_elo: float = float(INITIAL_ELO)
    h2h: dict[str, list[int]] = field(default_factory=dict)     # opp -> [wins, losses]
    recent_results: list[int] = field(default_factory=list)     # 1=win 0=loss
    fatigue: float = 0.0
    last_match: str = ""                                         # ISO date
    titles: int = 0
    matches_played: int = 0

    def surface_elo(self, surface: str) -> float:
        return self.elo.get(surface, self.overall_elo)

    def form_score(self) -> float:
        recent = self.recent_results[-FORM_WINDOW:]
        return sum(recent) / len(recent) if recent else 0.5

    def fatigue_score(self) -> float:
        """0 = fresh, 1 = exhausted"""
        return min(self.fatigue, 1.0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "elo": self.elo,
            "overall_elo": self.overall_elo,
            "h2h": self.h2h,
            "recent_results": self.recent_results,
            "fatigue": self.fatigue,
            "last_match": self.last_match,
            "titles": self.titles,
            "matches_played": self.matches_played,
        }

    @staticmethod
    def from_dict(d: dict) -> "PlayerStats":
        ps = PlayerStats(name=d["name"])
        ps.elo = d.get("elo", {s: float(INITIAL_ELO) for s in SURFACES})
        ps.overall_elo = d.get("overall_elo", float(INITIAL_ELO))
        ps.h2h = d.get("h2h", {})
        ps.recent_results = d.get("recent_results", [])
        ps.fatigue = d.get("fatigue", 0.0)
        ps.last_match = d.get("last_match", "")
        ps.titles = d.get("titles", 0)
        ps.matches_played = d.get("matches_played", 0)
        return ps


class DataStore:
    def __init__(self, path: str = DATA_FILE):
        self.path = path
        self.players: dict[str, PlayerStats] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    raw = json.load(f)
                for name, d in raw.items():
                    self.players[name] = PlayerStats.from_dict(d)
                log.info("Loaded %d players from %s", len(self.players), self.path)
            except Exception as exc:
                log.warning("DataStore load error: %s", exc)

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({n: p.to_dict() for n, p in self.players.items()}, f, ensure_ascii=False, indent=2)

    def get(self, name: str) -> PlayerStats:
        if name not in self.players:
            self.players[name] = PlayerStats(name=name)
        return self.players[name]

    def record_match(
        self,
        winner: str,
        loser: str,
        surface: str,
        tournament_level: str = "250",
    ) -> None:
        w = self.get(winner)
        l = self.get(loser)
        today = datetime.now(timezone.utc).date().isoformat()

        # ── ELO update ───────────────────────────────────────────────────────
        k = _k_factor(tournament_level)
        # surface ELO
        ew = _elo_expected(w.surface_elo(surface), l.surface_elo(surface))
        el = 1.0 - ew
        w.elo[surface] = w.surface_elo(surface) + k * (1 - ew)
        l.elo[surface] = l.surface_elo(surface) + k * (0 - el)
        # overall ELO
        ew2 = _elo_expected(w.overall_elo, l.overall_elo)
        w.overall_elo += k * (1 - ew2)
        l.overall_elo += k * (0 - (1 - ew2))

        # ── H2H ──────────────────────────────────────────────────────────────
        if loser not in w.h2h:
            w.h2h[loser] = [0, 0]
        w.h2h[loser][0] += 1
        if winner not in l.h2h:
            l.h2h[winner] = [0, 0]
        l.h2h[winner][1] += 1

        # ── Form ─────────────────────────────────────────────────────────────
        w.recent_results.append(1)
        l.recent_results.append(0)
        w.recent_results = w.recent_results[-50:]
        l.recent_results = l.recent_results[-50:]

        # ── Fatigue ──────────────────────────────────────────────────────────
        w.fatigue = w.fatigue * FATIGUE_DECAY + 0.2
        l.fatigue = l.fatigue * FATIGUE_DECAY + 0.2

        # ── Misc ─────────────────────────────────────────────────────────────
        for p in (w, l):
            p.last_match = today
            p.matches_played += 1

        self.save()


_store = DataStore()


# ══════════════════════════════════════════════════════════════════════════════
# 3. ELO 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


def _k_factor(level: str) -> float:
    return {"gs": 64, "masters": 48, "500": 40, "250": 32, "challenger": 24}.get(
        level.lower(), K_FACTOR
    )


def _decay_elo(player: PlayerStats) -> None:
    """Shrink ELO toward 1500 if player hasn't played recently."""
    if not player.last_match:
        return
    last = datetime.fromisoformat(player.last_match)
    days_idle = (datetime.now(timezone.utc).replace(tzinfo=None) - last).days
    if days_idle > DECAY_DAYS:
        factor = 0.99 ** (days_idle - DECAY_DAYS)
        player.overall_elo = INITIAL_ELO + (player.overall_elo - INITIAL_ELO) * factor
        for s in SURFACES:
            player.elo[s] = INITIAL_ELO + (player.elo.get(s, INITIAL_ELO) - INITIAL_ELO) * factor


# ══════════════════════════════════════════════════════════════════════════════
# 4. 場地偵測
# ══════════════════════════════════════════════════════════════════════════════

_CLAY_KW = [
    "clay", "terre battue", "arcilla",
    "roland garros", "roland-garros", "french open",
    "monte carlo", "monte-carlo", "barcelona", "madrid", "rome",
    "hamburg", "bucharest", "estoril", "marrakech",
    "geneva", "lyon", "nice", "istanbul", "münchen",
    "bastad", "gstaad", "umag", "kitzbühel", "bosio",
    "cordoba", "buenos aires", "rio", "sao paulo", "santiago",
    "bogota", "houston", "moroni", "casablanca",
]
_GRASS_KW = [
    "grass", "gazon",
    "wimbledon", "queens club", "queen's club", "halle",
    "s-hertogenbosch", "hertogenbosch", "eastbourne",
    "birmingham", "nottingham", "devonshire park", "ilkley",
    "newport", "mallorca",
]
_CARPET_KW = ["carpet", "indoor carpet", "moquette"]
_INDOOR_HARD_KW = [
    "indoor hard", "indoor_hard",
    "paris masters", "paris bercy",
    "vienna", "basel", "rotterdam",
    "marseille", "montpellier", "metz",
    "moscow", "st. petersburg", "saint-petersburg",
    "stockholm", "sofia", "belgrade indoor",
    "atp finals", "wta finals", "nitto atp", "year-end",
]


def detect_surface(text: str) -> str:
    t = text.lower()
    if any(k in t for k in _CLAY_KW):
        return "clay"
    if any(k in t for k in _GRASS_KW):
        return "grass"
    if any(k in t for k in _CARPET_KW):
        return "carpet"
    if any(k in t for k in _INDOOR_HARD_KW):
        return "indoor_hard"
    return "hard"


# ══════════════════════════════════════════════════════════════════════════════
# 5. 賠率 API 整合
# ══════════════════════════════════════════════════════════════════════════════

def _get_odds(sport: str = SPORT_TENNIS_ATP) -> list[dict]:
    """Fetch upcoming matches + odds from The-Odds-API."""
    if not ODDS_KEY:
        return []
    url = f"{ODDS_BASE}/sports/{sport}/odds"
    params = {
        "apiKey": ODDS_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("Odds API error: %s", exc)
        return []


def _implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def _best_odds(bookmakers: list[dict], team: str) -> float | None:
    """Return best (highest) decimal odd for team across all bookmakers."""
    best: float | None = None
    for bm in bookmakers:
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            for outcome in mkt.get("outcomes", []):
                if outcome.get("name", "").lower() == team.lower():
                    v = outcome.get("price", 0.0)
                    if best is None or v > best:
                        best = v
    return best


# ══════════════════════════════════════════════════════════════════════════════
# 6. 核心預測引擎
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MatchPrediction:
    player1: str
    player2: str
    surface: str
    tour: str                      # "ATP" / "WTA"
    p1_win_prob: float             # 0-1
    confidence: float              # 0-100
    factors: dict[str, Any]
    odds_edge: float = 0.0         # model - implied (positive = value)
    recommended: bool = False

    def summary(self) -> str:
        winner = self.player1 if self.p1_win_prob >= 0.5 else self.player2
        win_pct = max(self.p1_win_prob, 1 - self.p1_win_prob) * 100
        edge_str = f"  Edge={self.odds_edge:+.1f}%" if self.odds_edge else ""
        rec_str = "  ✅ 推薦" if self.recommended else ""
        f = self.factors
        return (
            f"🎾 {self.player1} vs {self.player2}\n"
            f"   場地: {self.surface.upper()} | {self.tour}\n"
            f"   預測: {winner} 勝 ({win_pct:.1f}%)\n"
            f"   信心: {self.confidence:.1f}/100{edge_str}{rec_str}\n"
            f"   ├ ELO差: {f.get('elo_diff', 0):+.0f}"
            f"  H2H: {f.get('h2h_score', 0.5):.0%}"
            f"  疲勞Δ: {f.get('fatigue_delta', 0):+.2f}\n"
            f"   └ 近況: {f.get('form_score', 0.5):.0%}"
            f"  戰術: {f.get('style_mult', 1.0):.3f}"
            f"  賠率隱含: {f.get('implied_prob', 0):.0%}\n"
        )


def predict_match(
    p1_name: str,
    p2_name: str,
    surface: str = "hard",
    tour: str = "ATP",
    odds_data: list[dict] | None = None,
    tournament: str = "",
) -> MatchPrediction:
    surface = surface.lower()
    if surface not in SURFACES:
        surface = "hard"

    p1 = _store.get(p1_name)
    p2 = _store.get(p2_name)
    _decay_elo(p1)
    _decay_elo(p2)

    # ── Factor 1: Surface ELO ─────────────────────────────────────────────────
    p1_sElo = p1.surface_elo(surface)
    p2_sElo = p2.surface_elo(surface)
    elo_prob = _elo_expected(p1_sElo, p2_sElo)
    elo_diff = p1_sElo - p2_sElo

    # ── Factor 2: Overall ELO ────────────────────────────────────────────────
    overall_prob = _elo_expected(p1.overall_elo, p2.overall_elo)

    # ── Factor 3: H2H ────────────────────────────────────────────────────────
    h2h = p1.h2h.get(p2_name, [0, 0])
    total_h2h = h2h[0] + h2h[1]
    h2h_score = h2h[0] / total_h2h if total_h2h >= 3 else 0.5

    # ── Factor 4: Form ───────────────────────────────────────────────────────
    form_delta = p1.form_score() - p2.form_score()   # -1 to +1
    form_prob = 0.5 + 0.15 * form_delta

    # ── Factor 5: Fatigue ────────────────────────────────────────────────────
    fatigue_delta = p2.fatigue_score() - p1.fatigue_score()  # positive → p1 advantage
    fatigue_adj = 0.5 + 0.10 * fatigue_delta

    # ── Factor 6: Playing-style × Surface ────────────────────────────────────
    style1 = PLAYSTYLE.get(p1_name, "baseliner")
    style2 = PLAYSTYLE.get(p2_name, "baseliner")
    surf_map = STYLE_SURFACE
    s1_mult = surf_map[style1].get(surface, 1.0)
    s2_mult = surf_map[style2].get(surface, 1.0)
    style_mult = s1_mult / s2_mult  # >1 favours p1
    style_prob = style_mult / (1.0 + style_mult)

    # ── Weighted blend ───────────────────────────────────────────────────────
    raw_prob = (
        0.30 * elo_prob
        + 0.15 * overall_prob
        + 0.15 * h2h_score
        + 0.15 * form_prob
        + 0.10 * fatigue_adj
        + 0.15 * style_prob
    )
    raw_prob = max(0.05, min(0.95, raw_prob))

    # ── Confidence (sharpness around 50 %) ───────────────────────────────────
    confidence = 50.0 + abs(raw_prob - 0.5) * 100

    # ── Odds edge ────────────────────────────────────────────────────────────
    odds_edge = 0.0
    implied_p = 0.0
    if odds_data:
        match_obj = _find_odds_match(p1_name, p2_name, odds_data)
        if match_obj:
            best = _best_odds(match_obj.get("bookmakers", []), p1_name)
            if best:
                implied_p = _implied_prob(best)
                odds_edge = (raw_prob - implied_p) * 100

    recommended = (confidence >= MIN_CONFIDENCE) and (
        not odds_data or abs(odds_edge) >= EDGE_THRESHOLD
    )

    return MatchPrediction(
        player1=p1_name,
        player2=p2_name,
        surface=surface,
        tour=tour,
        p1_win_prob=raw_prob,
        confidence=confidence,
        factors={
            "elo_diff": elo_diff,
            "h2h_score": h2h_score,
            "fatigue_delta": fatigue_delta,
            "form_score": p1.form_score(),
            "style_mult": style_mult,
            "implied_prob": implied_p,
        },
        odds_edge=odds_edge,
        recommended=recommended,
    )


def _find_odds_match(
    p1: str, p2: str, odds_data: list[dict]
) -> dict | None:
    p1l, p2l = p1.lower(), p2.lower()
    for m in odds_data:
        home = m.get("home_team", "").lower()
        away = m.get("away_team", "").lower()
        if (p1l in home or home in p1l) and (p2l in away or away in p2l):
            return m
        if (p2l in home or home in p2l) and (p1l in away or away in p1l):
            return m
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 7. 今日推薦主流程
# ══════════════════════════════════════════════════════════════════════════════

def _parse_matchup(text: str) -> tuple[str, str] | None:
    """Try to extract 'Player A vs Player B' from free text."""
    m = re.search(
        r"([A-Z][a-zA-Z\-'. ]{2,30})\s+(?:vs\.?|v\.?|against)\s+([A-Z][a-zA-Z\-'. ]{2,30})",
        text,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def today_picks(tour: str = "both") -> list[MatchPrediction]:
    picks: list[MatchPrediction] = []
    sports = []
    if tour.lower() in ("atp", "both"):
        sports.append((SPORT_TENNIS_ATP, "ATP"))
    if tour.lower() in ("wta", "both"):
        sports.append((SPORT_TENNIS_WTA, "WTA"))

    for sport_key, tour_label in sports:
        odds_data = _get_odds(sport_key)
        for match in odds_data:
            p1 = match.get("home_team", "")
            p2 = match.get("away_team", "")
            if not p1 or not p2:
                continue
            # Detect surface from tournament name / metadata
            tournament = match.get("sport_title", "") + " " + match.get("id", "")
            surface = detect_surface(tournament)
            pred = predict_match(p1, p2, surface, tour_label, odds_data, tournament)
            if pred.recommended:
                picks.append(pred)

    # Sort by confidence desc
    picks.sort(key=lambda x: x.confidence, reverse=True)
    return picks


# ══════════════════════════════════════════════════════════════════════════════
# 8. Telegram 指令處理器
# ══════════════════════════════════════════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    txt = (
        "🎾 *Tennis Bot v3.2* — ATP/WTA 預測系統\n\n"
        "指令列表：\n"
        "/picks — 今日推薦 (ATP+WTA)\n"
        "/atp — 僅 ATP 推薦\n"
        "/wta — 僅 WTA 推薦\n"
        "/predict A vs B [surface] — 單場預測\n"
        "/result W d L [surface] — 記錄結果\n"
        "/stats 球員名 — 查詢球員數據\n"
        "/h2h A vs B — 頭對頭紀錄\n"
        "/help — 幫助"
    )
    await update.message.reply_text(txt, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_picks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⏳ 抓取賠率並運算中…")
    picks = today_picks("both")
    if not picks:
        await update.message.reply_text("今日暫無高信心推薦，或 API 無數據。")
        return
    for pred in picks[:10]:
        await update.message.reply_text(pred.summary(), parse_mode="Markdown")
    log.info("=== Done — %d picks ===", len(picks))


async def cmd_atp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⏳ 抓取 ATP 賠率…")
    picks = today_picks("ATP")
    if not picks:
        await update.message.reply_text("今日 ATP 暫無推薦。")
        return
    for pred in picks[:8]:
        await update.message.reply_text(pred.summary(), parse_mode="Markdown")


async def cmd_wta(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("⏳ 抓取 WTA 賠率…")
    picks = today_picks("WTA")
    if not picks:
        await update.message.reply_text("今日 WTA 暫無推薦。")
        return
    for pred in picks[:8]:
        await update.message.reply_text(pred.summary(), parse_mode="Markdown")


async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Usage: /predict Carlos Alcaraz vs Jannik Sinner clay
    """
    raw = " ".join(context.args or [])
    # Try to extract surface hint at the end
    surface = "hard"
    for s in SURFACES:
        if raw.lower().endswith(s):
            surface = s
            raw = raw[: -len(s)].strip()
            break
    # Also check clay keywords in full text
    if detect_surface(raw) != "hard":
        surface = detect_surface(raw)

    pair = _parse_matchup(raw)
    if not pair:
        await update.message.reply_text(
            "格式錯誤。請使用：/predict 球員A vs 球員B [hard|clay|grass|carpet|indoor_hard]"
        )
        return
    p1, p2 = pair
    pred = predict_match(p1, p2, surface, tour="ATP")
    await update.message.reply_text(pred.summary(), parse_mode="Markdown")


async def cmd_result(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Usage: /result Carlos Alcaraz d Jannik Sinner clay 500
    """
    args = context.args or []
    raw = " ".join(args)
    # Extract surface
    surface = "hard"
    for s in SURFACES:
        if s in raw.lower():
            surface = s
            raw = re.sub(rf"\b{s}\b", "", raw, flags=re.IGNORECASE).strip()
            break
    # Extract tournament level
    level = "250"
    for lv in ("gs", "masters", "500", "250", "challenger"):
        if lv in raw.lower():
            level = lv
            raw = re.sub(rf"\b{lv}\b", "", raw, flags=re.IGNORECASE).strip()
            break
    # Split on 'd' or 'def'
    m = re.split(r"\s+d(?:ef)?\s+", raw, flags=re.IGNORECASE)
    if len(m) != 2:
        await update.message.reply_text(
            "格式錯誤。請使用：/result 勝者 d 敗者 [surface] [level]"
        )
        return
    winner, loser = m[0].strip(), m[1].strip()
    _store.record_match(winner, loser, surface, level)
    await update.message.reply_text(
        f"✅ 已記錄：{winner} def. {loser} ({surface.upper()}, {level.upper()})"
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    name = " ".join(context.args or "").strip()
    if not name:
        await update.message.reply_text("請提供球員名稱。例：/stats Carlos Alcaraz")
        return
    p = _store.get(name)
    _decay_elo(p)
    lines = [
        f"📊 *{p.name}*",
        f"整體 ELO: {p.overall_elo:.0f}",
    ]
    for s in SURFACES:
        lines.append(f"  {s}: {p.elo.get(s, INITIAL_ELO):.0f}")
    lines += [
        f"近期勝率: {p.form_score():.0%} (近{min(len(p.recent_results),FORM_WINDOW)}場)",
        f"疲勞值: {p.fatigue_score():.2f}",
        f"總比賽數: {p.matches_played}",
        f"最後比賽: {p.last_match or '無紀錄'}",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_h2h(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    raw = " ".join(context.args or [])
    pair = _parse_matchup(raw)
    if not pair:
        await update.message.reply_text("格式錯誤。例：/h2h Djokovic vs Federer")
        return
    p1n, p2n = pair
    p1 = _store.get(p1n)
    h2h = p1.h2h.get(p2n, [0, 0])
    await update.message.reply_text(
        f"🎾 H2H: {p1n} vs {p2n}\n"
        f"  {p1n}: {h2h[0]} 勝\n"
        f"  {p2n}: {h2h[1]} 勝\n"
        f"  總計: {sum(h2h)} 場"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fallback: try to parse free-text matchup."""
    text = update.message.text or ""
    pair = _parse_matchup(text)
    if pair:
        surface = detect_surface(text)
        pred = predict_match(pair[0], pair[1], surface)
        await update.message.reply_text(pred.summary(), parse_mode="Markdown")
    else:
        await update.message.reply_text(
            "未識別指令。輸入 /help 查看指令列表，或直接輸入 '球員A vs 球員B' 進行預測。"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 9. 排程：每日自動推送
# ══════════════════════════════════════════════════════════════════════════════

async def scheduled_daily_picks(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Push today's picks to all registered chats (stored in bot_data)."""
    chat_ids: set[int] = context.bot_data.get("subscribed_chats", set())
    if not chat_ids:
        return
    picks = today_picks("both")
    if not picks:
        return
    for chat_id in chat_ids:
        try:
            for pred in picks[:5]:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=pred.summary(),
                    parse_mode="Markdown",
                )
        except Exception as exc:
            log.warning("Scheduled push failed for %d: %s", chat_id, exc)


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cid = update.effective_chat.id
    chats: set[int] = context.bot_data.setdefault("subscribed_chats", set())
    chats.add(cid)
    await update.message.reply_text("✅ 已訂閱每日推薦 (UTC 06:00)")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cid = update.effective_chat.id
    chats: set[int] = context.bot_data.get("subscribed_chats", set())
    chats.discard(cid)
    await update.message.reply_text("❌ 已取消訂閱")


# ══════════════════════════════════════════════════════════════════════════════
# 10. 資格取消 / 棄賽 特殊處理
# ══════════════════════════════════════════════════════════════════════════════

def _dq_score(winner: str, loser: str, surface: str, reason: str = "walkover") -> None:
    """
    Record a walkover / retirement / disqualification.
    The winner gets a partial ELO gain; loser gets a partial penalty.
    ELO delta is scaled by 0.4 (incomplete match signal).
    """
    w = _store.get(winner)
    l = _store.get(loser)
    k = K_FACTOR * 0.4  # reduced K for incomplete matches
    ew = _elo_expected(w.surface_elo(surface), l.surface_elo(surface))
    w.elo[surface] = w.surface_elo(surface) + k * (1 - ew)
    l.elo[surface] = l.surface_elo(surface) + k * (0 - (1 - ew))
    ew2 = _elo_expected(w.overall_elo, l.overall_elo)
    w.overall_elo += k * (1 - ew2)
    l.overall_elo += k * (0 - (1 - ew2))
    # H2H not updated for walkovers
    w.recent_results.append(1)
    l.recent_results.append(0)
    w.recent_results = w.recent_results[-50:]
    l.recent_results = l.recent_results[-50:]
    today = datetime.now(timezone.utc).date().isoformat()
    for p in (w, l):
        p.last_match = today
        p.matches_played += 1
    _store.save()
    log.info("DQ/walkover recorded: %s over %s (%s) — %s", winner, loser, surface, reason)


async def cmd_walkover(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Usage: /walkover Winner d Loser [surface] [reason]
    Example: /walkover Medvedev d Rublev clay injury
    """
    args = context.args or []
    raw = " ".join(args)
    surface = "hard"
    for s in SURFACES:
        if s in raw.lower():
            surface = s
            raw = re.sub(rf"\b{s}\b", "", raw, flags=re.IGNORECASE).strip()
            break
    reason = "walkover"
    for r in ("injury", "illness", "disqualification", "dq", "retirement", "ret", "walkover", "wo"):
        if r in raw.lower():
            reason = r
            raw = re.sub(rf"\b{r}\b", "", raw, flags=re.IGNORECASE).strip()
            break
    m = re.split(r"\s+d(?:ef)?\s+", raw, flags=re.IGNORECASE)
    if len(m) != 2:
        await update.message.reply_text(
            "格式：/walkover 勝者 d 敗者 [surface] [reason]"
        )
        return
    winner, loser = m[0].strip(), m[1].strip()
    _dq_score(winner, loser, surface, reason)
    await update.message.reply_text(
        f"⚠️ 已記錄 {reason}: {winner} over {loser} ({surface.upper()})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 11. 應用程式入口
# ══════════════════════════════════════════════════════════════════════════════

def build_app() -> Application:
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("help",        cmd_help))
    app.add_handler(CommandHandler("picks",       cmd_picks))
    app.add_handler(CommandHandler("atp",         cmd_atp))
    app.add_handler(CommandHandler("wta",         cmd_wta))
    app.add_handler(CommandHandler("predict",     cmd_predict))
    app.add_handler(CommandHandler("result",      cmd_result))
    app.add_handler(CommandHandler("stats",       cmd_stats))
    app.add_handler(CommandHandler("h2h",         cmd_h2h))
    app.add_handler(CommandHandler("subscribe",   cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("walkover",    cmd_walkover))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    # Daily picks at 06:00 UTC
    job_queue = app.job_queue
    if job_queue:
        job_queue.run_daily(
            scheduled_daily_picks,
            time=datetime.strptime("06:00", "%H:%M").replace(tzinfo=timezone.utc).timetz(),
        )

    return app


def run() -> None:
    log.info("Starting Tennis Bot v3.2")
    app = build_app()
    app.run_polling(drop_pending_updates=True)
    log.info("=== Done — %d picks ===", len(picks))


if __name__ == "__main__":
    run()
