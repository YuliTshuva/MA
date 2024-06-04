"""
Yuli Tshuva
Find best model (by the neg-pos ration) for each year"""

import os
from os.path import join

import numpy as np
import pandas as pd
import json
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm.auto import tqdm

rcParams['font.size'] = 14
rcParams['font.family'] = 'Times New Roman'


RESULTS_FOLDER = "results"


def get_roc_curve(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return fpr, tpr


best_tpr = [0] * 9
best_model = [""] * 9

for results_folder in tqdm(os.listdir()):
    if not "results" in results_folder:
        continue
    if not os.path.isdir(results_folder):
        continue
    if results_folder == "results":
        continue

    ratio = results_folder.split("_")[-1]

    roc_path = lambda year: f'roc_details_{year}.pkl'

    for i, year in enumerate(range(2011, 2020)):
        with open(join(results_folder, roc_path(year)), 'rb') as f:
            dct = pickle.load(f)

        labels, scores = dct['labels'], dct['scores']

        fpr, tpr = get_roc_curve(labels, scores)

        narrowed_tpr = [tpr[i] for i in range(len(fpr)) if fpr[i] <= 0.1][-1]

        if narrowed_tpr > best_tpr[i]:
            best_tpr[i] = narrowed_tpr
            best_model[i] = ratio

print("Best tprs:", best_tpr)
print("Best models:", best_model)
