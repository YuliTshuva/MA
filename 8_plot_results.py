"""
Yuli Tshuva
Plot all results received
"""

import os
from os.path import join
import pandas as pd
import json
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.size'] = 14
rcParams['font.family'] = 'Times New Roman'


RESULTS_FOLDER = "results"


def plot_roc_curve(y_test, y_score, ax, year):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f'AUC: {roc_auc:.3f}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{year}')
    ax.legend(loc="lower right")
    return


for results_folder in os.listdir():
    if not "results" in results_folder:
        continue
    if not os.path.isdir(results_folder):
        continue
    if results_folder == "results":
        continue

    ratio = results_folder.split("_")[-1]

    results_lst = []

    # Load the results
    year = 2011
    for file in os.listdir(results_folder):
        if file.endswith('.json'):
            with open(join(results_folder, file), 'r') as f:
                results = json.load(f)
                results_lst.append(pd.DataFrame(results, index=[year]))
                year += 1

    # Concatenate the results
    results_df = pd.concat(results_lst)
    results_df.index = range(2011, 2020)

    # Save the results
    results_df.to_csv(join(RESULTS_FOLDER, f"results_{ratio}.csv"))

    model_path = lambda year: f"model_for_{year}.pkl"

    # Plot feature importance
    fig, axes = plt.subplots(3, 3, figsize=(25, 35))
    # Set axes track
    i, j = 0, 0

    # Set super title
    fig.suptitle("Feature Importance", fontsize=20, y=1.0)

    # Iterate over the years
    for year in range(2011, 2020):
        # Load the model
        with open(join(results_folder, model_path(year)), 'rb') as f:
            model = pickle.load(f)
        # Plot FI
        xgb.plot_importance(model, ax=axes[i, j], title=f"{year}")

        # Fix the index
        j += 1
        if j == 3:
            i += 1
            j = 0

    plt.tight_layout()
    plt.savefig(join(RESULTS_FOLDER, f"feature_importance_{ratio}.pdf"))

    roc_path = lambda year: f'roc_details_{year}.pkl'

    fig, ax = plt.subplots(3, 3, figsize=(15, 16))

    fig.suptitle("ROC Curve", y=0.99, fontsize=22)

    i, j = 0, 0

    for year in range(2011, 2020):
        with open(join(results_folder, roc_path(year)), 'rb') as f:
            dct = pickle.load(f)

        labels, scores = dct['labels'], dct['scores']

        plot_roc_curve(labels, scores, ax[i, j], year)

        j += 1
        if j == 3:
            j = 0
            i += 1

    plt.tight_layout()
    plt.savefig(join(RESULTS_FOLDER, f'roc_curve_{ratio}.png'))
