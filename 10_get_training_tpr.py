"""
Yuli Tshuva
Set up the code of XGBoost to run on the servers.
"""

import json

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import optuna
import warnings
import threading
import os
import gc
import pickle
from os.path import join

# Filter a lot of unnecessary warnings
warnings.filterwarnings("ignore")
# Ignore optuna
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
# Set weight decay for the weighted directors
ALPHA = 0.7
CROSS_VAL = 5
TRIALS = 1000

# The candidates of year XXXX is every company that submitted a 10-k report in year XXXX or XXXX-1
with open("find_candidates_per_year/candidates.json", "r") as f:
    CANDIDATES = json.load(f)  ### dictionary keys are strings
    # str(year) -> [str(cik1), str(cik2), ...]

# Constants
# The M&A data organized by index_col, head, tail, year
MA_PATH = "analyze_data/M&A_processed.csv"
# The directors data organized by index_col, director, cik, year
DIRECTORS_DF_PATH = "analyze_data/directors_parsed.csv"
# The features table organized by index_col, some features, year, cik
FEATURES_TABLE = "literals_mining/merged_data.csv"
# The topological features of the directors graph
TOPO_FEATURES = "topo_features/network_features_for_year.csv"
# The connection components of the directors graph
COMPONENTS = lambda year: join("connection_components", f"connections_components_{year}.csv")


### Helper functions ###

def create_false_samples(df, companies, factor=10):
    """I believe the name of the function speaks for itself."""
    # Set number of false samples
    FALSE_SAMPLES = int(df.shape[0] * factor)

    # Find the iloc to continue from
    last_index = df.shape[0]

    # Reset the index
    df.reset_index(inplace=True, drop=True)

    for i in range(FALSE_SAMPLES + 1):
        # Choose two random companies
        head = random.choice(companies)
        tail = random.choice(companies)
        # Check that they are not in the original df
        if ((df['head'] == head) & (df['tail'] == tail)).any():
            continue
        # Check that they are not the same company
        if head == tail:
            continue
        # Add to the dataframe as negative sample
        df.loc[last_index + i] = [head, tail, 0]

    # shuffle the DataFrame rows
    df = df.sample(frac=1)

    # Reset the df index
    df.reset_index(inplace=True, drop=True)

    return df


def find_mutual_directors_improved(directors_df: pd.DataFrame, candidates: list, max_year: int):
    """
    Let's take a different attitude to this problem.
    Last time I do it naively, and it took a lot of time O(n^3).
    Now, let's try to do it more efficiently with more space consuming (Not necessary actually).
    """
    # Create a dictionary of a company cik to its index in the candidates list - for major time complexity improvement
    cik_to_idx = {candidates[i]: i for i in range(len(candidates))}

    # Set a numpy array of zeros for the mutual and weighted mutual directors
    mutual_directors, weighted_mutual_directors = (np.zeros((len(candidates), len(candidates))),
                                                   np.zeros((len(candidates), len(candidates))))

    # Filter the directors_df to the relevant year and take only the candidates
    directors_df = directors_df[directors_df["year"] <= max_year]
    directors_df = directors_df[directors_df["cik"].isin(candidates)]

    temp_df = directors_df[directors_df["year"] == max_year]

    # Create a dictionary of director's name to its list of ciks per year
    director_to_ciks = {}
    for index, row in temp_df.iterrows():
        if row["director"] not in director_to_ciks:
            director_to_ciks[row["director"]] = [row["cik"]]
        else:
            director_to_ciks[row["director"]].append(row["cik"])

    for lst in director_to_ciks.values():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                mutual_directors[cik_to_idx[lst[i]], cik_to_idx[lst[j]]] += 1
                mutual_directors[cik_to_idx[lst[j]], cik_to_idx[lst[i]]] += 1
                weighted_mutual_directors[cik_to_idx[lst[i]], cik_to_idx[lst[j]]] += 1
                weighted_mutual_directors[cik_to_idx[lst[j]], cik_to_idx[lst[i]]] += 1

    del temp_df, director_to_ciks

    # Going 9 years back for the weighted mutual directors
    for num, year in enumerate(range(max_year - 1, max_year - 10, -1)):
        temp_df = directors_df[directors_df["year"] == year]
        # Create a dictionary of director's name to its list of ciks per year
        director_to_ciks = {}
        for index, row in temp_df.iterrows():
            if row["director"] not in director_to_ciks:
                director_to_ciks[row["director"]] = [row["cik"]]
            else:
                director_to_ciks[row["director"]].append(row["cik"])

        for lst in director_to_ciks.values():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    weighted_mutual_directors[cik_to_idx[lst[i]], cik_to_idx[lst[j]]] += ALPHA ** (num + 1)
                    weighted_mutual_directors[cik_to_idx[lst[j]], cik_to_idx[lst[i]]] += ALPHA ** (num + 1)

        del temp_df, director_to_ciks

    gc.collect()

    return mutual_directors, weighted_mutual_directors, cik_to_idx


def get_literals(literals_df: pd.DataFrame, cik: str, year: int):
    """
    Get the literals dataframe and cik and find the literals of the cik (for relevant year).
    Gets as a very long dictionary
    """
    # Filter data out of the literals_df
    results_df = literals_df[(literals_df["cik"] == str(cik)) & (literals_df["year"] == np.int64(year))]
    results_df.drop(["cik", "year"], axis=1, inplace=True)
    # If no data was found
    if results_df.shape[0] == 0:
        return pd.DataFrame([{col: np.nan for col in literals_df.columns if col not in ["cik", "year"]}])
    # If duplication was detected
    if results_df.shape[0] != 1:
        raise Exception("Line 152: Unexpected shape of DataFrame in get_literals function:", results_df.shape)
    return results_df


def get_topological_features(topo_df: pd.DataFrame, cik: str):
    """
    Get year and extract the topological features to the year (of the directors graph).
    Return a dataframe of all sort of features.
    """
    try:
        results_df = pd.DataFrame(topo_df.loc[int(cik)]).T
    except:
        results_df = pd.DataFrame([{col: np.nan for col in topo_df.columns}])
    if results_df.shape[0] != 1:
        raise Exception("Line 166: Unexpected shape of DataFrame in get_topological_features function:",
                        results_df.shape)
    return results_df


### MOST HEAVY FUNCTION - Build Function

def process_df(df: pd.DataFrame, data_df: pd.DataFrame, mutual_directors: dict, weighted_mutual_directors: dict,
               cik_to_idx: dict, connection_comp_df: pd.DataFrame, save=False):
    """
    Process the dataframe - df, so it will be ready to be inputted to the model.
    df: The train/test dataframe you want to process
    data_df: The pre-calculated dataframe of all charastersitics known
    """

    # Get the heads and tails in lists
    df["head"], df["tail"] = df["head"].astype(str), df["tail"].astype(str)
    heads, tails = list(df["head"]), list(df["tail"])

    # Let's try to restore the functionality more efficiently
    heads_df, tails_df = data_df.loc[heads], data_df.loc[tails]
    heads_df.columns, tails_df.columns = [f"head_{col}" for col in heads_df.columns], [f"tail_{col}" for col in
                                                                                       tails_df.columns]
    # Reset the index of the dfs
    heads_df.reset_index(inplace=True, drop=True)
    tails_df.reset_index(inplace=True, drop=True)

    mutual_directors_lst = [mutual_directors[cik_to_idx[heads[i]], cik_to_idx[tails[i]]] for i in range(len(heads))]
    weighted_mutual_directors_lst = [weighted_mutual_directors[cik_to_idx[heads[i]], cik_to_idx[tails[i]]]
                                     for i in range(len(heads))]

    # Add the connection components size
    con_comp_heads, con_comp_tails = [], []
    for head, tail in zip(heads, tails):
        head, tail = int(head), int(tail)
        if head in connection_comp_df.index:
            con_comp_heads.append(np.log(connection_comp_df.loc[head]["size"]))
        else:
            con_comp_heads.append(0)
        if tail in connection_comp_df.index:
            con_comp_tails.append(np.log(connection_comp_df.loc[tail]["size"]))
        else:
            con_comp_tails.append(0)

    # Clear space in memory by deleting the old dfs
    del df

    # Make a dataframe out of my data
    df = pd.concat([heads_df, tails_df], axis=1)
    df["mutual_directors"] = mutual_directors_lst
    df["weighted_mutual_directors"] = weighted_mutual_directors_lst
    df["log_size_connection_component_head"] = con_comp_heads
    df["log_size_connection_component_tail"] = con_comp_tails

    # Define all industries
    industries = [col.split("_")[-1] for col in df.columns if "industry" in col]

    # Define a column of mutual industry
    df["mutual_industry"] = 0

    # Iterate over all industries
    for industry in industries:
        df[f"tail_industry_{industry}"] = df[f"tail_industry_{industry}"].fillna(0)
        df[f"head_industry_{industry}"] = df[f"head_industry_{industry}"].fillna(0)
        df["mutual_industry"] += df[f"head_industry_{industry}"] * df[f"tail_industry_{industry}"]

    # Fix error
    df["mutual_industry"] = df["mutual_industry"].apply(lambda x: 1 if x > 0 else 0)

    del heads, tails, heads_df, tails_df, mutual_directors_lst, weighted_mutual_directors_lst

    gc.collect()

    return df


### Mass functions


def get_data_prepared_ahead(candidates: list, directors_df: pd.DataFrame, literals_df: pd.DataFrame, max_year: int):
    """
    Prepare all the data ahead of time to save time complexity.
    Basically everything I'll need for process_df function.
    """
    # Open graph topological features for the most relevant year
    topological_features_df = pd.read_csv(TOPO_FEATURES.replace("year", str(max_year)), index_col=0)

    # Open the connection components for the most relevant year
    connection_components_df = pd.read_csv(COMPONENTS(max_year), index_col=0)
    connection_components_df["size"] = connection_components_df["size"].astype(int)
    connection_components_df["component"] = connection_components_df["component"].astype(int)

    # Set empty lists to collect data
    literals, topological_features = [], []
    for candidate in candidates:
        literals.append(get_literals(literals_df, candidate, max_year))
        topological_features.append(get_topological_features(topological_features_df, candidate))

    # concat all collected data into one df and rename the columns
    candidates_literals_df = pd.concat(literals, axis=0, ignore_index=True)
    candidates_topological_features_df = pd.concat(topological_features, axis=0, ignore_index=True)

    df = pd.concat([candidates_literals_df, candidates_topological_features_df], axis=1)
    df.reset_index(inplace=True, drop=True)
    df.index = candidates

    del topological_features_df, literals, topological_features, candidates_literals_df, (
        candidates_topological_features_df)

    gc.collect()

    # Find all mutual and weighted mutual directors between a ll the possible pairs of candidates
    mutual_directors, weighted_mutual_directors, director_to_ciks = (
        find_mutual_directors_improved(directors_df, candidates, max_year))

    return df, mutual_directors, weighted_mutual_directors, director_to_ciks, connection_components_df


def dynamic_building_of_data(year, ma_df, candidates, data_df, mutual_directors,
                             weighted_mutual_directors, cik_to_idx, connection_comp_df, ratio):
    # Get a copy of M&A data to edit
    df = ma_df.copy()
    # Add positive label identically
    df["label"] = 1

    # Split into train and test
    train_data = df[df["year"] <= year]
    test_data = df[df["year"] == year + 1]

    # Drop the year column
    train_data.drop("year", axis=1, inplace=True)
    test_data.drop("year", axis=1, inplace=True)

    # Clean test data
    test_data = test_data[test_data["head"].isin(candidates) & test_data["tail"].isin(candidates)]

    # Add negative samples
    train_data = create_false_samples(train_data, candidates, factor=ratio)
    test_data = create_false_samples(test_data, candidates, factor=ratio)

    # Shuffle training data
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    # Save one example of the processed df for Yoram
    save = int(year) == 2016

    X_train = process_df(train_data, data_df, mutual_directors, weighted_mutual_directors,
                         cik_to_idx, connection_comp_df, save=save)
    X_test = process_df(test_data, data_df, mutual_directors, weighted_mutual_directors, cik_to_idx,
                        connection_comp_df)
    y_train, y_test = list(train_data["label"]), list(test_data["label"])

    test_data_idx = list(test_data.index)
    test_content = [[str(test_data.loc[idx]["head"]), str(test_data.loc[idx]["tail"])] for idx in test_data_idx if
                    int(test_data.loc[idx]["label"]) == 1]

    return X_train, X_test, y_train, y_test, test_content


def training_and_tuning_model_on_data(X_train, X_test, y_train, y_test, year, results_dir, trails_num=1000):
    model_save_path = join(results_dir, f"model_for_{year}.pkl")

    def objective(trial):
        """Define the objective function"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        }

        # Set an auc list
        auc_lst = []

        for i in range(CROSS_VAL):
            # Split the data
            X_train_opt, X_valid_opt, y_train_opt, y_valid_opt = train_test_split(X_train, y_train,
                                                                                  test_size=0.25)

            # Fit the model
            optuna_model = XGBClassifier(**params)
            optuna_model.fit(X_train_opt, y_train_opt)

            # Make predictions
            y_pred = optuna_model.predict_proba(X_valid_opt)[:, 1]

            # Evaluate predictions
            auc_lst.append(roc_auc_score(y_valid_opt, y_pred))

        return np.mean(auc_lst)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trails_num)
    trial = study.best_trial
    params = trial.params
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    with open(model_save_path, "wb") as file:
        pickle.dump(model, file)

    return model


def hits_calculations(model, test_content, data_df, mutual_directors, weighted_mutual_directors, candidates, year,
                      cik_to_idx, connection_comp_df):
    xgb_hits_1_tail = 0
    xgb_hits_3_tail = 0
    xgb_hits_5_tail = 0
    xgb_hits_10_tail = 0

    xgb_hits_1_head = 0
    xgb_hits_3_head = 0
    xgb_hits_5_head = 0
    xgb_hits_10_head = 0

    mr_head = 0
    mr_tail = 0

    mrr_head = 0
    mrr_tail = 0

    count = 0

    scores, labels = [], []

    for tup in test_content:
        count += 1
        if count % 50 == 0:
            print(f"Year {year} - Predicted {count} out of {len(test_content)} samples.")
        tail = tup[1]
        head = tup[0]

        ### HEAD PREDICTION ###
        # Create a temporary df for evaluation
        new_candidates = [c for c in candidates if c != tail]
        temp_df = pd.DataFrame({"head": new_candidates, "tail": [tail for _ in range(len(new_candidates))]})
        temp_df = process_df(temp_df, data_df, mutual_directors,
                             weighted_mutual_directors, cik_to_idx, connection_comp_df)

        xgb_head_scores = list(np.array(model.predict_proba(temp_df))[:, 1])
        scores += xgb_head_scores
        labels += [1 if c == head else 0 for c in new_candidates]
        xgb_top_heads = [new_candidates[i] for i in list(np.argsort(xgb_head_scores)[::-1])]

        xgb_hits_1_head += 1 if head in xgb_top_heads[:1] else 0
        xgb_hits_3_head += 1 if head in xgb_top_heads[:3] else 0
        xgb_hits_5_head += 1 if head in xgb_top_heads[:5] else 0
        xgb_hits_10_head += 1 if head in xgb_top_heads[:10] else 0
        rank = xgb_top_heads.index(head) + 1
        mr_head += rank
        mrr_head += 1 / rank

        ### TAIL PREDICTION ###
        # Create a temporary df for evaluation
        new_candidates = [c for c in candidates if c != head]
        temp_df = pd.DataFrame({"head": [head for _ in range(len(new_candidates))], "tail": new_candidates})
        temp_df = process_df(temp_df, data_df, mutual_directors,
                             weighted_mutual_directors, cik_to_idx, connection_comp_df)

        xgb_tail_scores = list(np.array(model.predict_proba(temp_df))[:, 1])
        scores += xgb_tail_scores
        labels += [1 if c == tail else 0 for c in new_candidates]
        xgb_top_tails = [new_candidates[i] for i in list(np.argsort(xgb_tail_scores)[::-1])]

        xgb_hits_1_tail += 1 if tail in xgb_top_tails[:1] else 0
        xgb_hits_3_tail += 1 if tail in xgb_top_tails[:3] else 0
        xgb_hits_5_tail += 1 if tail in xgb_top_tails[:5] else 0
        xgb_hits_10_tail += 1 if tail in xgb_top_tails[:10] else 0
        rank = xgb_top_tails.index(tail) + 1
        mr_tail += rank
        mrr_tail += 1 / rank

    xgb_hits_1_tail /= len(test_content)
    xgb_hits_3_tail /= len(test_content)
    xgb_hits_5_tail /= len(test_content)
    xgb_hits_10_tail /= len(test_content)
    mr_head /= len(test_content)
    mrr_head /= len(test_content)

    xgb_hits_1_head /= len(test_content)
    xgb_hits_3_head /= len(test_content)
    xgb_hits_5_head /= len(test_content)
    xgb_hits_10_head /= len(test_content)
    mr_tail /= len(test_content)
    mrr_tail /= len(test_content)

    hits = {"hits_1_tail": xgb_hits_1_tail,
            "hits_3_tail": xgb_hits_3_tail,
            "hits_5_tail": xgb_hits_5_tail,
            "hits_10_tail": xgb_hits_10_tail,
            "mr_head": mr_head,
            "mrr_head": mrr_head,

            "hits_1_head": xgb_hits_1_head,
            "hits_3_head": xgb_hits_3_head,
            "hits_5_head": xgb_hits_5_head,
            "hits_10_head": xgb_hits_10_head,
            "mr_tail": mr_tail,
            "mrr_tail": mrr_tail}

    labels_scores_dict = {"labels": labels, "scores": scores}

    return hits, labels_scores_dict


def run_model(year: int, literals_df: pd.DataFrame, directors_df: pd.DataFrame, ma_df: pd.DataFrame, candidates: list,
              ratio):
    """
    Learns until year including and predicts for following year.
    """
    # Define the results dir
    results_dir = f"results_{ratio}"

    # First I want all the data to be prepared ahead of time of need - time complexity
    ma_participants_appendix = list(set(list(ma_df["head"]) + list(ma_df["tail"])) - set(candidates))
    data_df, mutual_directors, weighted_mutual_directors, cik_to_idx, connection_comp_df = get_data_prepared_ahead(
        candidates + ma_participants_appendix,
        directors_df,
        literals_df,
        max_year=year)
    X_train, X_test, y_train, y_test, test_content = dynamic_building_of_data(year, ma_df, candidates, data_df,
                                                                              mutual_directors,
                                                                              weighted_mutual_directors,
                                                                              cik_to_idx,
                                                                              connection_comp_df, ratio)

    with open(join(results_dir, f"model_for_{year}.pkl"), "rb") as file:
        model = pickle.load(file)

    # Find the training tpr
    y_pred = model.predict_proba(X_train)[:, 1]
    fpr, tpr, _ = roc_curve(y_train, y_pred)
    tpr_01 = [tpr[i] for i in range(len(fpr)) if fpr[i] <= 0.1][-1]
    print(f"\t\tRATIO: {ratio}\tYear: {year}\tTraining TPR: {tpr_01}")


def run_for_ratio(ratio):
    results_dir = f"results_{ratio}"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load literals DataFrame
    literals_df = pd.read_csv(FEATURES_TABLE, index_col=0)  # Literal 1, Literal 2, ..., year, cik
    directors_df = pd.read_csv(DIRECTORS_DF_PATH, index_col=0)  #
    ma_df = pd.read_csv(MA_PATH, index_col=0)

    # Fix some stuff
    literals_df["year"] = literals_df["year"].astype(int)
    literals_df["cik"] = literals_df["cik"].astype(str)
    literals_df["cik"] = literals_df["cik"].apply(lambda x: x.split(".")[0])

    ma_df["head"], ma_df["tail"] = ma_df["head"].astype(str), ma_df["tail"].astype(str)
    ma_df["year"] = ma_df["year"].astype(int)
    directors_df["cik"] = directors_df["cik"].astype(str)
    directors_df["year"] = directors_df["year"].astype(int)

    range_years = range(2011, 2019 + 1)
    threads = []
    # Running in threads
    for year in range_years:
        # For each year, our goal is to learn until that year, includes, and predict for the following year.
        thread = threading.Thread(target=run_model,
                                  args=(year, literals_df, directors_df, ma_df,
                                        list(set(CANDIDATES[str(year)])), ratio))
        thread.start()
        threads.append(thread)
    # Wait for all the threads to be done
    for thread in threads:
        thread.join()


def main():
    threads = []
    for ration in [3, 6, 9, 10, 12, 15]:
        thread = threading.Thread(target=run_for_ratio, args=(ration,))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
