"""
Yuli Tshuva
"""
"""
Yuli Tshuva
Set up the code of XGBoost to run on the servers.
"""

import json
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna
import warnings
import threading
import os
import gc

# Filter a lot of unnecessary warnings
warnings.filterwarnings("ignore")
# Set weight decay for the weighted directors
ALPHA = 0.7

print("Start Running.")

with open("find_candidates_per_year/candidates.json", "r") as f:
    CANDIDATES = json.load(f)  ### dictionary keys are strings

MA_PATH = "analyze_data/M&A_processed.csv"
DIRECTORS_DF_PATH = "analyze_data/directors_parsed.csv"
FEATURES_TABLE = "literals_mining/features_table.csv"
TOPO_FEATURES = "topo_features/network_features_for_year.csv"


### Helper functions ###

def create_false_samples(df, companies, factor=5):
    """I beleive the name of the function speaks for itself."""
    FALSE_SAMPLES = int(df.shape[0] * factor)

    heads, tails, labels = [], [], []
    for i in range(FALSE_SAMPLES + 1):
        head = random.choice(companies)
        tail = random.choice(companies)
        if ((df['head'] == head) & (df['tail'] == tail)).any():
            continue
        label = 0
        heads.append(head)
        tails.append(tail)
        labels.append(label)

    temp_df = pd.DataFrame({"head": heads, "tail": tails, "label": labels})
    df = pd.concat([df, temp_df])

    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)
    return df


def find_amount_of_mutual_directors(directors_df: pd.DataFrame, cik1: str, cik2: str, max_year: int):
    temp_df = directors_df[directors_df["cik"].isin([cik1, cik2])]
    temp_df = temp_df[temp_df["year"] <= max_year]
    temp_df.drop("year", axis=1, inplace=True)
    temp_df.drop_duplicates(subset=["director", "cik"], inplace=True)
    directors = list(temp_df["director"])
    # By doing set on the list I get rid of all the mutual directors.
    # Hence, if we take the len of the list and return the remaining len.
    # We get the exact amount we've got rid of.
    mutual_directors = len(directors) - len(set(directors))
    return mutual_directors


def find_weighted_amount_of_mutual_directors(directors_df: pd.DataFrame, cik1: str, cik2: str, max_year: int):
    temp_df = directors_df[directors_df["cik"].isin([cik1, cik2])]
    temp_df = temp_df[temp_df["year"] <= max_year]
    weighted_mutual_directors = 0
    years_lst = list(temp_df["year"])
    if len(years_lst) == 0:
        return 0
    years_range = range(min(years_lst), max(years_lst) + 1)
    for year in years_range:
        year_df = temp_df[temp_df["year"] == year]
        year_df.drop_duplicates(subset=["director", "cik"], inplace=True)
        directors = list(year_df["director"])
        # By doing set on the list I get rid of all the mutual directors.
        # Hence, if we take the len of the list and return the remaining len.
        # We get the exact amount we've got rid of.
        mutual_directors = len(directors) - len(set(directors))
        weighted_mutual_directors *= ALPHA
        weighted_mutual_directors += mutual_directors
    return weighted_mutual_directors


def get_dirs_data(directors_df: pd.DataFrame, cik1: str, cik2: str, max_year: int):
    """A nice function for directors data - unweighted and weighted"""
    return (find_amount_of_mutual_directors(directors_df, cik1, cik2, max_year),
            find_weighted_amount_of_mutual_directors(directors_df, cik1, cik2, max_year))


def get_literals(literals_df: pd.DataFrame, cik: str, year: int):
    """
    Get the literals dataframe and cik and find the literals of the cik (for relevant year).
    Gets as a very long dictionary
    """
    results_df = literals_df[(literals_df["cik"] == cik) & (literals_df["year"] == year)]
    results_df.drop(["cik", "year"], axis=1, inplace=True)
    return results_df


def get_topological_features(topo_df: pd.DataFrame, cik: str):
    """
    Get year and extract the topological features to the year (of the directors graph).
    Return a dataframe of all sort of features.
    """
    try:
        return pd.DataFrame(topo_df.loc[int(cik)]).T
    except:
        return pd.DataFrame([{col: np.nan for col in topo_df.columns}])


### MOST HEAVY FUNCTION - build function

def process_df(df: pd.DataFrame, directors_df: pd.DataFrame, literals_df: pd.DataFrame, max_year: int):
    """
    Get a dataframe of with columns 'head', 'tail' and return the dataframe enriched
    with all the data we can possibly infer.
    Important note: we don't change any columns in the dataframe but adding new ones and
    dropping 'head' and 'tail'.

    df: input dataframe
    directors_df: director,cik,year - pretty strait forwards.
    literals_df: characteristic1,characteristic2,...,year,cik
    max_year: max_year
    return: edited dataframe
    """
    # Open graph topological features for the most relevant year
    topological_features_df = pd.read_csv(TOPO_FEATURES.replace("year", str(max_year)), index_col=0)

    # Get the heads and tails in lists
    df["head"], df["tail"] = df["head"].astype(str), df["tail"].astype(str)
    heads, tails = list(df["head"]), list(df["tail"])

    # Set empty lists to collect data
    heads_literals, tails_literals = [], []
    head_topological_features, tail_topological_features = [], []
    mutual_directors, weighted_mutual_directors = [], []
    for head, tail in zip(heads, tails):
        heads_literals.append(get_literals(literals_df, head, max_year))
        tails_literals.append(get_literals(literals_df, tail, max_year))

        head_topological_features.append(get_topological_features(topological_features_df, head))
        tail_topological_features.append(get_topological_features(topological_features_df, tail))

        mutual_dirs, weighted_mutual_dirs = get_dirs_data(directors_df, head, tail, max_year)
        mutual_directors.append(mutual_dirs)
        weighted_mutual_directors.append(weighted_mutual_dirs)

    # concat all collected data into one df and rename the columns
    heads_literals_df = pd.concat(heads_literals, axis=0, ignore_index=True)
    heads_literals_df.columns = ["head_" + str(col) for col in heads_literals_df]

    tails_literals_df = pd.concat(tails_literals, axis=0, ignore_index=True)
    tails_literals_df.columns = ["tail_" + str(col) for col in tails_literals_df]

    head_topological_features_df = pd.concat(head_topological_features, axis=0, ignore_index=True)
    head_topological_features_df.columns = ["head_" + str(col) for col in head_topological_features_df]

    tail_topological_features_df = pd.concat(tail_topological_features, axis=0, ignore_index=True)
    tail_topological_features_df.columns = ["tail_" + str(col) for col in tail_topological_features_df]

    del df

    df = pd.concat(
        [heads_literals_df, tails_literals_df, head_topological_features_df, tail_topological_features_df],
        axis=1,
    )

    df["mutual_directors"] = mutual_directors
    df["weighted_mutual_directors"] = weighted_mutual_directors

    df.reset_index(inplace=True)

    # Drop neglected column created "index"
    df.drop("index", axis=1, inplace=True)

    del topological_features_df, heads, tails, heads_literals, tails_literals
    del head_topological_features, tail_topological_features, mutual_directors, weighted_mutual_directors

    gc.collect()

    if df.shape[0] > 10000:
        df.to_csv("results/example_df.csv")

    return df


### Mass functions


def dynamic_building_of_data(year, ma_df, directors_df, literals_df, candidates):
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
    test_data = test_data[test_data["head"].isin(candidates)]
    test_data = test_data[test_data["tail"].isin(candidates)]

    # Add negative samples
    train_data = create_false_samples(train_data, candidates, factor=5)
    test_data = create_false_samples(test_data, candidates, factor=5)

    # Shuffle training data
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    X_train = process_df(train_data, directors_df, literals_df, year)
    X_test = process_df(test_data, directors_df, literals_df, year + 1)
    y_train, y_test = list(train_data["label"]), list(test_data["label"])

    test_data_idx = list(test_data.index)
    test_content = [[str(test_data.loc[idx]["head"]), str(test_data.loc[idx]["head"])] for idx in test_data_idx if
                    int(test_data.loc[idx]["label"]) == 1]

    return X_train, X_test, y_train, y_test, test_content


def training_and_tuning_model_on_data(X_train, X_test, y_train, y_test, year, trails_num=1000):
    if not os.path.exists(f"results/model_params_both_for_{year}.json"):
        xgb = XGBClassifier()
        xgb.fit(X_train, y_train)
        xgb_pred_bin = xgb.predict(X_test)

        xgb_default_acc = accuracy_score(y_test, xgb_pred_bin)

        X_train_opt, X_valid_opt, y_train_opt, y_valid_opt = train_test_split(X_train, y_train,
                                                                              test_size=0.2,
                                                                              random_state=42)

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
                'use_label_encoder': False
            }

            # Fit the model
            optuna_model = XGBClassifier(**params)
            optuna_model.fit(X_train_opt, y_train_opt)

            # Make predictions
            y_pred = optuna_model.predict(X_valid_opt)

            # Evaluate predictions
            accuracy = accuracy_score(y_valid_opt, y_pred)
            return accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=trails_num)
        trial = study.best_trial
        params = trial.params
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_hpo = accuracy_score(y_test, y_pred)

        with open(f"results/model_params_both_for_{year}.json", "w") as file:
            json.dump(params, file)

        if xgb_default_acc > acc_hpo:
            model = XGBClassifier()
            model.fit(X_train, y_train)

            with open(f"results/model_params_both_for_{year}.json", "w") as file:
                json.dump({"params": "default_params"}, file)

    else:
        with open(f"results/model_params_both_for_{year}.json", "r") as file:
            params = json.load(file)
        if "params" in params:
            model = XGBClassifier()
        else:
            model = XGBClassifier(**params)
        model.fit(X_train, y_train)

    return model


def hits_calculations(model, test_content, directors_df, literals_df, year, candidates):
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
    for tup in test_content:
        count += 1
        print(f"Year {year} - {count}/{len(test_content)}")
        tail = tup[1]
        head = tup[0]

        # Create a temporary df for evaluation
        temp_df = pd.DataFrame({"head": candidates, "tail": [tail for _ in range(len(candidates))]})
        temp_df = process_df(temp_df, directors_df, literals_df, year)

        xgb_head_scores = list(np.array(model.predict_proba(temp_df))[:, 1])
        xgb_top_heads = [candidates[i] for i in list(np.argsort(xgb_head_scores)[::-1])]

        xgb_hits_1_head += 1 if head in xgb_top_heads[:1] else 0
        xgb_hits_3_head += 1 if head in xgb_top_heads[:3] else 0
        xgb_hits_5_head += 1 if head in xgb_top_heads[:5] else 0
        xgb_hits_10_head += 1 if head in xgb_top_heads[:10] else 0
        rank = xgb_top_heads.index(head) + 1
        mr_head += rank
        mrr_head += 1 / rank

        # Create a temporary df for evaluation
        temp_df = pd.DataFrame({"head": [head for _ in range(len(candidates))], "tail": candidates})
        temp_df = process_df(temp_df, directors_df, literals_df, year)

        xgb_tail_scores = list(np.array(model.predict_proba(temp_df))[:, 1])
        xgb_top_tails = [candidates[i] for i in list(np.argsort(xgb_tail_scores)[::-1])]

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

    return hits


def run_model(year: int, literals_df: pd.DataFrame, directors_df: pd.DataFrame, ma_df: pd.DataFrame, candidates: list):
    """
    Learns until year including and predicts for following year.
    """
    X_train, X_test, y_train, y_test, test_content = dynamic_building_of_data(year, ma_df, directors_df, literals_df,
                                                                              candidates)
    model = training_and_tuning_model_on_data(X_train, X_test, y_train, y_test, year)
    hits = hits_calculations(model, test_content, directors_df, literals_df, year, candidates)

    with open(f"results/hits_both_for_{year}.json", "w") as file:
        json.dump(hits, file)

    print("Year", year, "done.")


def main():
    # Load literals DataFrame
    literals_df = pd.read_csv(FEATURES_TABLE, index_col=0)  # Literal 1, Literal 2, ..., year, cik
    directors_df = pd.read_csv(DIRECTORS_DF_PATH, index_col=0)  #
    ma_df = pd.read_csv(MA_PATH, index_col=0)

    # Fix some stuff
    literals_df["year"] = literals_df["year"].astype(int)
    literals_df["cik"] = literals_df["cik"].astype(str)
    ma_df["head"], ma_df["tail"] = ma_df["head"].astype(str), ma_df["tail"].astype(str)
    ma_df["year"] = ma_df["year"].astype(int)
    directors_df["cik"] = directors_df["cik"].astype(str)
    directors_df["year"] = directors_df["year"].astype(int)

    range_years = range(2011, 2020)
    threads = []
    for year in range_years:
        thread = threading.Thread(target=run_model,
                                  args=(year, literals_df, directors_df, ma_df, CANDIDATES[str(year)]))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    print("All Done.")


if __name__ == "__main__":
    main()
