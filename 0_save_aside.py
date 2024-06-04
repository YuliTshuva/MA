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