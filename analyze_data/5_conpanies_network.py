"""
Let's build the companies network based on the directors graph.
"""

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore")


def create_companies_network(year, df, net_arr=None):
    network = []

    df = df[df["year"] == year]
    df.drop("year", axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"Year {year}: {df.shape[0]} records to work on.")
    all_directors_exist = list(set(df["director"]))
    for director in all_directors_exist:
        director_df = df[df["director"] == director]
        ciks_with_director = list(set(director_df["cik"]))
        for i in range(len(ciks_with_director)):
            for j in range(len(ciks_with_director)):
                network.append([ciks_with_director[i], ciks_with_director[j]])

    network = np.array(network)
    if net_arr is not None:
        network = np.vstack([net_arr, network])
    network = pd.DataFrame(network)
    network.drop_duplicates(inplace=True)
    network = network.to_numpy()

    with open(f"companies_network/network_until_{year}.txt", 'w') as f:
        for i in range(network.shape[0]):
            f.write(f"{network[i, 0]},{network[i, 1]}\n")

    return network


def main():
    # Read the dataframe
    df = pd.read_csv("directors_parsed.csv", index_col=0)
    years = list(set(df["year"]))

    temp_df = df.copy()
    net_arr = create_companies_network(1994, temp_df)
    del temp_df

    for year in tqdm(years[1:]):
        temp_df = df.copy()
        net_arr = create_companies_network(year, temp_df, net_arr)
        del temp_df


if __name__ == "__main__":
    main()
