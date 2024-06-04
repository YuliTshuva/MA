"""
Let's build the companies network based on the directors graph.
"""

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
from os.path import join
import threading

warnings.filterwarnings("ignore")


def create_companies_network(year, df, net_arr=None):
    # Setup the network
    network = []

    # Filter the dataframe
    df = df[df["year"] == year].copy()
    df.drop("year", axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    # Print status
    print(f"[INFO] Year {year}: {df.shape[0]} records to work on.")

    # Set a list for the directors
    all_directors_exist = list(set(df["director"]))
    for director in all_directors_exist:
        # Get the director part of the df
        director_df = df[df["director"] == director]
        # Get the ciks the director connected to
        ciks_with_director = list(set(director_df["cik"]))
        for i in range(len(ciks_with_director)):
            for j in range(i + 1, len(ciks_with_director)):
                network.append([ciks_with_director[i], ciks_with_director[j]])

    if net_arr:
        network += net_arr

    temp_df = pd.DataFrame({"cik1": [n[0] for n in network], "cik2": [n[1] for n in network]})
    temp_df.drop_duplicates(inplace=True)
    network = temp_df.to_numpy().tolist()

    return network


def connection_components(network):
    """Get network and return a dictionary from each company to its component size."""
    # Find vertices in network
    idx_to_vertices, vertices_to_idx = [], {}
    for pair in network:
        for cik in pair:
            if cik not in vertices_to_idx:
                idx_to_vertices.append(cik)
                vertices_to_idx[cik] = len(idx_to_vertices) - 1

    # Create the union-find data structure
    union_find = np.array(list(range(len(idx_to_vertices))))

    # Update the union find for each connection
    for pair in network:
        pair_ids = [vertices_to_idx[pair[0]], vertices_to_idx[pair[1]]]
        if union_find[pair_ids[0]] == union_find[pair_ids[1]]:
            continue
        pair_uf = [union_find[pair_ids[0]], union_find[pair_ids[1]]]
        if pair_uf[0] < pair_uf[1]:
            small, big = pair_ids
        else:
            big, small = pair_ids
        update_places = np.where(union_find == union_find[big])
        union_find[update_places] = union_find[small]

    # Sum up and save
    cik_to_component = {cik: union_find[vertices_to_idx[cik]] for cik in idx_to_vertices}
    components, sizes = np.unique(union_find, return_counts=True)
    component_to_size = {c: s for c, s in zip(components, sizes)}

    return cik_to_component, component_to_size


def helper_function(new_network, year):
    cik_to_component, component_to_size = connection_components(new_network)
    connections_components_df = pd.DataFrame({"cik": list(cik_to_component.keys()),
                                              "component": list(cik_to_component.values()),
                                              "size": [component_to_size[c] for c in cik_to_component.values()]})
    connections_components_df.to_csv(join("..", "connection_components",
                                          f"connections_components_{year}.csv"))


def main():
    # Read the dataframe
    df = pd.read_csv("directors_parsed.csv", index_col=0)
    df["year"] = df["year"].astype(np.int64)
    df["cik"] = df["cik"].astype(np.int64)

    years = sorted(list(set(df["year"])))

    network = []

    for year in tqdm(years):
        new_network = create_companies_network(year, df, network)

        thread = threading.Thread(target=helper_function, args=(new_network, year))
        thread.start()

        network = new_network


if __name__ == "__main__":
    main()
