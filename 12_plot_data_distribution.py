"""
Yuli Tshuva
Plotting the data distribution for the paper:

V   1) Interlocking Directors per year
V   2) Directors degree distribution
V   3) Amount of Directors per year
V   4) Market value distribution
5) Companies degree distribution
V   6) Amount of M&A per year
V   7) Industrial sectors per year
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random

DIRECTORS_DF_PATH = "analyze_data/directors_parsed.csv"
MA_PATH = "analyze_data/M&A_processed.csv"
FEATURES_TABLE = "literals_mining/merged_data.csv"
YEVGENI_DATA = "literals_mining/augmented_filtered_data.csv"
RANDOM_YEAR = random.randint(2015, 2019)


def find_directors_chars(directors_df):
    # Set initial dictionaries
    interlocking_directors_per_year, directors_per_year = {}, {}
    for year in np.unique(directors_df["year"]):
        # Find total amount of total directors for year
        directors_per_year[year] = len(set(directors_df[directors_df["year"] == year]["director"]))
        # Find interlocking directors for year
        directors, counts = np.unique(directors_df[directors_df["year"] == year]["director"], return_counts=True)
        interlocking_directors_per_year[year] = int(np.sum([c for c in counts if c > 1]))

    return interlocking_directors_per_year, directors_per_year


def find_mas_per_year(ma_df):
    mas_per_year = {}
    # Iterate through years
    for year in np.unique(ma_df["year"]):
        # Find the amount of M&A for year
        mas_per_year[year] = len(ma_df[ma_df["year"] == year])

    return mas_per_year


def find_industrial_sectors_per_year(narrowed_data):
    # Set initial dictionary
    industrial_sectors_per_year = []
    # Find all sectors
    all_sectors = np.unique(narrowed_data["industry"])
    # Iterate through years
    for year in np.unique(narrowed_data["fyear"]):
        # Find the amount of industrial sectors for year
        sectors, counts = np.unique(narrowed_data[narrowed_data["fyear"] == year]["industry"], return_counts=True)
        # Set in dict
        sectors_count = {sector: count for sector, count in zip(sectors, counts)}

        # Add to list
        year_list = []
        for sector in all_sectors:
            if sector in sectors_count:
                year_list.append(sectors_count[sector])
            else:
                year_list.append(0)
        # Add dict to total
        industrial_sectors_per_year.append(year_list)

    # Create DataFrame
    industrial_sectors_per_year = pd.DataFrame(industrial_sectors_per_year,
                                               index=np.unique(narrowed_data["fyear"]), columns=all_sectors)

    return industrial_sectors_per_year


def find_directors_degree_distribution(directors_df):
    # Find the degree distribution of directors
    directors, counts = np.unique(directors_df[directors_df["year"] == RANDOM_YEAR]["director"], return_counts=True)
    # Set in dictionary
    return counts


def find_companies_degree_distribution(directors_df):
    # Find the degree distribution of directors
    ciks, counts = np.unique(directors_df[directors_df["year"] == RANDOM_YEAR]["cik"], return_counts=True)
    # Set in dictionary
    return counts


def find_at_distribution(narrowed_data):
    # Find the distribution of the AT
    at_distribution = list(narrowed_data[narrowed_data["fyear"] == RANDOM_YEAR]["at"])
    return at_distribution


def main():
    # Set an axis and a figure
    fig, ax = plt.subplots(3, 3, figsize=(18, 15))

    # Load the directors data
    directors_df = pd.read_csv(DIRECTORS_DF_PATH, index_col=0)
    # Drop one row with missing director
    directors_df.dropna(inplace=True)

    # Load the M&A data
    ma_df = pd.read_csv(MA_PATH, index_col=0)

    # Load Yevgeni's data
    narrowed_data = pd.read_csv(YEVGENI_DATA, index_col=0)
    narrowed_data["fyear"] = narrowed_data["fyear"].astype(int)

    # Get directors characteristics
    interlocking_directors_per_year, directors_per_year = find_directors_chars(directors_df)
    # Plot
    ax[0, 0].bar(interlocking_directors_per_year.keys(), interlocking_directors_per_year.values(),
                 color="salmon", edgecolor="black")
    ax[0, 0].set_title("Interlocking Directors per year")
    ax[0, 0].set_xlabel("Year")
    ax[0, 0].set_ylabel("Amount")
    ax[0, 2].bar(directors_per_year.keys(), directors_per_year.values(),
                 color="salmon", edgecolor="black")
    ax[0, 2].set_title("Directors per year")
    ax[0, 2].set_xlabel("Year")
    ax[0, 2].set_ylabel("Amount")

    # Get directors degree distribution
    directors_degree_distribution = find_directors_degree_distribution(directors_df)
    # Plot
    ax[0, 1].hist(directors_degree_distribution, bins=25,
                  color="salmon", edgecolor="black", log=True)
    ax[0, 1].set_title("Directors degree distribution")
    ax[0, 1].set_xlabel("Degree")
    ax[0, 1].set_ylabel("Frequency")

    # Get AT distribution
    at_distribution = find_at_distribution(narrowed_data)
    # Plot
    ax[1, 0].hist(at_distribution, bins=30,
                  color="lightblue", edgecolor="black", log=True)
    ax[1, 0].set_title("AT distribution")
    ax[1, 0].set_xlabel("AT")
    ax[1, 0].set_ylabel("Frequency")

    # Get companies degree distribution
    companies_degree_distribution = find_companies_degree_distribution(directors_df)
    # Plot
    ax[1, 1].hist(companies_degree_distribution, bins=35,
                  color="lightblue", edgecolor="black", log=True)
    ax[1, 1].set_title("Companies degree distribution")
    ax[1, 1].set_xlabel("Degree")
    ax[1, 1].set_ylabel("Frequency")

    # Find the amount of M&A per year
    mas_per_year = find_mas_per_year(ma_df)
    ax[1, 2].bar(mas_per_year.keys(), mas_per_year.values(), color="turquoise", edgecolor="black")
    ax[1, 2].set_title("Amount of M&A per year")
    ax[1, 2].set_xlabel("Year")
    ax[1, 2].set_ylabel("Amount")

    # Find industrial sectors per year
    industrial_sectors_df = find_industrial_sectors_per_year(narrowed_data)
    # Plot the data distribution
    row_ax = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    industrial_sectors_df.plot(ax=row_ax, kind='bar', stacked=True)
    row_ax.set_title("Industrial sectors distribution per year")
    row_ax.set_xlabel("Year")
    row_ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("plots_for_paper/data_distribution.png")
    plt.show()


if __name__ == "__main__":
    main()
