"""
Yuli Tshuva
Aggregate through the data.
"""

from os import listdir
from os.path import join
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# Set the data path - Somehow it only works with full path
DATA_PATH = join("..", "directors_data")
COMPANIES_PATH = "companies.json"
MA_PATH = "M&A_processed.csv"

# Get the folders in the data
dirs = [folder for folder in listdir(DATA_PATH) if folder.isdigit()]

# load the companies we look for
with open(COMPANIES_PATH, "r") as f:
    companies = json.load(f)

# Get a list of our 21495 companies
companies = companies["companies"]

# Get a ciks per year dictionary
ciks_per_year = {}

percents = [[], []]
# Iterate through the directories
for dir in dirs:
    dir_name = join(DATA_PATH, dir, dir)
    dir_files = listdir(dir_name)
    dir_ciks = [file.split("_")[0] for file in dir_files]
    ciks_per_year[int(dir)] = dir_ciks
    print(dir, ":", len(set(companies) & set(dir_ciks)), "/", len(dir_ciks))
    percents[1].append(len(set(companies) & set(dir_ciks)) / len(dir_ciks) * 100)
    percents[0].append(int(dir))

plt.plot(percents[0], percents[1])
plt.title("How many companies in the data\nare relevant in each year.")
plt.ylabel("Percents")
plt.xlabel("Year")
plt.show()

# Get the M&A data
ma_df = pd.read_csv(MA_PATH, index_col=0)

# Fix the dtype problem
ma_df["head"], ma_df["tail"] = ma_df["head"].astype(str), ma_df["tail"].astype(str)

print(ma_df.head())

# Get a list of all the years which a M&A took place at
years = np.unique(ma_df["year"])
print(years, years.dtype)

ma_companies_per_year = {}
transaction_per_year = [[], []]
for year in years:
    df = ma_df[ma_df["year"] == year]
    ma_companies_per_year[year] = list(set(list(df["head"]) + list(df["tail"])))
    print(year, ":", df.shape[0])
    transaction_per_year[1].append(df.shape[0])
    transaction_per_year[0].append(year)

plt.plot(transaction_per_year[0], transaction_per_year[1])
plt.title("M&A transactions per year")
plt.xlabel("Year")
plt.show()

cum = [transaction_per_year[1][0]] + [0 for i in range(len(transaction_per_year[1]) - 1)]
for i in range(1, len(transaction_per_year[1])):
    cum[i] = cum[i-1] + transaction_per_year[1][i]
plt.bar(transaction_per_year[0], cum)
plt.title("M&A cumulative transactions per year")
plt.xlabel("Year")
plt.show()

print("\nM&A Companies per year")
matching_percents_for_year = [[], []]
for key, value in ma_companies_per_year.items():
    try:
        x = len(set(value) & set(ciks_per_year[key-2]))
        percents = x/len(value)*100
        print(key, ":", x, "/", len(value), ":", f"{percents:.2f}%")
        matching_percents_for_year[0].append(key)
        matching_percents_for_year[1].append(percents)
    except:
        print("No data for:", key)

plt.plot(matching_percents_for_year[0], matching_percents_for_year[1])
plt.title("M&A companies we have data for (in two years before) per year")
plt.xlabel("Year")
plt.ylabel("Percents")
plt.show()

print("\nM&A Companies per year")
matching_percents_for_year = [[], []]
for key, value in ma_companies_per_year.items():
    try:
        x = len(set(value) & set(ciks_per_year[key-1]))
        percents = x/len(value)*100
        print(key, ":", x, "/", len(value), ":", f"{percents:.2f}%")
        matching_percents_for_year[0].append(key)
        matching_percents_for_year[1].append(percents)
    except:
        print("No data for:", key)

plt.plot(matching_percents_for_year[0], matching_percents_for_year[1])
plt.title("M&A companies we have data for (in a year before) per year")
plt.xlabel("Year")
plt.ylabel("Percents")
plt.show()

print("\nM&A Companies per year")
matching_percents_for_year = [[], []]
for key, value in ma_companies_per_year.items():
    try:
        x = len(set(value) & set(ciks_per_year[key-1] + ciks_per_year[key-2]))
        percents = x/len(value)*100
        print(key, ":", x, "/", len(value), ":", f"{percents:.2f}%")
        matching_percents_for_year[0].append(key)
        matching_percents_for_year[1].append(percents)
    except:
        print("No data for:", key)

plt.plot(matching_percents_for_year[0], matching_percents_for_year[1])
plt.title("M&A companies we have data for\n(in both year and two years before) per year")
plt.xlabel("Year")
plt.ylabel("Percents")
plt.show()
