"""
Yuli Tshuva
Merging between the mined data and Yevegny's data
"""

import pandas as pd
import os
import shutil
import numpy as np
from tqdm.auto import tqdm
import random

# First get yevgeny's data to dir
# If not already been copied
if not os.path.exists("augmented_filtered_data.csv"):
    # Copy to current directory
    shutil.copy("../yevgeny_data/augmented_filtered_data.csv", "augmented_filtered_data.csv")

# Load the data
yevgeny_data = pd.read_csv("augmented_filtered_data.csv")

# Load the mined data
mined_data = pd.read_csv("features_table.csv", index_col=0)

### Start merging the data

# Adjust data types
yevgeny_data["cik"] = yevgeny_data["cik"].astype(np.int64)
yevgeny_data["year"] = yevgeny_data["fyear"].astype(np.int64)

# Drop fyear column since it was replaced by year
yevgeny_data.drop("fyear", axis=1, inplace=True)

# Sort the dfs by the year
yevgeny_data.sort_values(by=["year", "cik"], inplace=True)
mined_data.sort_values(by=["year", "cik"], inplace=True)

# Reset the indexes
yevgeny_data.reset_index(drop=True, inplace=True)
mined_data.reset_index(drop=True, inplace=True)

# One hot encode the industry column
industries = pd.unique(yevgeny_data["industry"])
for industry in industries:
    yevgeny_data["industry_" + industry] = (yevgeny_data["industry"] == industry).apply(lambda x: 1 if x else 0)
yevgeny_data.drop("industry", axis=1, inplace=True)

# View an example
print("Example for a row in yevgeny data:")
print(yevgeny_data.iloc[5])

# Setup need to be update cols
update_cols = [col for col in yevgeny_data.columns if col not in ["year", "cik"]]

# Add columns to mined data and yevgeny data
for col in yevgeny_data.columns:
    if col not in mined_data.columns:
        mined_data[col] = np.nan

for col in mined_data.columns:
    if col not in yevgeny_data.columns:
        yevgeny_data[col] = np.nan

# Fix dtypes
mined_data["year"] = mined_data["year"].astype(np.int64)
mined_data["cik"] = mined_data["cik"].astype(np.int64)

count1, count2 = 0, 0
# Add data to the mined data
for i in tqdm(range(yevgeny_data.shape[0]), desc="Merging data"):
    # Get the year and cik
    year = yevgeny_data.loc[i, "year"]

    cik = yevgeny_data.loc[i, "cik"]

    # Find the row in mined data
    mined_row = mined_data[(mined_data["year"] == year) & (mined_data["cik"] == cik)]

    # Check for singular value
    if mined_row.shape[0] > 1:
        raise Exception(f"Duplication in mined data for year {year} and cik {cik}.")

    # Check if the row exists
    if mined_row.shape[0] == 0:
        # Add the row to mined_data
        mined_rows = mined_data.shape[0]
        mined_data.loc[mined_rows] = yevgeny_data.loc[i]
        count1 += 1
    else:
        count2 += 1
        # Update the row
        idx = mined_row.index
        for col in update_cols:
            mined_data.loc[idx, col] = yevgeny_data.loc[i, col]

print("Computation Report:")
print(f"Out of {yevgeny_data.shape[0]} rows, {count1} rows were added to mined data and {count2} rows were updated.")

# Save the result
mined_data.to_csv("merged_data.csv")
