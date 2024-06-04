"""
Yuli Tshuva
Initial look at the data
I actually protest against functioning my code this time (I tried that before, and it was a mess).
"""

# Imports
import json
from os.path import join
import pandas as pd

# Set the data path - Somehow it only works with full path
DATA_PATH = r"C:\Users\ThinkBook\Documents\M&A\directors_data"

# Get the M&A DataFrame
ma_df = pd.read_csv(join(DATA_PATH, "M&A.csv"))

# Index over the interesting parts
ma_df = ma_df[["DateAnnounced", "DateEffective", "AcquirorCIK", "TargetCIK"]]

# Customize columns names
ma_df.columns = ["default date", "date", "head", "tail"]

# Set the date
count_changes = 0
for i in range(ma_df.shape[0]):
    if type(ma_df.iloc[i]["date"]) == float:
        ma_df.at[i, "date"] = ma_df.iloc[i]["default date"]
        count_changes += 1

# Report the amount of changes in code
print(f"There were {count_changes} changes in date.")

# Drop the default date column
ma_df.drop("default date", axis=1, inplace=True)

# Set data types
ma_df["head"], ma_df["tail"] = ma_df["head"].astype(str), ma_df["tail"].astype(str)

# Extract the year out of the date
ma_df["year"] = ma_df["date"].apply(lambda x: x[-4:])

# View the result
print(ma_df.info())

# Save the dataframe
ma_df.to_csv("M&A_processed.csv")

# Get the companies dataframe
comps_df = pd.read_csv(join(DATA_PATH, "companies.csv"))

# Very fast it goes from df to list
companies = list(comps_df["cik"].astype(str))

# View first 10 companies
print(companies[:10])

# Save the companies in a json format
with open("companies.json", "w") as f:
    json.dump({"companies": companies}, f)
