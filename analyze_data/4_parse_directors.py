"""
Yuli Tshuva
Parsing the directors' data.
"""

import pandas as pd
import numpy as np
from time import sleep

# Get the dataframe
df = pd.read_csv("directors.csv", index_col=0)

# Change the data type of cik
df["cik"] = df["cik"].astype(str)

# Parse the dataframe
df.reset_index(drop=True, inplace=True)

nans = 0


def parse_directors(director_name):
    try:
        new_name = ""
        # Parse unnecessary symbols
        for char in director_name:
            if char.isalpha() or char == " ":
                new_name += char
        # Get the name in upper letters
        new_name = new_name.upper()
        # Delete duplications
        name_by_list = new_name.split()
        new_list = []
        for name in name_by_list:
            if not name in new_list:
                new_list.append(name)
        if len(new_list) <= 1 or len(new_list) >= 5:
            return np.nan
        final_name = "".join(new_list)
        if "NAME" in final_name or "TITLE" in final_name:
            return np.nan
        return final_name
    except:
        global nans
        nans += 1
        return np.nan


# Check data types
print(df.info())

# Parse the directors' names
df["director"] = df["director"].apply(parse_directors, convert_dtype=False)

print("There are", nans, "nan values in the directors column.")
print(df.shape[0] - df.drop_duplicates(subset=["director", 'cik', 'year']).shape[0], "duplicates were deleted.")

# Delete the duplicates
df.drop_duplicates(subset=["director", 'cik', 'year'], inplace=True)

print("The shape of the DataFrame is:", df.shape)
df.dropna(inplace=True)
print("The shape of the DataFrame without nan values is", df.shape)

# Reset the indexes again
df.reset_index(drop=True, inplace=True)

# Save the output
df.to_csv("directors_parsed.csv")
