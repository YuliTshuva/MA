"""
Yuli Tshuva
Parsing the data received
"""

import json
import pandas as pd
import numpy as np

# Load the features
with open("companies_features.json", "r") as f:
    companies_features = json.load(f)

# Check some stuff
chars = []
vals = list(companies_features.values())

print("We have literals values for currently:", len(vals), "companies.")

for val in vals:
    for key in val.keys():
        if key not in chars:
            chars.append(key)

print("The characteristics of the companies are:")
print(chars)


# Now, let's fix it real nice in format of company_cik ->
# {year1: {'Assets': XXXX, ...}, year2: {'Assets': XXXX, ...}, ...}

def transform_value(value):
    years = {}
    for key in value:
        for dct in value[key]:
            if not list(dct.keys())[0] in years:
                years[list(dct.keys())[0]] = {key: list(dct.values())[0]}
            else:
                years[list(dct.keys())[0]][key] = list(dct.values())[0]
    return years


features = {key: transform_value(value) for key, value in companies_features.items()}

with open("companies_features_organized.json", 'w') as file:
    json.dump(features, file)

companies = list(features.keys())


def get_value(dct, value):
    if value in dct:
        return dct[value]
    return np.nan


lst = []
for company in companies:
    dct = features[company]
    for year in dct:
        lst.append([get_value(dct[year], ch) for ch in chars] + [int(year), company])

arr = np.array(lst)

df = pd.DataFrame(data=arr, columns=chars + ["year", "cik"])
df = df.sort_values(by=["year"])
df.reset_index(inplace=True)
df.drop("index", axis=1, inplace=True)
df.to_csv("features_table.csv")
