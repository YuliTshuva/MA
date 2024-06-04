"""
Yuli Tshuva
Check for missing values in features_table.csv
"""

import pandas as pd
import numpy as np

# Load the data
features = pd.read_csv('features_table.csv')
# Keep two last columns only
features = features[["year", "cik"]]
features["year"] = features["year"].astype(int)
features["cik"] = features["cik"].astype(str)

# Get the M&A data
ma = pd.read_csv('../analyze_data/M&A_processed.csv')
ma.drop("date", axis=1, inplace=True)
ma['year'] = ma["year"].astype(int)
ma["head"], ma["tail"] = ma["head"].astype(str), ma["tail"].astype(str)

for year in range(2012, 2020):
    ma_year = ma[ma['year'] == year]
    companies = set(list(ma_year["head"]) + list(ma_year["tail"]))
    features_year = features[features['year'] == year - 1]
    count = 0
    for company in companies:
        if company in list(features_year["cik"]):
            count += 1

    print(f"Year {year}: {count}/{len(companies)} companies in M&A data are in features data")


