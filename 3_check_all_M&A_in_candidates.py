"""
Yuli Tshuva
This script checks all the M&A companies are in the candidates list every year.
"""

import json
import pandas as pd

# Load the candidates
with open('find_candidates_per_year/candidates.json', "r") as file:
    candidates = json.load(file)

# Load the M&A dataframe
ma = pd.read_csv("analyze_data/M&A_processed.csv", index_col=0)
ma["head"] = ma["head"].astype(str)
ma["tail"] = ma["tail"].astype(str)
# ma.drop(columns=["date"], inplace=True)
# ma.drop_duplicates(subset=["head", "tail"], inplace=True, keep="last")
# ma.reset_index(drop=True, inplace=True)
print(ma.info())
# ma.to_csv('analyze_data/M&A_processed.csv')

heads, tails, years = list(ma["head"]), list(ma["tail"]), list(ma["year"])

u_heads, u_tails, u_years = [heads[0]], [tails[0]], [years[0]]

count = 0
for i in range(1, len(heads)):
    if heads[i] in heads[:i] and tails[i] in tails[:i]:
        count += 1
        print("Duplicate: ", heads[i], tails[i], years[i])
        print("Original: ", heads[i], tails[i], years[heads[:i].index(heads[i])])

print("Manually found", count, "duplicates.")

print("Meanwhile, drop_duplicates detected:",
      ma.shape[0] - ma.drop_duplicates(subset=["head", "tail"]).shape[0], "duplicates.")

#
# for i in range(1, len(heads)):
#     if heads[i] in heads[:i] and tails[i] in tails[:i]:
#         year_of_duplicate = years[i]
#         index_of_first = heads[:i].index(heads[i])
#         year_of_first = years[index_of_first]
#         if year_of_first < year_of_duplicate:
#             u_years[index_of_first] = years[i]
#     else:
#         u_heads.append(heads[i])
#         u_tails.append(tails[i])
#         u_years.append(years[i])
#
# print("len u_heads:", len(u_heads))
# print("len u_tails:", len(u_tails))
# print("len u_years:", len(u_years))
#
# ma = pd.DataFrame({"head": u_heads, "tail": u_tails, "year": u_years})
#
# print(ma.shape)
# print(ma.info())
# print(ma.head())
# ma.to_csv('analyze_data/M&A_processed.csv')

# print(ma.shape[0] - ma.drop_duplicates(subset=["head", "tail", "year"]).shape[0])

# years_range = range(2011, 2021)
# for year in years_range:
#     # Load the candidates for the year
#     candidates_year = candidates[str(year - 1)]
#     companies = list(set(list(ma[ma["year"] == year]["head"])))
#     count = 0
#     missing = []
#     for company in companies:
#         if company in candidates_year:
#             count += 1
#         else:
#             missing.append(company)
#     print("Year: ", year, "Count: ", count, "/", len(companies))
#     print("10 Missing companies:", missing[:10])
