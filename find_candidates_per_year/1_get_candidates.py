"""
Yuli Tshuva
Iterate through the 10-k reports submitted at each year.
Take each company which submitted a report as a candidate for M&A transaction.
"""

import os
from os.path import join
import json
from tqdm.auto import tqdm

# Path to 10-k reports
REPORTS_PATH = join('..', 'directors_data')


def main():
    candidates_per_year = {}
    for year in tqdm(range(2010, 2020+1), desc="years:"):
        candidates = []
        # For year = year
        directory_path = join(REPORTS_PATH, str(year), str(year))
        for file_name in os.listdir(directory_path):
            comapny_cik = file_name.split('_')[0]
            candidates.append(comapny_cik)
        # For year = year-1
        directory_path = join(REPORTS_PATH, str(year-1), str(year-1))
        for file_name in os.listdir(directory_path):
            comapny_cik = file_name.split('_')[0]
            candidates.append(comapny_cik)

        # Add to the dictionary
        candidates_per_year[int(year)] = list(set(candidates))  # Remove duplicates

    with open("candidates.json", "w") as file:
        json.dump(candidates_per_year, file)


if __name__ == "__main__":
    main()
