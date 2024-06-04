"""
Yuli Tshuva
Extract the directors of each file.
"""

import pandas as pd
import json
from os.path import join
from os import listdir
import re
from tqdm.auto import tqdm

# Set the data path - Somehow it only works with full path
DATA_PATH = join("..", "directors_data")
COMPANIES_PATH = "companies.json"

# Get the folders in the data
dirs = [folder for folder in listdir(DATA_PATH) if folder.isdigit()]

# load the companies we look for
with open(COMPANIES_PATH, "r") as f:
    companies = json.load(f)

# Get a list of our 21495 companies
companies = companies["companies"]


def signatures(data):
    cik = data['cik']
    data = data['item_15']
    data = data.replace('Director', '\n')
    data = data.replace('President', '\n')
    data = data.replace('Chairman', '\n')
    data = data.replace('Vice', '\n')
    data = data.replace('Senior', '\n')
    data = data.replace('Manager', '\n')
    data = data.replace('Chief', '\n')
    data = data.replace('Chife', '\n')
    data = data.replace('Corporate', '\n')
    data = data.replace('Executive', '\n')
    data = data.replace('Controller', '\n')
    data = data.replace('Chair', '\n')
    data = data.replace('Independent', '\n')
    data = data.replace('Co-', '\n')
    data = data.replace('Trustee', '\n')
    data = data.replace('Principal', '\n')
    data = data.replace('Director', '\n')
    data = data.replace('Corresponding', '\n')
    data = data.replace('Recording', '\n')
    data = data.replace('Member', '\n')
    data = data.replace('Trustee', '\n')
    data = data.replace('Trustee', '\n')
    data = data.replace('Trustee', '\n')
    data = data.replace('Trustee', '\n')
    data = data.replace('Trustee', '\n')
    data = data.replace('Trustee', '\n')
    data = data.replace('Trustee', '\n')
    data = data.replace('Date', '\n')
    data = data.replace('March', '\n')
    data = data.replace('February', '\n')
    data = data.replace('Date', '\n')
    data = data.replace('Date', '\n')
    data = data.replace('Date', '\n')
    data = data.replace('Date', '\n')
    data = data.replace('Date', '\n')
    data = data.replace('Date', '\n')

    # Replace digits with empty line
    data = re.sub('\d', '\n', data)

    # Reset the list to be
    lst = (re.findall('/[sS]/(.+?)\n', data))
    lst = [l.strip(' ,_* (') for l in lst]
    lst = [l.title() for l in lst]

    # Filtering elements from the list
    lst = [l for l in lst if not 'Llp' in l]
    lst = [l for l in lst if not 'Llc' in l]
    lst = [l for l in lst if not 'S.A.' in l]
    lst = [l for l in lst if not 'Cpa' in l]
    lst = [l for l in lst if not 'P.C.' in l]
    lst = [l for l in lst if not 'Llp' in l]
    lst = [l for l in lst if not 'L.L.P' in l]
    lst = [l for l in lst if not 'L.L.P.' in l]
    lst = [l for l in lst if not 'L.L.P' in l]
    lst = [l for l in lst if not 'L.L.P' in l]
    lst = [l for l in lst if not '&' in l]
    lst = [l for l in lst if not ', Pc' in l]

    # Removing adjectives from the name of people included in the list
    lst = [l.replace('Ph.D.', '') for l in lst]
    lst = [l.replace('M.D.', '') for l in lst]
    lst = [l.replace('Dr.', '') for l in lst]
    lst = [l.replace('M.P.H.', '') for l in lst]
    lst = [l.replace('Ch.B.', '') for l in lst]
    lst = [l.replace('M.D', '') for l in lst]
    lst = [l.replace('J.D.', '') for l in lst]
    lst = [l.replace('Md.', '') for l in lst]
    lst = [l.replace('M.B.', '') for l in lst]
    lst = [l.replace('F.A.C.P.', '') for l in lst]
    lst = [l.replace('Pharm. D.', '') for l in lst]
    lst = [l.replace('M.R.C.P.', '') for l in lst]
    lst = [l.replace('Mph', '') for l in lst]
    lst = [l.replace('Pharm. D..', '') for l in lst]
    lst = [l.replace('Pharm. D..', '') for l in lst]
    lst = [l.replace('Pharm. D..', '') for l in lst]

    lst = [l.split('/S/') for l in lst]
    lst = [l[0] for l in lst]

    df = pd.DataFrame({"director": list(set(lst))})
    df['cik'] = cik
    return df

# Set a dataframe for all the directors
directors_df = []

# Iterate through the directories
for dir in tqdm(dirs):
    # Get the directory path
    dir_name = join(DATA_PATH, str(dir), str(dir))
    # Get the files in the directory
    dir_files = listdir(dir_name)
    # Set a dictionary: cik it represents -> file's name
    cik_to_dir = {file_name.split("_")[0]: file_name for file_name in dir_files}
    # Get the ciks in the directory as list
    ciks_in_dir = list(cik_to_dir.keys())
    # Narrow it to the relevant only ciks
    ciks_in_dir = list(set(companies) & set(ciks_in_dir))
    # Narrow the files according to the ciks
    dir_files = [cik_to_dir[cik] for cik in ciks_in_dir]
    # Set a directors dataframe for current year
    year_directors_df = []
    for file in dir_files:
        with open(join(dir_name, file), "r") as f:
            report = json.load(f)
        report_directors_df = signatures(report)
        year_directors_df.append(report_directors_df)
    # Concat all the dataframes into one dataframe
    year_directors_df = pd.concat(year_directors_df)
    # Add a year column
    year_directors_df["year"] = int(dir)
    # Add the dataframe to the total one
    directors_df.append(year_directors_df)

# At last concat all years dataframe
directors_df = pd.concat(directors_df)

# Save the dataframe
directors_df.to_csv("directors.csv")
