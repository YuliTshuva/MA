"""
Yuli Tshuva
Drop the index column for the connection components
"""

import os
from os.path import join
import pandas as pd

DATA_PATH = join("..", "connection_components")

# Load the data
for file in os.listdir(DATA_PATH):
    path = join(DATA_PATH, file)
    df = pd.read_csv(path, index_col=0)
    df.to_csv(path, index=False)

print("Expirement:")
df = pd.read_csv(join(DATA_PATH, "connections_components_2019.csv"), index_col=0)

companies_to_search = [906553, 1769804, 744187]

print(df.loc[companies_to_search]["size"])


