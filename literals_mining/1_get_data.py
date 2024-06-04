"""
Yuli Tshuva
Running the code for mining data from internet about companies' literals
"""

import requests
import json
from os.path import join
import threading
from threading import Lock

# Create a lock
lock = Lock()
lock2 = Lock()
companies_features = {}
count = 0
not_found = 0

# Requests headers
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0",
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,/;q=0.8"}

# A list of possible facts
FACTS = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'OperatingIncomeLoss',
         'ProfitLoss', 'NetIncomeLoss', 'RevenuesNetOfInterestExpense', 'BenefitsLossesAndExpenses',
         'Assets', 'Liabilities', 'StockholdersEquity', 'EntityCommonStockSharesOutstanding',
         'CommonStockSharesOutstanding']

# Get companies list
with open("companies.json", 'r') as json_file:
    COMPANIES = json.load(json_file)["companies"]


def get_literals_dict(cik):
    """
    Receives a company's cik and return its literals' dict.
    """
    cik = str(cik)
    cik = cik.zfill(10)
    try:
        response = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=headers)
        content = json.loads(response.content)
    except:
        content = None
    return content


def trace_attribute(fact, dct, path=""):
    """
    Recursive function that:
    Get a fact and trace backwards to its source (key1 -> key2 -> ... -> fact).
    """
    if type(dct) != dict:
        return ""
    if fact in dct:
        return join(path, fact)
    return "".join([trace_attribute(fact, dct[key], join(path, key)) for key in dct])


def find_companies_features(company):
    """Get a company and extract its features"""
    literals = get_literals_dict(company)

    if not literals:
        return None

    # Locate where the interesting part of the dictionary is
    fact_path = trace_attribute("NetIncomeLoss", literals)
    fact_keys = fact_path.split("\\")[:-1]

    # Get the dictionary to the interesting part
    for key in fact_keys:
        literals = literals[key]

    # Get a dictionary of only the companies we need
    facts_dict = {key: value for key, value in literals.items() if key in FACTS}
    facts_dict = {key: list(value["units"].values())[0] for key, value in facts_dict.items()}

    # Keep only the 10k forms
    facts_dict = {key: [dct for dct in value if dct["form"] == "10-K"] for key, value in facts_dict.items()}

    # Transform into format of [{year: value}]
    facts_dict = {key: [{int(dct['end'][:4]): dct['val']} for dct in value] for key, value in facts_dict.items()}

    return facts_dict


def threading_split(companies, i):
    global companies_features, count
    for company in companies:
        company_feats = find_companies_features(company)
        if company_feats:
            with lock:
                companies_features[company] = company_feats
                count += 1
            print(count)
        else:
            with lock2:
                global not_found
                not_found += 1
            # if not_found <= 10:
            if True:
                print("company not found:", company)


def multi_threading_process(splits):
    batch_size = len(COMPANIES) // splits
    print("batch_size:", batch_size)
    threads = []
    for i in range(splits):
        companies_part = COMPANIES[i * batch_size: min((i + 1) * batch_size, len(COMPANIES))]
        thread = threading.Thread(target=threading_split, args=(companies_part, i))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    with open(join("companies_features.json"), "w") as file:
        json.dump(companies_features, file)


def main():
    multi_threading_process(3)


if __name__ == '__main__':
    main()
