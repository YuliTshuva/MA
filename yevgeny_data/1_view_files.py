"""
Yuli Tshuva
Viewing data files
"""

import pandas as pd

# Read the dataset
df = pd.read_stata("khfmujf6loclc3gq.dta")

# Keep only necessary columns
df = df[["cik", "dltt", "at", "fyear", "sic"]]

# Normalize dltt
df['dltt'] = df['dltt'] / df['at']

# Define a list of all industries
industries = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]


def get_industry(sic):
    """Get a sic as string or int and return its industry as string."""
    try:
        # Check data type
        sic = int(sic)
    except:
        return "Unknown"

    # Define industry
    if 100 <= sic <= 999:
        return "NoDur"
    elif 2000 <= sic <= 2399 or 2700 <= sic <= 2749 or 2770 <= sic <= 2799 or 3100 <= sic <= 3199 or 3940 <= sic <= 3989:
        return "NoDur"
    elif 2500 <= sic <= 2519 or 2590 <= sic <= 2599 or 3630 <= sic <= 3659 or 3710 <= sic <= 3711 or 3714 <= sic <= 3714 or 3716 <= sic <= 3716 or 3750 <= sic <= 3751 or 3792 <= sic <= 3792 or 3900 <= sic <= 3939 or 3990 <= sic <= 3999:
        return "Durbl"
    elif 2520 <= sic <= 2589 or 2600 <= sic <= 2699 or 2750 <= sic <= 2769 or 3000 <= sic <= 3099 or 3200 <= sic <= 3569 or 3580 <= sic <= 3629 or 3700 <= sic <= 3709 or 3712 <= sic <= 3713 or 3715 <= sic <= 3715 or 3717 <= sic <= 3749 or 3752 <= sic <= 3791 or 3793 <= sic <= 3799 or 3830 <= sic <= 3839 or 3860 <= sic <= 3899:
        return "Manuf"
    elif 1200 <= sic <= 1399 or 2900 <= sic <= 2999:
        return "Enrgy"
    elif 2800 <= sic <= 2829 or 2840 <= sic <= 2899:
        return "Chems"
    elif 3570 <= sic <= 3579 or 3660 <= sic <= 3692 or 3694 <= sic <= 3699 or 3810 <= sic <= 3829 or 7370 <= sic <= 7379:
        return "BusEq"
    elif 4800 <= sic <= 4899:
        return "Telcm"
    elif 4900 <= sic <= 4949:
        return "Utils"
    elif 5000 <= sic <= 5999 or 7200 <= sic <= 7299 or 7600 <= sic <= 7699:
        return "Shops"
    elif 2830 <= sic <= 2839 or 3693 <= sic <= 3693 or 3840 <= sic <= 3859 or 8000 <= sic <= 8099:
        return "Hlth"
    elif 6000 <= sic <= 6999:
        return "Money"
    else:
        return "Other"


# Add industry to the dataset
df['industry'] = df['sic'].apply(get_industry)

# Drop unnecessary column
df.drop("sic", inplace=True, axis=1)

# Save the dataset
df.to_csv("filtered_data.csv", index=False)
