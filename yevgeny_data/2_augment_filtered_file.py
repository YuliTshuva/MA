"""
Yuli Tshuva
Augment the filtered file
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the filtered file
df = pd.read_csv('filtered_data.csv')

# Obviously we don't need rows where we miss cik
df.dropna(subset=['cik'], inplace=True)

# Drop data ahead of our timeline
df = df[df['fyear'] <= 2019]

# Drop duplicates
print(f'Dropping {df.duplicated(subset=["cik", "fyear"]).sum()} duplicates')
df.drop_duplicates(subset=['cik', 'fyear'], inplace=True)

# Now we want to view the vary of years in the dataset
counts = df['fyear'].value_counts()
counts = counts.sort_index()

# Plot
plt.plot(counts.index, counts.values, color='dodgerblue', linewidth=2)
plt.title("Amount of companies per year")
plt.xlabel("Year")
plt.ylabel("Amount of companies")

# Add mean to plot
mean = counts.iloc[-10:].mean()
plt.plot(counts.index[-10:], [mean] * 10, color="hotpink",
         linestyle="--", label=f"mean {mean:.2f}", linewidth=3)
plt.legend()
plt.show()

# Save the augmented file
df.to_csv('augmented_filtered_data.csv', index=False)

# Get the industries distribution
industries = df[df["fyear"] >= 2010]["industry"].value_counts(sort=False)

# Plot a bar plot
plt.figure(figsize=(10, 7))
plt.bar(industries.index, industries.values, color='turquoise')
plt.title("Industries distribution")
plt.xlabel("Industry")
plt.ylabel("Amount of companies")
# Rotate the x labels by 45 degrees
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
