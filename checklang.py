import pandas as pd

# Load your dataset
df = pd.read_csv("data/indian movies.csv")  # replace with your correct file name

# See available columns
exclude_languages = ['urdu', 'nepali', 'sanskrit']

# Filter dataset
df = df[~df['Language'].isin(exclude_languages)]

# Save updated CSV
df.to_csv("indian_movies_filtered.csv", index=False)

print("Filtered dataset saved as 'indian_movies_filtered.csv'")
print("Remaining languages:", df['Language'].unique())
