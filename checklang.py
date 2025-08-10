import pandas as pd

df = pd.read_csv('data/indian movies.csv')
movie_name = 'Avakai Biryani'
exists = df['Movie Name'].str.lower().eq(movie_name.lower()).any()

if exists:
    print(f"The movie '{movie_name}' exists in the CSV.")
else:
    print(f"The movie '{movie_name}' does NOT exist in the CSV.")
