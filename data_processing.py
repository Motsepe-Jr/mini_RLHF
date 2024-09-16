import pandas as pd

df = pd.read_csv("data/imdb-dataset.csv")

imdb_str = " <|endoftext|> ".join(df['review'].tolist())

with open ('data/imdb.txt', 'w', encoding='utf-8') as f:
    f.write(imdb_str)