import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("hf://datasets/sapientinc/sudoku-extreme/train.csv", nrows=10000)

df = df.rename(columns={"answer" : "solution", "question" : "puzzle"}) 
df["num_empty"] = df["puzzle"].apply(lambda puzzle: puzzle.count("."))
df["num_clues"] = 81 - df["num_empty"]
df["log_rating"] = np.log1p(df["rating"])
df = df.drop(columns="source")

print(df.head())
print(df.describe())
print(df["log_rating"].skew())

df["rating"].hist(bins=50)

plt.xlabel("Difficulty Rating")
plt.ylabel("Frequency")
plt.title("Distribution of Sudoku Difficulty Ratings")

plt.savefig("figures/rating_distribution.png")
