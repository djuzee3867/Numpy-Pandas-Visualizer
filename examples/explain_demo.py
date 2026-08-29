import numpy as np
import pandas as pd

a = np.arange(12).reshape(3, 4)
block = a[0:2, 1:3]
flipped = a.T
totals = a.sum(axis=0)
b = np.arange(3).reshape(3, 1)
grid = np.arange(3) + b

df = pd.DataFrame({"team": ["a", "b", "a"], "score": [10, 7, 13]})
by_team = df.groupby("team")["score"].sum()

right = pd.DataFrame({"team": ["a", "c"], "city": ["x", "y"]})
joined = df.merge(right, on="team", how="left")
