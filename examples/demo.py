import numpy as np
import pandas as pd

a = np.arange(6).reshape(2, 3)
b = a[:, 1:]          # b is a view into a
b += 10               # writing to b writes through to a
total = a.sum(axis=0)
print("total:", total)

df = pd.DataFrame({"k": ["x", "y", "x"], "v": [1.0, float("nan"), 3.0]})
g = df.groupby("k")["v"].sum()
