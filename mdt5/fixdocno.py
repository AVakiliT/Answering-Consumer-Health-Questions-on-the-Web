import os

import pandas as pd
directory = "output_Top1kEBM25_mt5_2021_base-med"
df = pd.concat([pd.read_csv(f"{directory}/{f}", header=None, sep=" ") for f in sorted(os.listdir(directory))])
df[2] = df[2].map(lambda x: f"en.noclean.c4-train.0{x[3:7]}-of-07168.{int(x[8:])}")
df.to_csv(f"run-{directory}.txt", header=None, index=False, sep=" ")