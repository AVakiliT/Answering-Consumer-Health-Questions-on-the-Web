# module load StdEnv gcc cuda/11 faiss arrow/8 python java
import glob
from pathlib import Path
import numpy as np
import pandas as pd
# from pyspark.sql import SparkSession
# from pyspark.sql.types import StructType, IntegerType, StringType, StructField
from tqdm import tqdm

# %%
from utils.util import unfixdocno

files = sorted(list(
    Path('/project/6004803/smucker/group-data/c4-parquet').rglob('*.snappy.parquet')
))

qrels = pd.read_csv("/home/avakilit/resources21/qrels/qrels-35topics.txt",
                    header=None,
                    names="topic iter docno usefulness supportiveness credibility".split(),
                    sep=' ')

qrels = qrels.drop(columns="iter".split())
qrels.docno = qrels.docno.apply(unfixdocno)


def func(file):
    df = pd.read_parquet(file)
    df = df.merge(qrels, on="docno", how="inner")
    return df


from multiprocessing import Pool


def parallelize_dataframe(files, func, n_cores=4):
    with Pool(n_cores) as pool:
        df = pd.concat(pool.imap(func, tqdm(files)))
    return df


df = parallelize_dataframe(files, func)
df = df.reset_index(drop=True)
df.to_parquet('qreldataset/2021-qrels-docs.snappy.parquet')





# %%
import pandas as pd
df = pd.read_parquet('qreldataset/2021-qrels-docs.snappy.parquet')


