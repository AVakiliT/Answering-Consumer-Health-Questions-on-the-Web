import glob
import sys
import pandas as pd
from tqdm import trange

files = sorted(glob.glob("/project/6004803/avakilit/c4_parquet/*.parquet"))
n = int(sys.argv[2])
start = int(sys.argv[1])

docnos = pd.read_csv("/project/6004803/smucker/group-data/runs/trec2021-misinfo/automatic/run.c4.noclean.bm25.topics.2021.10K.fixed_docno.txt",
                     names="topic iter docno score ranks tag".split(), index_col="docno", sep=" ")

docnos = docnos[["docno", "topic", "score"]]

for i in trange(start * n, (start+1) * n):
    try:
        df = pd.read_parquet(files[i]).set_index("docno")
    except IndexError as _:
        continue
    collection_m = df.join(docnos, "docno", "inner")
    collection_m.to_parquet("Top1kBM25/" + files[i].split("/")[-1], index="docno")