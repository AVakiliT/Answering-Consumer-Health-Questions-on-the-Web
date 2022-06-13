import glob
import sys
import pandas as pd
from tqdm import trange

files = sorted(glob.glob("/project/6004803/smucker/group-data/c4-parquet/*.parquet"))
n = int(sys.argv[2])
start = int(sys.argv[1])
output_dir = sys.argv[3]
print(output_dir)
docnos = pd.read_csv(sys.argv[4],
                      names="topic iter docno ranks score tag".split(), index_col="docno", sep=" ")

docnos =docnos[docnos.ranks <= 1000][["topic", "score"]]
print(docnos.count())

for i in trange(start * n, (start+1) * n):
    try:
        df = pd.read_parquet(files[i]).set_index("docno")
        print(df.count())
    except IndexError as _:
        continue
    collection_m = df.join(docnos, "docno", "inner")
    print(collection_m.count())
    collection_m.to_parquet(output_dir + "_32p/" + files[i].split("/")[-1], index="docno")