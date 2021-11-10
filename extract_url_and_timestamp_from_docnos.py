import glob
import sys
import pandas as pd


files = sorted(glob.glob("/project/6004803/avakilit/c4_parquet/*.parquet"))
i = int(sys.argv[1])
df = pd.read_parquet(files[i]).set_index("docno")

docnos = pd.read_csv("/project/6004803/smucker/group-data/misc/HON_Docnos.csv/part-00000-9b3df246-3321-48cc-90e2"
                     "-f0683a9c425b-c000.csv", names=["docno"], index_col="docno")
collection_m = df.join(docnos, "docno", "inner")
collection_m.to_parquet("Collection_M/" + files[i].split("/")[-1], index="docno")