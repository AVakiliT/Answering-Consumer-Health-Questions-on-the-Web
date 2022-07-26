# %%
import re
import sys

import pandas as pd
import spacy

# df = spark.read.load("/project/6004803/avakilit/Trec21_Data/data/qrel_2021")
from tqdm import tqdm

# df = pd.concat([
#     pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25_2019"),
#     pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25"),
#     pd.read_parquet("/project/6004803/avakilit/Trec21_Data/Top1kRWBM25_32p")
# ])
from qreldataset.mt5lib import MonoT5

df = pd.read_parquet("data/Top1kBM25.snappy.parquet")

window_size, step = 6, 3

# print(df.count())
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 0
k = 10000

def func(df):
    return df.text.apply(sentencize)

def func2(df):
    return df.passage.apply(lambda x: pd.Series(dict(passage_index=x[0], passage=x[1])))

from multiprocessing import Pool
def get_chunks(df, size):
    for i in range(0, len(df), size):
        yield df.iloc[i:min(i+size, len(df))]

def parallelize_dataframe(df, func, n_cores=28):
    with Pool(n_cores) as pool:
        df_new = pd.concat(pool.imap(func, tqdm(get_chunks(df, k), total=df.shape[0]/k)))
    return df_new

def sentencize(s):
    s = re.sub('\s+', " ", s.strip())
    sentences = [sent.sent.text.strip() for sent in nlp(s).sents if len(sent) > 3]
    if len(sentences) <= window_size:
        return [(0, s)]
    return [(i, ' '.join(sentences[i: i + window_size])) for i in range(0, len(sentences), step)]


def wordize(s):
    s = re.sub('\s+', " ", s.strip())
    doc = nlp(s)
    sentences = [sent.sent.text.strip() for sent in doc.sents if len(sent) > 5]
    tokens = nlp(' '.join(sentences))

    if len(tokens) <= window_size:
        return [tokens.text.strip()]
    return [tokens[i: i + window_size].text.strip() for i in range(0, len(tokens), step)]

tqdm.pandas()
# x = df[n * k: n * k + k].reset_index()
# x["passage"] = x.text.progress_apply(sentencize)
temp = parallelize_dataframe(df, func, 24)
df["passage"] = temp


df = df.explode("passage")
# df[['docno', 'timestamp', 'url', 'topic', 'score', 'passage']].to_parquet(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}.snappy.parquet")
temp2 = parallelize_dataframe(df, func2)
df = df.reset_index(drop=True)
df = df.drop(columns="passage")
df_new = pd.concat([df, temp2.reset_index(drop=True)], axis=1)
for n, x in enumerate(get_chunks(df_new, 10000)):
    x.to_parquet(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}/{n}.snappy.parquet")
df_new = pd.read_parquet(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}")

# df_new = df_new.selectExpr("topic,docno,timestamp,url,usefulness,stance,credibility,explode(passage) as passage".split(','))
# df_new = df_new.selectExpr('topic,docno,timestamp,url,usefulness,stance,credibility,passage["passage_index"] as passage_index,passage["passage"] as passage'.split(','))
# df_new.repartition(1).write.save(f"/project/6004803/avakilit/Trec21_Data/data/qrels.2021.passages_{window_size}_{step}", mode="overwrite")

# df_new = df_new.selectExpr('topic,docno,timestamp,url,score as bm25,explode(passage) as passage'.split(','))
# df_new = df_new.selectExpr(
#     'topic,docno,timestamp,url,bm25,passage["passage_index"] as passage_index,passage["passage"] as passage'.split(','))
# df_new.repartition(1).write.save(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}",
#                                  mode="overwrite")

#%%
reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-base-med-msmarco")

