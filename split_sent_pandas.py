# %%
import re
import sys

import pandas as pd
import spacy

# df = spark.read.load("/project/6004803/avakilit/Trec21_Data/data/qrel_2021")
from tqdm import tqdm

# df = pd.read_parquet("data/Top1kBM25.snappy.parquet")
df = pd.read_parquet('qreldataset/2021-qrels-docs')

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
        yield df.iloc[i:min(i + size, len(df))]


def parallelize_dataframe(df, func, n_cores=28):
    with Pool(n_cores) as pool:
        df_new = pd.concat(pool.imap(func, tqdm(get_chunks(df, k), total=df.shape[0] / k)))
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
df = df.drop(columns="passage text".split())
df = df.rename(columns={"score": "bm25"})
df_new = pd.concat([df, temp2.reset_index(drop=True)], axis=1)



# for t in tqdm(list(range(1, 52)) + list(range(101, 201)) + list(range(1001, 1091))):
#     df_new[df_new.topic.eq(t)].to_parquet(
#         f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_6_3_t/topic_{t}.snappy.parquet")
for t in df_new.topic.unique():
    df_new[df_new.topic.eq(t)].to_parquet(
        f"qreldataset/Qrels.2021.passages_6_3_t/topic_{t}.snappy.parquet")



# %%

#mt5-mt5
#%%
import pandas as pd
from utils.util import fixdocno
dfx = pd.read_parquet(f"data/RunBM25.1k.passages_mt5.top_mt5").sort_values("topic score".split(), ascending=[True, False])
dfx = pd.read_parquet(f"data/RunBM25.1k.passages_mt5.top_mt5").sort_values("topic score".split(), ascending=[True, False])
dfx["ranking"] = list(range(1,1001)) * dfx.topic.nunique()
run = dfx.apply(lambda x: f"{x.topic} Q0 {fixdocno(x.docno)} {x.ranking} {x.score} WatS-MT5-MT5", axis=1)
run.to_csv("runs/WatS-MT5-MT5.all", index=False, header=False)
