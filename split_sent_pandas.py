# %%
import re
import sys

import pandas as pd
import spacy

# df = spark.read.load("/project/6004803/avakilit/Trec21_Data/data/qrel_2021")
from tqdm import tqdm

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
# for n, x in enumerate(get_chunks(df_new, 10000)):
#     x.to_parquet(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}/{n}.snappy.parquet")
#
# def func3():
#     for t in tqdm(df_new.topic.unique()):
#         yield t, df_new[df_new.topic.eq(t)]

# def func4(stuff):
#     t, x = stuff
#     x.to_parquet(
#         f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}_t/topic_{t}.snappy.parquet")
# with Pool(24) as p:
#     p.map(func4, func3())

for t in tqdm(list(range(1, 52)) + list(range(101, 201)) + list(range(1001, 1091))):
    df_new[df_new.topic.eq(t)].to_parquet(
        f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_6_3_t/topic_{t}.snappy.parquet")

# df_new = df_new.selectExpr("topic,docno,timestamp,url,usefulness,stance,credibility,explode(passage) as passage".split(','))
# df_new = df_new.selectExpr('topic,docno,timestamp,url,usefulness,stance,credibility,passage["passage_index"] as passage_index,passage["passage"] as passage'.split(','))
# df_new.repartition(1).write.save(f"/project/6004803/avakilit/Trec21_Data/data/qrels.2021.passages_{window_size}_{step}", mode="overwrite")

# df_new = df_new.selectExpr('topic,docno,timestamp,url,score as bm25,explode(passage) as passage'.split(','))
# df_new = df_new.selectExpr(
#     'topic,docno,timestamp,url,bm25,passage["passage_index"] as passage_index,passage["passage"] as passage'.split(','))
# df_new.repartition(1).write.save(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}",
#                                  mode="overwrite")

# %%
import torch
from qreldataset.mt5lib import MonoT5, Query, Text
import pandas as pd
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp

def run_inference(rank, world_size, topic):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-base-med-msmarco")
    reranker.model.eval()
    reranker.model.to(rank)
    df = pd.read_parquet(
        f"data/RunBM25.1k.passages_6_3_t/topic_{topic}.snappy.parquet")
    topics = pd.read_csv("./data/topics_fixed_extended.tsv.txt", sep="\t")
    # df = df.merge(topics["topic description efficacy".split()], on="topic", how="inner")
    df = df.merge(topics["topic efficacy".split()], on="topic", how="inner")

    # df = df[df.topic == topic].merge(topics[topics.topic == topic]["topic description efficacy".split()], on="topic", how="inner")
    query = Query(topics[topics.topic == topic].iloc[0].description)

    texts = [Text(p[1].passage, p[1]) for p in
             df.iterrows()]
    with torch.no_grad():
        reranked = reranker.rerank(query, texts)
    top_passage_per_doc = {x.metadata["docno"]: (x, x.score) for x in sorted(reranked, key=lambda x: x.score)}

    run = [{"score": x[1], **x[0].metadata.to_dict()} for i, x in enumerate(
        sorted(top_passage_per_doc.values(), key=lambda x: x[1], reverse=True))]

    run_df = pd.DataFrame(run)

    run_df = run_df.sort_values("topic score".split(), ascending=[True, False])
    run_df.to_parquet(f"data/RunBM25.1k.passages_mt5.top_mt5/{topic}.snappy.parquet")


topics = list(range(1, 52)) + list(range(101, 201)) + list(range(1001, 1091))

pbar = tqdm(topics)
for topic in pbar:
    world_size = 4
    mp.spawn(run_inference,
             args=(world_size,topic),
             nprocs=world_size,
             join=True)




#%%
import pandas as pd
from utils.util import fixdocno
dfx = pd.read_parquet(f"data/RunBM25.1k.passages_mt5.top_mt5").sort_values("topic score".split(), ascending=[True, False])
dfx["ranking"] = list(range(1,1001)) * dfx.topic.nunique()
run = dfx.apply(lambda x: f"{x.topic} Q0 {fixdocno(x.docno)} {x.ranking} {x.score} WatS-MT5-MT5", axis=1)
run.to_csv("runs/WatS-MT5-MT5.all", index=False, header=False)
