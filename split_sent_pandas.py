# %%
import re
import sys

import pandas as pd
import spacy

# df = spark.read.load("/project/6004803/avakilit/Trec21_Data/data/qrel_2021")
from tqdm import tqdm

df = pd.concat([
    pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25_2019"),
    pd.read_parquet("/project/6004803/avakilit/Trec21_Data/data/Top1kBM25"),
    pd.read_parquet("/project/6004803/avakilit/Trec21_Data/Top1kRWBM25_32p")
])

window_size, step = 1200, 600

# print(df.count())
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 0
k = 10000


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
x = df[n * k: n * k + k].reset_index()
x["passage"] = x.text.progress_apply(wordize)
x = x.explode("passage")
x[['docno', 'timestamp', 'url', 'topic', 'score', 'passage'].to_parquet(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}/{n}.snappy.parquet")
# df_new = df_new.selectExpr("topic,docno,timestamp,url,usefulness,stance,credibility,explode(passage) as passage".split(','))
# df_new = df_new.selectExpr('topic,docno,timestamp,url,usefulness,stance,credibility,passage["passage_index"] as passage_index,passage["passage"] as passage'.split(','))
# df_new.repartition(1).write.save(f"/project/6004803/avakilit/Trec21_Data/data/qrels.2021.passages_{window_size}_{step}", mode="overwrite")

# df_new = df_new.selectExpr('topic,docno,timestamp,url,score as bm25,explode(passage) as passage'.split(','))
# df_new = df_new.selectExpr(
#     'topic,docno,timestamp,url,bm25,passage["passage_index"] as passage_index,passage["passage"] as passage'.split(','))
# df_new.repartition(1).write.save(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}",
#                                  mode="overwrite")
