# input file - topic
# output
import sys

import pandas as pd
import xmltodict

df = pd.read_parquet(sys.argv[2])

from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

type = "base"
reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-{type}-med-msmarco")

topic_no = sys.argv[1]
query = Query('aleve migraine')

with open(sys.argv[2]) as f:
    topics = xmltodict.parse(f.read())[0]

topic = filter(lambda x: x["number"], topics).__next__()

texts = [Text(p.passage, {'docid': p.docno}, 0) for p in
         df[df.topic == topic].itertuples()]

reranked = reranker.rerank(query, texts[:1000])
reranked = sorted(reranked, key=lambda x: x.score, reverse=True)

top_passage_per_doc = sorted(list({x.metadata['docid'] : x for x in sorted(reranked, key=lambda i : i.score)}.values()),
                             key=lambda i: i.score, reverse=True)


reranker = DuoT5(model=DuoT5.get_model(f"castorini/duot5-{type}-msmarco"))

reranked = reranker.rerank(query, top_passage_per_doc[:1])
reranked = sorted(reranked, key=lambda x: x.score, reverse=True)

run = [(topic_no, 0, x.metadata["docid"], x.score, i + 1, type) for i, x in enumerate(reranked)]

pd.DataFrame(run)
