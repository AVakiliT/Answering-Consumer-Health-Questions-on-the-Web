# input file - topic
# output
import argparse
from timeit import default_timer as timer

import pandas as pd
import xmltodict
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5, DuoT5

parser = argparse.ArgumentParser()
parser.add_argument("--topic_no", default=122, required=False)
parser.add_argument("--topic_file", default="/project/6004803/smucker/group-data/topics/misinfo-2021-topics.xml",
                    required=False)
parser.add_argument("--model_type", default="base", required=False)
parser.add_argument("--bm25run",
                    default="/project/6004803/avakilit/Trec21_Data/Top1kbm25_1p_passages/part-00000-2bef8f95-53dc-49f9-8b45-31f5deaf0be1-c000.snappy.parquet",
                    required=False)
args = parser.parse_known_args()

type = args[0].model_type
topic_no = args[0].topic_no
topic_file = args[0].topic_file
df = pd.read_parquet(args[0].bm25run)

reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-{type}-med-msmarco")

with open(topic_file) as f:
    topics = xmltodict.parse(f.read())['topics']['topic']

topic = filter(lambda x: x["number"] == str(topic_no), topics).__next__()
query = Query(topic["query"])

texts = [Text(p.passage, {'docid': p.docno}, 0) for p in
         df[df.topic == topic_no].itertuples()]

start = timer()
reranked = reranker.rerank(query, texts)
end = timer()
print(f"monot5-{type}-med-msmarco took {end-start} seconds.")
reranked = sorted(reranked, key=lambda x: x.score, reverse=True)

top_passage_per_doc = sorted(list({x.metadata['docid']: x for x in sorted(reranked, key=lambda i: i.score)}.values()),
                             key=lambda i: i.score, reverse=True)

del reranked
reranker = DuoT5(model=DuoT5.get_model(f"castorini/duot5-{type}-msmarco"))


start = timer()
reranked2 = reranker.rerank(query, top_passage_per_doc)
end = timer()
print(f"duot5-{type}-msmarco took {end-start} seconds.")
reranked2 = sorted(reranked2, key=lambda x: x.score, reverse=True)
run = [(topic_no, 0, x.metadata["docid"], x.score, i + 1, type) for i, x in enumerate(reranked2)]

pd.DataFrame(run)
