# input file - topic
# output
import os

from torch.cuda import amp

print("Importing...", flush=True)
import argparse
from random import sample
from timeit import default_timer as timer

import pandas as pd
import xmltodict
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5, DuoT5



print("Parsing args...", flush=True)
parser = argparse.ArgumentParser()
parser.add_argument("--topic_no", default=101, type=int)
parser.add_argument("--topic_file", default="/project/6004803/smucker/group-data/topics/misinfo-2021-topics.xml")
parser.add_argument("--model_type", default="base")
parser.add_argument("--bm25run",
                    default="/project/6004803/avakilit/Trec21_Data/Top1kBM25_1p_passages/part-00000-0da9fef6-fd3a-48a8-96d8-f05f4d9e9da2-c000.snappy.parquet")

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--duo', dest='duo', action='store_true')
feature_parser.add_argument('--no-duo', dest='duo', action='store_false')
parser.set_defaults(duo=False)

args = parser.parse_known_args()

type = args[0].model_type
topic_no = args[0].topic_no
topic_file = args[0].topic_file
print("Reading Passages Dataframe...", flush=True)
df = pd.read_parquet(args[0].bm25run)
duo = args[0].duo
print("Done.", flush=True)

output_dir = f"output_m{'d' if duo else ''}t5_{2021 if '2021' in topic_file else '2019'}_{type}"
try:
    os.mkdir(output_dir)
except FileExistsError:
    pass

print("Loading topic file...", flush=True)
with open(topic_file) as f:
    topics = xmltodict.parse(f.read())['topics']['topic']

topic = filter(lambda x: x["number"] == str(topic_no), topics).__next__()
query = Query(topic["query"])

print("Topic query is:", flush=True)
print(topic["query"], flush=True)



print(f"Loading MonoT5 ...", flush=True)
reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-{type}-msmarco")
print("Done.", flush=True)

texts = [Text(p.passage, {'docid': p.docno}, 0) for p in
         df[df.topic == topic_no].itertuples()]
# texts = sample(texts, 1000)
print("Reranking with MonoT5...", flush=True)
start = timer()
with amp.autocast():
    reranked = reranker.rerank(query, texts)
end = timer()
print(f"Done. reraking {len(texts)} passages with monot5-{type}-med-msmarco took {end - start} seconds.", flush=True)

top_passage_per_doc = sorted(list(
    {x.metadata['docid']: x for x in sorted(reranked, key=lambda i: i.score)}
        .values()),
                             key=lambda i: i.score, reverse=True)

# del reranked


if duo:
    print("Loading DuoT5...", flush=True)
    reranker = DuoT5(model=DuoT5.get_model(f"castorini/duot5-{type}-msmarco"))
    print("Done.", flush=True)
    print("Reranking with DuoT5...", flush=True)
    start = timer()
    top_passage_per_doc = reranker.rerank(query, top_passage_per_doc)
    end = timer()
    print(f"Done. Reranking {len(top_passage_per_doc)} with duot5-{type}-msmarco took {end - start} seconds.",
          flush=True)

run = [(topic_no, 0, x.metadata["docid"], i + 1, x.score, type) for i, x in enumerate(top_passage_per_doc)]
run_df = pd.DataFrame(run)

for i in top_passage_per_doc:
    print(i.metadata['docid'], i.score, i.text[:200])

print("Writing Run file...", flush=True)

run_df.to_csv(f"{output_dir}/topic-{topic_no}.run", sep=" ",
              index=False, header=False)
print("Done.", flush=True)
