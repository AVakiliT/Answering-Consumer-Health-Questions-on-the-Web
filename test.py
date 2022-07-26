from qreldataset.mt5lib import MonoT5, Query, Text
import pandas as pd
from tqdm import tqdm

reranker = MonoT5(pretrained_model_name_or_path=f"castorini/monot5-base-med-msmarco")
for topic in tqdm(list(range(1, 52)) + list(range(101, 201)) + list(range(1001, 1091))):
    df = pd.read_parquet(
        f"data/RunBM25.1k.passages_6_3_t/topic_{topic}.snappy.parquet")
    topics = pd.read_csv("./data/topics_fixed_extended.tsv.txt", sep="\t")
    # df = df.merge(topics["topic description efficacy".split()], on="topic", how="inner")
    df = df.merge(topics["topic efficacy".split()], on="topic", how="inner")

    # df = df[df.topic == topic].merge(topics[topics.topic == topic]["topic description efficacy".split()], on="topic", how="inner")
    query = Query(topics[topics.topic == topic].iloc[0].description)

    texts = [Text(p[1].passage, p[1]) for p in
             df.iterrows()]

    reranked = reranker.rerank(query, texts)

    top_passage_per_doc = {x.metadata["docno"]: (x, x.score) for x in sorted(reranked, key=lambda x: x.score)}

    run = [{"score": x[1], **x[0].metadata.to_dict()} for i, x in enumerate(
        sorted(top_passage_per_doc.values(), key=lambda x: x[1], reverse=True))]

    run_df = pd.DataFrame(run)

    run_df = run_df.sort_values("topic score".split(), ascending=[True, False])