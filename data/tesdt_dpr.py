import torch
from tqdm import trange
import pandas as pd

topics = pd.read_csv("data/topics.csv", index_col="topic", sep="\t")
year = 2021
from datasets import load_dataset
ds = load_dataset("parquet",
                  data_files={
                      "train": f"data/Top1kBM25_{'' if year == 2021 else '2019'}1p_passages/*.parquet"
                  },
                  split='train')\
    # .filter(lambda e: e['topic'] == 101)
torch.set_grad_enabled(False)
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
df1 = pd.read_parquet("data/Top1kBM25_1p_passages")
ds.load_faiss_index("epoch_0", "data/faiss_index/2021/epoch_0")

#%%
# runs = []
# for topic in trange(101,151):
#     temp_ds = ds
#     # temp_ds.load_faiss_index("epoch_0", "data/faiss_index/2021/epoch_0")
#     question = topics.loc[topic].description
#     question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt"))[0][0]
#     s, r = temp_ds.get_nearest_examples("epoch_0", question_embedding, k=temp_ds.__len__())
    # a = list(dict.fromkeys(r["docno"]))
    # df = pd.DataFrame.from_dict({'docno': r["docno"], 'score': s})
    # df["score"] = df.score.apply(lambda x: 100 - x)
    # bm25 = df1[["topic", "docno"]][df1.topic == 101].drop_duplicates("docno")
    # temp = bm25.merge(df, on="docno", how="inner").drop_duplicates("docno")
    # temp.insert(1, 'iter', 0)
    # temp.insert(len(temp.columns), 'rank', range(1,len(temp)+1))
    # temp.insert(len(temp.columns), 'tag', 'DPR_0')
    # runs.append(temp)
#%%

questions = [topics.loc[topic].description for topic in trange(101,151)]
x = q_tokenizer(questions, return_tensors="pt", truncation=True, padding=True)
question_embedding = q_encoder(**x)
s, r = ds.get_nearest_examples_batch("epoch_0", question_embedding[0].numpy(), k=10000)
#%%
runs = []
for i, topic in enumerate(trange(101,151)):
    temp = pd.DataFrame.from_dict({'docno': r[i]["docno"], 'score': s[i]})
    bm25 = df1[["topic", "docno"]][df1.topic == topic].drop_duplicates("docno")
    temp = bm25.merge(temp, on="docno", how="inner").drop_duplicates("docno")
    temp["score"] = temp.score.apply(lambda x: 100 - x)
    # temp.insert(0, 'topic', topic)
    temp.insert(1, 'iter', 0)
    temp.insert(len(temp.columns), 'rank', range(1,len(temp)+1))
    temp.insert(len(temp.columns), 'tag', 'DPR_0')
    temp = temp.drop_duplicates('docno')[:1000]
    runs.append(temp)

runs_df = pd.concat(runs)
runs_df.shape
#%%
runs_df["docno"] = runs_df["docno"].map(lambda x: f"en.noclean.c4-train.0{x[3:7]}-of-07168.{int(x[8:])}")
runs_df.to_csv("runs/run.dpr0", sep=" ", index=False, header=False)