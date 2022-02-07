#%%


import pandas as pd
import xml.etree.cElementTree as et

from torch.nn import DataParallel
from tqdm import trange

df1 = pd.read_parquet("data/Top1kBM25_1p_passages")

topics = pd.read_csv("data/topics.csv", index_col="topic", sep="\t")

#%%
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import os
# def setup_ddp():
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     i = 0
#     N = 1
#     torch.cuda.set_device(i)
#     torch.distributed.init_process_group(
#         backend='gloo', world_size=N, rank=i
#     )
#     return i
# rank = setup_ddp()

torch.set_grad_enabled(False)
ctx_encoder = \
    DPRContextEncoder\
        .from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(0)

ctx_encoder = DataParallel(ctx_encoder, device_ids=list(range(torch.cuda.device_count())))
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")


#%%
year = 2021
from datasets import load_dataset
ds = load_dataset("parquet",
                  data_files={
                      "train": f"data/Top1kBM25_{'' if year == 2021 else '2019'}1p_passages/*.parquet"
                  },
                  split='train')\
    # .filter(lambda e: e['topic'] == 101)
#%%
# ds = ds.shard(num_shards=8, index=0, contiguous=True)
with torch.cuda.amp.autocast():
    ds_with_embeddings = ds.map(
        lambda example: {
            'embeddings': ctx_encoder(**ctx_tokenizer(example["passage"], return_tensors="pt", truncation=True, padding=True).to(0))[0].cpu().numpy()
        }
                            , batched=True, batch_size=128)
#%%
ds_with_embeddings.add_faiss_index(column='embeddings', index_name="epoch_0")
#%%

#%%
# question = "Can cranberries prevent urinary tract infections?"
# question = "Will wearing an ankle brace help heal achilles tendonitis?"
# question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
# scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('epoch_0', question_embedding, k=10)
#%%
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
torch.set_grad_enabled(False)
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
ds_with_embeddings.save_faiss_index("epoch_0", "data/faiss_index/2021/epoch_0")

#%%


#%%
