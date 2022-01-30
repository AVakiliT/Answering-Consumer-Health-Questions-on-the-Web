#%%


import pandas as pd
import xml.etree.cElementTree as et

df = pd.read_parquet("./data/Top1kBM25_2019_1p_passages")

#%%
topics_head = ['topic', 'query', 'cochranedoi', 'description', 'narrative']
xml_root = et.parse("./data/2019topics.xml")
rows = xml_root.findall('topic')
xml_data = [
    [int(row.find('number').text), row.find('query').text, row.find('cochranedoi').text,
     row.find('description').text, row.find('narrative').text] for row in rows]
topics = pd.DataFrame(xml_data, columns=topics_head)
topics_answers = pd.read_csv("./data/2019topics_efficacy.txt", header=None, sep=' ',
                             names=['topic', 'efficacy'])
topics = pd.merge(topics, topics_answers, how='left', on='topic')

#%%
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import os
def setup_ddp():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    i = 0
    N = 1
    torch.cuda.set_device(i)
    torch.distributed.init_process_group(
        backend='gloo', world_size=N, rank=i
    )
    return i
rank = setup_ddp()

torch.set_grad_enabled(False)
ctx_encoder = \
    DPRContextEncoder\
        .from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(rank)

ctx_encoder = DDP(ctx_encoder, device_ids=[rank])
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")


#%%
from datasets import load_dataset
ds = load_dataset("parquet",
                  data_files={
                      "train": "data/Top1kBM25_2019_1p_passages/part-00000-9e1561a2-6119-46ab-8c19-9076394ba1dd-c000.snappy.parquet"
                  },
                  split='train')\
    .filter(lambda e: e['topic'] == 1)
#%%
# ds = ds.shard(num_shards=8, index=0, contiguous=True)
ds_with_embeddings = ds.map(
    lambda example: {
        'embeddings': ctx_encoder(**ctx_tokenizer(example["passage"], return_tensors="pt", truncation=True, padding=True))[0].cpu().numpy()
    }
                        , batched=True, batch_size=128)
#%%
ds_with_embeddings.add_faiss_index(column='embeddings', index_name="epoch_0")
#%%
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
#%%
question = "Can cranberries prevent urinary tract infections?"
question_embedding = q_encoder(**q_tokenizer(question, return_tensors="pt"))[0][0].numpy()
scores, retrieved_examples = ds_with_embeddings.get_nearest_examples('embeddings', question_embedding, k=10)

#%%
ds_with_embeddings.save_faiss_index("epoch_0", "data/faiss_index/epoch_0")