import gzip
import json
from pathlib import Path

import jsonlines
# from bs4 import BeautifulSoup
# from html2text import HTML2Text
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator

#%%
# path = "html2text"
# Path(f"./{path}").mkdir(parents=True, exist_ok=True)
#
# h = HTML2Text()
# h.ignore_links = True
# h.images_to_alt = True
# h.single_line_break = True
# h.escape_snob = True


docs_file = jsonlines.open("./data/2019qrels.docs.jsonl", mode='w')

with open('./data/2019qrels.docids.txt', 'r') as f:
    docs = sorted(list(set([i.strip() for i in f.readlines()])))

#%%
for doc_id in tqdm(docs):
    # print("1")
    #ClueWeb12_12/1215wb/1215wb-
    (_, warc, warc_part, id) = doc_id.split("-")
    corp_dir = "/project/6003284/smucker/group-data/corpora/ClueWeb12-B13/DiskB/"
    warc_file = f"{corp_dir}/ClueWeb12_{warc[:2]}/{warc}/{warc}-{warc_part}.warc.gz"
    with gzip.open(warc_file, 'rb') as stream:
        for i, record in enumerate(ArchiveIterator(stream)):
            if record.rec_type == 'response':
                trec_id = (record.rec_headers.get_header('WARC-TREC-ID'))
                uri = (record.rec_headers.get_header('WARC-Target-URI'))
                if trec_id == doc_id:
                    # print(trec_id)
                    # continue

                    byte_stream = record.content_stream().read()
                    # soup = BeautifulSoup(byte_stream, "html.parser")
                    # title = soup.find("title")
                    # if title:
                    #     print(title.string)
                    #     title_string = title.string
                    # else:
                    #     title_string = ""
                    # text = h.handle(byte_stream.decode(errors='ignore'))
                    # text = '\n'.join([t for t in soup.stripped_strings])
                    # text = soup.get_text()
                    # break into lines and remove leading and trailing space on each
                    # lines = (line.strip() for line in text.splitlines())
                    # # break multi-headlines into a line each
                    # chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    # # drop blank lines
                    # text = '\n'.join(chunk for chunk in chunks if chunk)
                    a = {
                        'id': trec_id,
                        # "text" : text,
                        # "title": title_string,
                        'url': uri}
                    docs_file.write(a)

#%%
from joblib import Parallel, delayed
def f(doc_id):
    (_, warc, warc_part, id) = doc_id.split("-")
    corp_dir = "/project/6003284/smucker/group-data/corpora/ClueWeb12-B13/DiskB/"
    warc_file = f"{corp_dir}/ClueWeb12_{warc[:2]}/{warc}/{warc}-{warc_part}.warc.gz"
    with gzip.open(warc_file, 'rb') as stream:
        for i, record in enumerate(ArchiveIterator(stream)):
            if record.rec_type == 'response':
                trec_id = (record.rec_headers.get_header('WARC-TREC-ID'))
                uri = (record.rec_headers.get_header('WARC-Target-URI'))
                if trec_id == doc_id:
                    a = {
                        'id': trec_id,
                        # "text" : text,
                        # "title": title_string,
                        'url': uri}
                    return a

doc_urls = Parallel(n_jobs=16)(delayed(f)(i) for i in tqdm(docs))
with jsonlines.open('2019qrels.urls.jsonl', mode='w') as writer:
    for o in doc_urls:
        writer.write(o)

import pandas as pd
df = pd.DataFrame(filter(lambda x: x is not None, doc_urls))

df.to_csv("2019qrels.urls.csv", sep=" ", index=False)

#%%
