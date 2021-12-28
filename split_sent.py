import re
import sys

import spacy
from nltk import sent_tokenize
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import pyspark.sql.functions as f
from spacy.tokens import Span
from typing import List

window_size, step = 256, 128

spark = SparkSession.builder.appName("MyApp").getOrCreate()

df = spark.read.load(f"{sys.argv[1]}")
print(df.count())

schema = ArrayType(StringType())

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


def tokenize_windows(s):
    s = re.sub('\s+', " ", s.strip())
    doc = nlp(s)
    sentences = [sent.sent.text.strip() for sent in doc.sents if len(sent) > 5]
    tokens = nlp(' '.join(sentences))

    if len(tokens) <= window_size:
        return tokens.text.strip()
    return [' '.join(tokens[i: i + window_size]) for i in range(0, len(tokens), step)]


def sentencize(s):
    s = re.sub('\s+', " ", s.strip())
    sentences = [sent.sent.text.strip() for sent in nlp(s).sents if len(sent) > 3]
    if len(sentences) <= window_size:
        return [s]
    return [' '.join(sentences[i: i + window_size]) for i in range(0, len(sentences), step)]

lol_udf = udf(tokenize_windows, schema)

df_new = df.withColumn("passage", lol_udf("text")).selectExpr("docno", "topic", "score as bm25",
                                                              "explode(passage) as passage")

df_new.repartition(1).write.mode("overwrite").save(f"{sys.argv[2]}_1p_passages")
