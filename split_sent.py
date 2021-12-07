from nltk import sent_tokenize
from pyspark.sql.functions import udf
from pyspark.sql.types import *

window_size, step = 6, 3

df = spark.read.load("Top1kBM25_32p")

schema = ArrayType(StringType())


def lol(s):
    seq = sent_tokenize(s)
    if len(seq) <= window_size:
        return [s]
    return [' '.join(seq[i: i + window_size]) for i in range(0, len(seq) - window_size + 1, step)]


lol_udf = udf(lol, schema)

df_new = df.withColumn("passage", lol_udf("text")).selectExpr("docno", "topic", "score as bm25",
                                                              "explode(passage) as passage")

df_new.repartition(1).write.mode("overwrite").save("Top1kBM25_1p_passages")
