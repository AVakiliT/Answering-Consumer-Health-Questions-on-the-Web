from nltk import sent_tokenize
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import pyspark.sql.functions as f

window_size, step = 6, 3

year = 2019
suffix = "_2019" if year == 2019 else ""

df = spark.read.load(f"Top1kBM25{suffix}")
print(df.count())

schema = ArrayType(StringType())


def lol(s):
    seq = sent_tokenize(s)
    if len(seq) <= window_size:
        return [s]
    return [' '.join(seq[i: i + window_size]) for i in range(0, len(seq) - window_size + 1, step)]


lol_udf = udf(lol, schema)

df_new = df.withColumn("passage", lol_udf("text")).selectExpr("docno", "topic", "score as bm25",
                                                              "explode(passage) as passage")

df_new.repartition(1).write.mode("overwrite").save(f"Top1kBM25{suffix}_1p_passages")
