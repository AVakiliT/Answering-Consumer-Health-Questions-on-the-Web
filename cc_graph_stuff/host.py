from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from tldextract import extract
#%%


schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("r_host", StringType(), True)])

spark = SparkSession.builder.appName("MyApp").getOrCreate()
path = "/project/6004803/smucker/group-data/commoncrawl/cc-graph/"
v = spark.read.csv(path + "host-vertices", header=False, schema=schema, sep="\t")

e = spark.read.csv(path + "host-edges", header=False, schema=StructType([
    StructField("id_from", IntegerType(), True),
    StructField("id_to", IntegerType(), True)]), sep="\t")

@udf(StringType())
def f(x):
    return ".".join(x.split(".")[::-1])
v = v.withColumn("host", f(col("r_host")))
v.head(5)

df1 = spark.read.load("./data/Top1kBM25_2019")
df2 = spark.read.load("./data/Top1kBM25")
# df3 = spark.read.load("./data/qrel_2021")
df = df1.select("topic docno url".split()).union(
    df2.select("topic docno url".split())
)\
#     .union(
#     df3.select("topic docno url".split())
# )
#%%
@udf(StringType())
def url2host(url):
    e = extract(url)
    return f"{(e.subdomain + '.') if e.subdomain and e.subdomain != 'www' else ''}{e.domain}.{e.suffix}"
df = df.withColumn("host", url2host(col("url")))
fv = v.join(df.select("host").distinct(), "host", "inner")
fv.write.save("./data/host-graph/filtered_verticies", mode="overwrite")
#%%

fe = e.join(fv.selectExpr("id as id_to,host as host_to".split(",")), "id_to", "inner")\
    .join(fv.selectExpr("id as id_from,host as host_from".split(",")), "id_from", "inner")

fe.write.save("./data/host-graph/filtered_edges", mode="overwrite")
# df
# v = pd.concat((pd.read_csv(f, names="id r_host".split(), sep="\t") for f in tqdm(sorted(glob(f"{path}/*.txt.gz")))))
# v["host"] = v.r_host.apply(lambda x : ".".join(x.split(".")[::-1]))
#
# #%%
#
#
#
# df = pd.concat(
#     [pd.read_parquet(f"./data/Top1kBM25"),
#      pd.read_parquet(f"./data/Top1kBM25_2019")])
#
# def url2host(url):
#     e = tldextract.extract(url)
#     return f"{e.subdomain}{'.' if e.subdomain else ' '}{e.domain}.{e.suffix}"
#
# hosts = df.url.apply(url2host).drop_duplicates()
