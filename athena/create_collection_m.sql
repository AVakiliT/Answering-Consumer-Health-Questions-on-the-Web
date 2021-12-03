CREATE EXTERNAL TABLE IF NOT EXISTS `ccindex`.`Collection_M` (
  `docno` string,
  `text` string,
  `timestamp` timestamp,
  `url` string
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
WITH SERDEPROPERTIES (
  'serialization.format' = '1'
) LOCATION 's3://amirvt/Collection_M/'
TBLPROPERTIES ('has_encrypted_data'='false');