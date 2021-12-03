CREATE TABLE IF NOT EXISTS collection_m_from_c4
  WITH (external_location = 's3://amirvt/results/c4_collection_m',
      format='PARQUET') AS
      SELECT A.url AS url,
         A.fetch_time AS fetch_time,
         warc_filename,
         warc_record_offset,
         warc_record_length,
         warc_segment
FROM
    (SELECT url,
         warc_filename,
         warc_record_offset,
         warc_record_length,
         warc_segment,
         fetch_time
    FROM ccindex
    WHERE (crawl = 'CC-MAIN-2019-18'
            AND subset = 'warc')) AS A
INNER JOIN
    (collection_m) AS B
    ON A.url=B.url
        AND A.fetch_time=B.timestamp;