"""One-time raw CSV.gz -> Parquet conversion.

Why this exists: gzip is NOT splittable, so Spark reads the 379MB source in a
single task on one core. Everything downstream reads Parquet instead, which is
splittable, columnar and compressed -- so the other 27 cores actually get used.

This step is a faithful type-cast of the raw file. No feature engineering here;
that lives in src/features/build_features.py so the raw layer stays reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyspark.sql import functions as F
from pyspark.sql.types import (
    IntegerType, LongType, StringType, StructField, StructType, DoubleType,
)

from src.spark_session import PROJECT_ROOT, get_spark

RAW = PROJECT_ROOT / "data" / "raw" / "learning_traces.13m.csv.gz"
OUT = PROJECT_ROOT / "data" / "processed" / "traces.parquet"

# Explicit schema: inferSchema would force a second full pass over a
# non-splittable gzip file just to guess types we already know.
SCHEMA = StructType([
    StructField("p_recall",          DoubleType(),  True),
    StructField("timestamp",         LongType(),    True),
    StructField("delta",             LongType(),    True),
    StructField("user_id",           StringType(),  True),
    StructField("learning_language", StringType(),  True),
    StructField("ui_language",       StringType(),  True),
    StructField("lexeme_id",         StringType(),  True),
    StructField("lexeme_string",     StringType(),  True),
    StructField("history_seen",      IntegerType(), True),
    StructField("history_correct",   IntegerType(), True),
    StructField("session_seen",      IntegerType(), True),
    StructField("session_correct",   IntegerType(), True),
])


def main() -> None:
    spark = get_spark("ingest")
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.csv(str(RAW), header=True, schema=SCHEMA, mode="PERMISSIVE")

    # The gz read yields one partition; repartition so the Parquet write and all
    # downstream scans are parallel. Partition by user_id so that every future
    # per-learner window (streaks, learner ability) is shuffle-free.
    df = df.repartition(64, "user_id")

    df.write.mode("overwrite").parquet(str(OUT))

    out = spark.read.parquet(str(OUT))
    n = out.count()
    print(f"\nrows written      : {n:,}")
    print(f"distinct users    : {out.select('user_id').distinct().count():,}")
    print(f"distinct lexemes  : {out.select('lexeme_id').distinct().count():,}")
    print(f"null p_recall     : {out.filter(F.col('p_recall').isNull()).count():,}")
    print(f"output partitions : {out.rdd.getNumPartitions()}")
    spark.stop()


if __name__ == "__main__":
    main()
