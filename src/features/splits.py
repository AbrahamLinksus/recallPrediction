"""Deterministic, hash-based sampling and train/test splitting.

Everything here keys off a hash of user_id rather than a random shuffle, for
two reasons:

1. Reproducibility without carrying a seed or a materialised id list around --
   the same user lands in the same bucket on every run, on any machine.
2. Nesting. The prototype sample is a bucket range of the full dataset, so the
   "sample -> scale" step changes ONE constant and nothing else. A user in the
   500K prototype keeps the same train/test assignment at 13M rows.

Split is by USER, not by row: every feature we build (streak, historical
success rate, learner ability) is a per-learner window, so splitting rows would
put the same learner's history on both sides and leak. Holding out whole
learners is what licenses the claim "generalises to learners not seen in
training".
"""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

# Salts keep the sample bucket and the split bucket statistically independent,
# so the prototype isn't accidentally skewed toward one side of the split.
_SAMPLE_SALT = "sample_v1"
_SPLIT_SALT = "split_v1"


def _bucket(col: str, salt: str, n: int):
    """Stable integer in [0, n) derived from a column value."""
    return F.abs(F.xxhash64(F.concat(F.col(col), F.lit(salt)))) % n


def add_sample_bucket(df: DataFrame, n: int = 1000) -> DataFrame:
    return df.withColumn("_sample_bucket", _bucket("user_id", _SAMPLE_SALT, n))


def sample_by_user(df: DataFrame, fraction: float, n: int = 1000) -> DataFrame:
    """Keep every row of a deterministic ~`fraction` subset of learners.

    Selects whole learners, so each sampled learner's trace stays intact.
    """
    cutoff = int(round(fraction * n))
    return add_sample_bucket(df, n).filter(F.col("_sample_bucket") < cutoff).drop("_sample_bucket")


def add_user_split(
    df: DataFrame, train_frac: float = 0.7, calib_frac: float = 0.1, n: int = 100
) -> DataFrame:
    """Label each row train/calib/test by holding out whole learners.

    Three ways, not two, because the calibration layer must be fit on data the
    model did not train on. Fitting isotonic regression on training predictions
    would learn to correct the model's (much better behaved) in-sample
    confidence, baking in the wrong correction for unseen data.

    Every boundary is drawn on user_id, so no learner appears in more than one
    split and the "generalises to unseen learners" claim covers all three.
    """
    train_cut = int(round(train_frac * n))
    calib_cut = int(round((train_frac + calib_frac) * n))
    b = _bucket("user_id", _SPLIT_SALT, n)
    return df.withColumn(
        "split",
        F.when(b < train_cut, F.lit("train"))
        .when(b < calib_cut, F.lit("calib"))
        .otherwise(F.lit("test")),
    )


def add_fit_holdout(df: DataFrame, fit_frac: float = 0.85, n: int = 100) -> DataFrame:
    """Carve an early-stopping holdout out of the TRAIN split, again by user.

    Kept distinct from `calib` so the calibration set stays completely untouched
    by model fitting -- using one set for both early stopping and calibration
    would let the calibrator see data the model had already tuned against.
    """
    cut = int(round(fit_frac * n))
    return df.withColumn(
        "fit_role",
        F.when(F.col("split") != "train", F.lit("na"))
        .when(_bucket("user_id", "fit_v1", n) < cut, F.lit("fit"))
        .otherwise(F.lit("val")),
    )


def add_time_split(df: DataFrame, train_frac: float = 0.7, calib_frac: float = 0.1) -> DataFrame:
    """Secondary robustness split: train on the past, test on the future.

    Uses a global timestamp cutoff, so a learner may appear on both sides -- but
    only with their earlier reviews in train and later ones in test, which is
    exactly what a scheduler faces in production.
    """
    q = df.approxQuantile("timestamp", [train_frac, train_frac + calib_frac], 0.001)
    return df.withColumn(
        "split",
        F.when(F.col("timestamp") <= F.lit(q[0]), F.lit("train"))
        .when(F.col("timestamp") <= F.lit(q[1]), F.lit("calib"))
        .otherwise(F.lit("test")),
    )
