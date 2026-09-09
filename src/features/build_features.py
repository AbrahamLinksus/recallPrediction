"""Feature engineering for recall prediction.

Target convention
-----------------
We keep `session_seen` (trials) and `session_correct` (successes) as the target
columns rather than collapsing to a single bit. Training expands each row into
two weighted rows (y=1 weight=correct, y=0 weight=seen-correct), which is an
exact representation of `session_seen` Bernoulli trials. That keeps the model's
output on the per-review probability scale a scheduler consumes.

Leakage discipline
------------------
Every learner/item feature here is computed over rows STRICTLY PRIOR to the
current SESSION (`rangeBetween` on timestamp, not `rowsBetween`). Lexeme encoding is
fit on training users only. Both are load-bearing for the claim that the model
generalises to learners it never saw.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyspark.sql import DataFrame, Window, functions as F

from src.spark_session import PROJECT_ROOT, get_spark
from src.features.splits import add_fit_holdout, add_time_split, add_user_split, sample_by_user

SECONDS_PER_DAY = 86400.0
# Basis of exponential-decay curves at different half-lives (days). HLR predicts
# a per-item half-life; we instead hand the model several fixed decay rates and
# let it learn which mixture fits, which keeps the pipeline a pure classifier.
HALF_LIVES_DAYS = [1.0, 7.0, 30.0]
# Smoothing strength for lexeme target encoding, in pseudo-trials. A lexeme needs
# ~TE_SMOOTHING trials before its own rate outweighs the global prior.
TE_SMOOTHING = 20.0


def add_recency(df: DataFrame) -> DataFrame:
    d = F.col("delta") / F.lit(SECONDS_PER_DAY)
    df = df.withColumn("delta_days", d).withColumn("log_delta", F.log1p(d))
    for h in HALF_LIVES_DAYS:
        name = f"decay_h{int(h)}"
        df = df.withColumn(name, F.pow(F.lit(2.0), -d / F.lit(h)))
    return df


def add_frequency(df: DataFrame) -> DataFrame:
    seen = F.col("history_seen")
    return (
        df.withColumn("history_wrong", seen - F.col("history_correct"))
        .withColumn("history_rate", F.col("history_correct") / F.greatest(seen, F.lit(1)))
        .withColumn("log_history_seen", F.log1p(seen.cast("double")))
    )


def add_learner_features(df: DataFrame) -> DataFrame:
    """Expanding per-learner ability over that learner's own earlier reviews."""
    # rangeBetween (not rowsBetween) over the timestamp: 99.7% of rows share a
    # timestamp with another row of the same user, because one session reviews
    # many items at a single instant. A row-based frame lets item N of a session
    # see the outcomes of items 1..N-1 of that SAME session -- information a
    # scheduler does not have, since it scores every item in a session before any
    # is answered. A range frame ending at -1 second admits only strictly earlier
    # sessions. rangeBetween needs a single numeric ORDER BY, hence no tie-break.
    w = (
        Window.partitionBy("user_id")
        .orderBy("timestamp")
        .rangeBetween(Window.unboundedPreceding, -1)
    )
    prior_trials = F.coalesce(F.sum("session_seen").over(w), F.lit(0))
    prior_correct = F.coalesce(F.sum("session_correct").over(w), F.lit(0))
    return (
        df.withColumn("user_prior_trials", prior_trials)
        .withColumn(
            "user_ability",
            # Smoothed toward the global prior so a learner's first few reviews
            # don't produce 0.0 or 1.0 ability estimates.
            (prior_correct + F.lit(TE_SMOOTHING) * F.col("_global_rate"))
            / (prior_trials + F.lit(TE_SMOOTHING)),
        )
        .withColumn("log_user_prior_trials", F.log1p(prior_trials.cast("double")))
    )


def add_streak(df: DataFrame) -> DataFrame:
    """Consecutive perfect sessions immediately before the current review.

    Computed per (user, lexeme). Trick: track the row index of the most recent
    imperfect session among prior rows; the streak is however many rows have
    elapsed since then.
    """
    order = Window.partitionBy("user_id", "lexeme_id").orderBy("timestamp")
    # Same range-frame reasoning as add_learner_features: only strictly earlier
    # sessions count. dense_rank (not row_number) so rows tied on timestamp are
    # counted as one session rather than several.
    prior = order.rangeBetween(Window.unboundedPreceding, -1)

    df = df.withColumn("_rn", F.dense_rank().over(order))
    # 1 when a session was not perfect.
    df = df.withColumn("_imperfect", (F.col("p_recall") < 1.0).cast("int"))
    # Row index of the last imperfect session strictly before this row (0 if none).
    df = df.withColumn(
        "_last_fail_rn",
        F.coalesce(F.max(F.col("_rn") * F.col("_imperfect")).over(prior), F.lit(0)),
    )
    df = df.withColumn("item_prior_sessions", F.col("_rn") - F.lit(1))
    df = df.withColumn("streak", F.col("item_prior_sessions") - F.col("_last_fail_rn"))
    return df.drop("_rn", "_imperfect", "_last_fail_rn")


def add_item_difficulty(df: DataFrame, global_rate: float) -> DataFrame:
    """Smoothed target encoding on lexeme_id, fit on TRAINING rows only.

    Unseen lexemes fall back to the global prior -- required, since ~10% of test
    lexemes never appear in training under a by-user split.
    """
    train = df.filter(F.col("split") == "train")
    stats = train.groupBy("lexeme_id").agg(
        F.sum("session_seen").alias("_lex_trials"),
        F.sum("session_correct").alias("_lex_correct"),
    )
    stats = stats.withColumn(
        "lexeme_difficulty",
        (F.col("_lex_correct") + F.lit(TE_SMOOTHING * global_rate))
        / (F.col("_lex_trials") + F.lit(TE_SMOOTHING)),
    ).withColumn("log_lexeme_trials", F.log1p(F.col("_lex_trials").cast("double")))

    df = df.join(stats.select("lexeme_id", "lexeme_difficulty", "log_lexeme_trials"), "lexeme_id", "left")
    # Fallback for lexemes absent from training.
    df = df.withColumn("lexeme_unseen", F.col("lexeme_difficulty").isNull().cast("int"))
    df = df.fillna({"lexeme_difficulty": global_rate, "log_lexeme_trials": 0.0})
    return df


FEATURE_COLUMNS = [
    # recency
    "delta_days", "log_delta", "decay_h1", "decay_h7", "decay_h30",
    # frequency
    "history_seen", "history_correct", "history_wrong", "history_rate", "log_history_seen",
    # item
    "lexeme_difficulty", "log_lexeme_trials", "lexeme_unseen",
    # learner
    "user_ability", "user_prior_trials", "log_user_prior_trials",
    "streak", "item_prior_sessions",
]
BASELINE_COLUMNS = [
    "delta_days", "log_delta", "decay_h1", "decay_h7", "decay_h30",
    "history_seen", "history_correct", "history_wrong", "history_rate", "log_history_seen",
]


def build(sample_fraction: float | None = 0.039, split: str = "user") -> None:
    spark = get_spark("build-features")
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(str(PROJECT_ROOT / "data/processed/traces.parquet"))
    if sample_fraction:
        df = sample_by_user(df, sample_fraction)
    df = df.filter(F.col("session_seen") > 0)

    df = add_user_split(df) if split == "user" else add_time_split(df)
    df = add_fit_holdout(df)

    # Global per-trial recall, computed on TRAINING rows only -- it is the prior
    # that both target encodings smooth toward, so it must not see test data.
    tr = df.filter(F.col("split") == "train").agg(
        F.sum("session_correct").alias("c"), F.sum("session_seen").alias("s")
    ).first()
    global_rate = tr["c"] / tr["s"]

    df = df.withColumn("_global_rate", F.lit(global_rate))
    df = add_recency(df)
    df = add_frequency(df)
    df = add_learner_features(df)
    df = add_streak(df)
    df = add_item_difficulty(df, global_rate)
    df = df.drop("_global_rate")

    keep = ["user_id", "lexeme_id", "timestamp", "split", "fit_role", "session_seen", "session_correct", "p_recall"]
    out = df.select(*keep, *FEATURE_COLUMNS)

    suffix = "sample" if sample_fraction else "full"
    dest = PROJECT_ROOT / "data/processed" / f"features_{split}_{suffix}.parquet"
    out.write.mode("overwrite").parquet(str(dest))
    print(f"global per-trial rate (train only): {global_rate:.6f}")
    print(f"wrote {dest}")
    spark.stop()


if __name__ == "__main__":
    frac = None if "--full" in sys.argv else 0.039
    sp = "time" if "--time-split" in sys.argv else "user"
    build(sample_fraction=frac, split=sp)
