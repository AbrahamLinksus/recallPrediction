"""Spark session factory tuned for local-mode work on a single box.

Local mode runs the driver and executor in ONE JVM, so `spark.driver.memory`
is effectively the whole memory budget -- `spark.executor.memory` is ignored.
"""
from __future__ import annotations

import os
import pathlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Spark 4.x needs a JDK that still exposes jdk.internal.ref.Cleaner.
# JDK 26 removed it, so we pin 21 (LTS) regardless of the system default.
# Prefer the project-local Temurin JDK, fall back to a system install.
_JDK_CANDIDATES = [
    PROJECT_ROOT / ".jdk",
    pathlib.Path("/usr/lib/jvm/java-21-openjdk"),
    pathlib.Path("/usr/lib/jvm/java-17-openjdk"),
]


def _find_java_home() -> str | None:
    for c in _JDK_CANDIDATES:
        if (c / "bin" / "java").exists():
            return str(c)
    return None


def get_spark(app_name: str = "recall-prediction", cores: str = "*", driver_memory: str = "8g"):
    from pyspark.sql import SparkSession

    java_home = _find_java_home()
    if java_home is None:
        raise RuntimeError(
            "No compatible JDK found (need 17 or 21; JDK 26 removed a class Spark requires)."
        )
    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = f"{java_home}/bin:{os.environ['PATH']}"

    # /tmp is tmpfs (RAM) on this machine -- spilling there would consume the
    # memory the spill is meant to relieve. Point scratch at real disk.
    scratch = PROJECT_ROOT / "data" / "spark-tmp"
    scratch.mkdir(parents=True, exist_ok=True)

    return (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.local.dir", str(scratch))
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
