"""Model settings per version, so a new version can change the method while older ones keep
theirs exactly (the daily job re-rates the live version with its own settings).

v3 = V2.1 as validated. v4 = the refinements found by spot-checking V2.1 (design doc, 16 Nov):
  1. par from the match itself      match_par_weight
  2. minimum charge per ball         min_ball_charge
  3. smooth batting-stage edges      stage_blend_balls
  4/7. all-rounders by role norms    secondary = "role", secondary_c
  5. harsher low end of the scale    anchors
  6. blend in the cost to the team   raw_blend
"""
import json
from dataclasses import asdict, dataclass, replace
from typing import Optional, Tuple

ANCHORS_V3: Tuple[Tuple[float, float], ...] = (
    (0.0, 3.0), (0.01, 3.0), (0.25, 5.8), (0.50, 6.5), (0.90, 8.0), (0.99, 9.3), (1.0, 10.0))
# v4: failures read harsher (bottom 10% ends at 4.5, bottom quarter at 5.5); median and top unchanged.
ANCHORS_V4: Tuple[Tuple[float, float], ...] = (
    (0.0, 3.0), (0.01, 3.0), (0.10, 4.5), (0.25, 5.5), (0.50, 6.5), (0.90, 8.0), (0.99, 9.3), (1.0, 10.0))


@dataclass(frozen=True)
class ModelParams:
    # --- rating engine (06)
    match_par_weight: float = 0.0   # 1: scale x (match's own scoring rate / expected) ** weight
    min_ball_charge: float = 0.0    # 2: a batter is charged at least this share of par runs per ball
    # --- 0-10 scale (07)
    stage_blend_balls: int = 0      # 3: within this many balls of a stage edge, blend with the next stage
    secondary: str = "balls"        # 4/7: "balls": smaller part by raw balls, weight n/(n+k);
    secondary_k: float = 24.0       #      "role": by balls relative to a typical innings (for that
    secondary_c: float = 1.0        #      stage) or spell, weight r/(r+c)
    anchors: Tuple[Tuple[float, float], ...] = ANCHORS_V3   # 5
    raw_blend: float = 0.0          # 6: share of the score from the plain within-competition rank of impact

    def to_json(self) -> dict:
        d = asdict(self)
        d["anchors"] = [list(a) for a in self.anchors]
        return d


V3 = ModelParams()
V4 = ModelParams(match_par_weight=0.4, min_ball_charge=0.5, stage_blend_balls=10,
                 secondary="role", secondary_c=1.0, anchors=ANCHORS_V4, raw_blend=0.2)
BY_VERSION = {"v1": V3, "v2": V3, "v3": V3, "v4": V4}


def for_version(version: str, **overrides) -> ModelParams:
    """The settings for a version (unknown, newer versions get the latest), with any
    non-None overrides applied (e.g. a notebook widget)."""
    p = BY_VERSION.get(version, V4)
    changes = {k: v for k, v in overrides.items() if v is not None}
    return replace(p, **changes) if changes else p


def record(spark, t, version: str, extra: dict) -> None:
    """Merge settings into this version's params in gold.model_versions (if the row exists)."""
    from pyspark.sql import functions as F
    from . import storage
    rows = spark.table(t.gold_model_versions).filter(F.col("version") == version).collect()
    if not rows:
        return
    row = rows[0].asDict()
    params = json.loads(row["params"] or "{}")
    params.update(extra)
    row["params"] = json.dumps(params)
    schema = spark.table(t.gold_model_versions).schema
    df = spark.createDataFrame([tuple(row[f.name] for f in schema.fields)], schema)
    storage.replace_rows(spark, t.gold_model_versions, df, df.select("version"), "version")


def describe(p: Optional[ModelParams] = None) -> dict:
    return (p or V3).to_json()
