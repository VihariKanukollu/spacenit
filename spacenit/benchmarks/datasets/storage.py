"""Dataset paths configured via environment variables.

This repo's downstream evaluation datasets are typically provided via a mount
in cluster environments. The upstream defaults point to internal locations
(``/weka/...``), but in many external setups the persistent mount point is
``/workspace``.

We therefore:
- Always honor explicit environment variables (``GEOBENCH_DIR``, etc.)
- Otherwise, prefer common ``/workspace`` locations if they exist
- Fall back to the original internal defaults as a last resort
"""

import os

from upath import UPath

# Only available to internal users
_DEFAULTS = {
    "GEOBENCH_DIR": "/weka/dfive-default/presto-geobench/dataset/geobench",
    "BREIZHCROPS_DIR": "/weka/dfive-default/skylight/presto_eval_sets/breizhcrops",
    "MADOS_DIR": "/weka/dfive-default/presto_eval_sets/mados",
    "FLOODS_DIR": "/weka/dfive-default/presto_eval_sets/floods",
    "PASTIS_DIR": "/weka/dfive-default/presto_eval_sets/pastis_r",
    "PASTIS_DIR_ORIG": "/weka/dfive-default/presto_eval_sets/pastis_r_origsize",
    "PASTIS_DIR_PARTITION": "/weka/dfive-default/presto_eval_sets/pastis",
    "NANDI_DIR": "/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625",
    "AWF_DIR": "/weka/dfive-default/rslearn-eai/datasets/crop/awf_2023",
}

def _first_existing_path(*candidates: str) -> str | None:
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    return None


def _dataset_path(env_var: str, *, workspace_candidates: list[str], fallback: str) -> UPath:
    if (v := os.getenv(env_var)) is not None and v != "":
        return UPath(v)
    if (picked := _first_existing_path(*workspace_candidates)) is not None:
        return UPath(picked)
    return UPath(fallback)


GEOBENCH_DIR = _dataset_path(
    "GEOBENCH_DIR",
    workspace_candidates=[
        "/workspace/datasets/geobench",
        "/workspace/geobench",
        "/workspace/Spatial/geobench",
    ],
    fallback=_DEFAULTS["GEOBENCH_DIR"],
)
BREIZHCROPS_DIR = _dataset_path(
    "BREIZHCROPS_DIR",
    workspace_candidates=[
        "/workspace/datasets/breizhcrops",
        "/workspace/breizhcrops",
        "/workspace/Spatial/breizhcrops",
    ],
    fallback=_DEFAULTS["BREIZHCROPS_DIR"],
)
MADOS_DIR = _dataset_path(
    "MADOS_DIR",
    workspace_candidates=[
        "/workspace/datasets/mados",
        "/workspace/mados",
        "/workspace/Spatial/mados",
    ],
    fallback=_DEFAULTS["MADOS_DIR"],
)
FLOODS_DIR = _dataset_path(
    "FLOODS_DIR",
    workspace_candidates=[
        "/workspace/datasets/floods",
        "/workspace/floods",
        "/workspace/Spatial/floods",
    ],
    fallback=_DEFAULTS["FLOODS_DIR"],
)
PASTIS_DIR = _dataset_path(
    "PASTIS_DIR",
    workspace_candidates=[
        "/workspace/datasets/pastis_r",
        "/workspace/pastis_r",
        "/workspace/Spatial/pastis_r",
    ],
    fallback=_DEFAULTS["PASTIS_DIR"],
)
PASTIS_DIR_ORIG = _dataset_path(
    "PASTIS_DIR_ORIG",
    workspace_candidates=[
        "/workspace/datasets/pastis_r_origsize",
        "/workspace/pastis_r_origsize",
        "/workspace/Spatial/pastis_r_origsize",
    ],
    fallback=_DEFAULTS["PASTIS_DIR_ORIG"],
)
PASTIS_DIR_PARTITION = _dataset_path(
    "PASTIS_DIR_PARTITION",
    workspace_candidates=[
        "/workspace/datasets/pastis",
        "/workspace/pastis",
        "/workspace/Spatial/pastis",
    ],
    fallback=_DEFAULTS["PASTIS_DIR_PARTITION"],
)
NANDI_DIR = _dataset_path(
    "NANDI_DIR",
    workspace_candidates=[
        "/workspace/datasets/nandi",
        "/workspace/nandi",
        "/workspace/Spatial/nandi",
    ],
    fallback=_DEFAULTS["NANDI_DIR"],
)
AWF_DIR = _dataset_path(
    "AWF_DIR",
    workspace_candidates=[
        "/workspace/datasets/awf",
        "/workspace/awf",
        "/workspace/Spatial/awf",
    ],
    fallback=_DEFAULTS["AWF_DIR"],
)
