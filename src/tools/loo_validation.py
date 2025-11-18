"""Leave-one-out validation by family for DOFT structural noise outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_noise(noise_json: Path) -> Dict[str, dict]:
    return json.loads(noise_json.read_text())


def mean_template(entries: Dict[str, dict]) -> dict:
    def _mean_vec(vecs: List[dict]) -> dict:
        if not vecs:
            return {}
        keys = vecs[0].keys()
        out = {}
        for key in keys:
            vals = [float(v.get(key, 0.0)) for v in vecs if isinstance(v, dict)]
            out[key] = float(np.mean(vals)) if vals else 0.0
        return out

    xi = float(np.mean([float(e.get("xi", 0.0)) for e in entries.values()])) if entries else 0.0
    delta_T_list = [next(iter(e.get("delta_T", {}).values()), {}) for e in entries.values()]
    delta_space_list = [next(iter(e.get("delta_space", {}).values()), {}) for e in entries.values()]
    delta_P_list = [next(iter(e.get("delta_P", {}).values()), {}) for e in entries.values()]
    return {
        "xi": xi,
        "delta_T": _mean_vec(delta_T_list),
        "delta_space": _mean_vec(delta_space_list),
        "delta_P": _mean_vec(delta_P_list),
    }


def diff_metric(target: dict, predicted: dict) -> float:
    flat_target = []
    flat_pred = []
    for key in ("xi",):
        flat_target.append(float(target.get(key, 0.0)))
        flat_pred.append(float(predicted.get(key, 0.0)))
    for vec_key in ("delta_T", "delta_space", "delta_P"):
        tvec = target.get(vec_key, {})
        pvec = predicted.get(vec_key, {})
        if isinstance(tvec, dict) and isinstance(pvec, dict):
            for k in tvec.keys():
                flat_target.append(float(tvec.get(k, 0.0)))
                flat_pred.append(float(pvec.get(k, 0.0)))
    arr_t = np.array(flat_target)
    arr_p = np.array(flat_pred)
    if arr_t.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr_t - arr_p)))


def loo_validation(noise_json: Path) -> pd.DataFrame:
    data = load_noise(noise_json)
    # Build family map
    fam_map: Dict[str, List[str]] = {}
    for name, entry in data.items():
        fam = str(entry.get("category", "Unknown"))
        fam_map.setdefault(fam, []).append(name)
    rows = []
    for fam, materials in fam_map.items():
        if len(materials) < 3:
            continue
        for left_out in materials:
            train_entries = {m: data[m] for m in materials if m != left_out}
            template = mean_template(train_entries)
            target = data[left_out]
            loss = diff_metric(target, template)
            rows.append({"family": fam, "left_out": left_out, "abs_diff": loss})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-out validation over families.")
    parser.add_argument("--noise-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="CSV to store LOO results")
    args = parser.parse_args()
    df = loo_validation(args.noise_json)
    if args.output:
        df.to_csv(args.output, index=False)
    else:
        print(df)


if __name__ == "__main__":
    main()
