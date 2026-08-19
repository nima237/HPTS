# -*- coding: utf-8 -*-
"""Permutation test of cross-asset exchangeability.

Under exact exchangeability the asset label carries no information, so the fitted
asset-specific deviations delta_i are pure estimation noise. Shuffling the asset
labels within each date destroys any genuine asset-specific structure while leaving
the cross-sectional composition of every date untouched. Refitting on the shuffled
panel therefore gives the null distribution of the symmetry-breaking ratio.

Observed ratio above that null => exchangeability is rejected for that block.
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "robustness"
SEED, REFIT_DAYS, ALPHA, POOL = 77, 60, 100.0, 10.0
HOLDOUT = "2024-01-01"

HAR  = ["c_logrv_6", "c_logrv_24", "c_logrv_72", "c_logrv_168"]
IV   = ["log_dvol_open", "dvol_chg_1_open", "dvol_chg_5_open", "iv_rv_gap_open",
        "log_dvol_eth_open", "dvol_eth_chg_1_open"]
XSEC = ["c_xs_logrv24", "c_xs_logrv24_rank", "c_xs_dispersion", "c_xs_breadth",
        "c_beta_mkt_168", "c_corr_mkt_168", "c_resid_ret_24", "c_mkt_ret_24"]
TAIL = ["c_signed_jump_24", "c_jump_asym_24", "c_jump_share_24", "c_vol_ratio_6_72",
        "c_vol_ratio_24_168", "c_path_eff_24", "c_range_rel", "c_amihud_24"]
FULL = HAR + XSEC + TAIL + IV
BLOCKS = {"HAR": HAR, "IV": IV, "TAIL": TAIL, "XSEC": XSEC}


def fit_deltas(train, test_dates_mask_unused, feats, alpha, pool):
    """Fit the panel on `train` and return (common slope, deviation matrix)."""
    x = train[feats].to_numpy(float)
    m, s = x.mean(0), x.std(0)
    s = np.where(s < 1e-12, 1.0, s)
    x0 = (x - m) / s
    syms = sorted(train.symbol.unique())
    smap = {v: i for i, v in enumerate(syms)}
    ids = train.symbol.map(smap).to_numpy()
    oh = np.zeros((len(train), len(syms)))
    oh[np.arange(len(train)), ids] = 1
    inter = np.concatenate([oh * x0[:, [j]] for j in range(len(feats))], axis=1)
    X = np.concatenate([np.ones((len(train), 1)), x0, oh, inter], axis=1)
    pen = np.asarray([0.0] + [alpha] * len(feats) + [0.25 * alpha] * len(syms)
                     + [alpha * pool] * inter.shape[1])
    beta = np.linalg.solve(X.T @ X + np.diag(pen + 1e-10), X.T @ train.y.to_numpy(float))
    k = len(feats)
    return beta[1:1 + k], beta[1 + k + len(syms):].reshape(k, len(syms))


def block_ratios(common, delta, feats):
    out = {}
    for name, cols in BLOCKS.items():
        j = [feats.index(c) for c in cols]
        out[name] = float(np.sqrt((delta[j] ** 2).mean()) / np.abs(common[j]).mean())
    return out


def run(df, cutoffs, shuffle_rng=None):
    """Mean symmetry-breaking ratio per block across refits.
    If shuffle_rng is given, asset labels are permuted within each date first."""
    d = df
    if shuffle_rng is not None:
        d = df.copy()
        lab = d.symbol.to_numpy().copy()
        # permute labels inside each date block
        order = np.argsort(d.timestamp.to_numpy(), kind="stable")
        ts = d.timestamp.to_numpy()[order]
        bounds = np.flatnonzero(np.diff(ts)) + 1
        for chunk in np.split(order, bounds):
            lab[chunk] = shuffle_rng.permutation(lab[chunk])
        d["symbol"] = lab
    acc = []
    for cut in cutoffs:
        train = d[d.timestamp < cut]
        common, delta = fit_deltas(train.reset_index(drop=True), None, FULL, ALPHA, POOL)
        acc.append(block_ratios(common, delta, FULL))
    return {b: float(np.mean([a[b] for a in acc])) for b in BLOCKS}


if __name__ == "__main__":
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    df = pd.read_csv(ROOT / "model_data.csv", parse_dates=["timestamp"])
    dates = np.array(sorted(df[df.timestamp >= HOLDOUT].timestamp.unique()))
    cutoffs = [dates[i] for i in range(0, len(dates), REFIT_DAYS)]
    print(f"refits: {len(cutoffs)}   permutations: {n_perm}", flush=True)

    t0 = time.time()
    observed = run(df, cutoffs)
    print(f"observed: { {k: round(v,4) for k,v in observed.items()} }   ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.default_rng(SEED)
    null = []
    t0 = time.time()
    for i in range(n_perm):
        null.append(run(df, cutoffs, shuffle_rng=rng))
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n_perm}  ({el:.0f}s elapsed, ~{el/(i+1)*(n_perm-i-1):.0f}s left)", flush=True)

    rows = []
    for b in BLOCKS:
        draws = np.array([x[b] for x in null])
        obs = observed[b]
        p = float((np.sum(draws >= obs) + 1) / (len(draws) + 1))
        rows.append({"block": b, "observed": obs,
                     "null_mean": float(draws.mean()),
                     "null_q025": float(np.quantile(draws, .025)),
                     "null_q975": float(np.quantile(draws, .975)),
                     "p_one_sided": p})
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "exchangeability_test.csv", index=False)
    json.dump({"seed": SEED, "permutations": n_perm, "refits": len(cutoffs),
               "alpha": ALPHA, "pool": POOL},
              open(OUT / "exchangeability_meta.json", "w"), indent=2)
    print(res.round(4).to_string(index=False), flush=True)
