from __future__ import annotations

from pathlib import Path
import json
import os
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

import analysis_main as core

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "robustness"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
RNG = np.random.default_rng(core.SEED)


def performance(frame: pd.DataFrame, model: str, comparator: str = "HAR") -> dict:
    a = (frame.y - frame[comparator]) ** 2
    b = (frame.y - frame[model]) ** 2
    daily = (a - b).groupby(frame.timestamp).mean()
    t = core.nw_t(daily)
    return {
        "model": model, "comparator": comparator, "n": len(frame),
        "mse_gain_pct": 100 * (1 - b.sum() / a.sum()),
        "nw_t": t, "p_two_sided": core.two_sided_p(t),
    }


def panel_hgb_design(train: pd.DataFrame, test: pd.DataFrame):
    x, z = core.standardise(train, test, core.FULL)
    symbols = sorted(train.symbol.unique())
    ids = {s: i for i, s in enumerate(symbols)}
    tr_id = train.symbol.map(ids).to_numpy()
    te_id = test.symbol.map(ids).fillna(-1).astype(int).to_numpy()
    tr_oh = np.zeros((len(train), len(symbols)))
    te_oh = np.zeros((len(test), len(symbols)))
    tr_oh[np.arange(len(train)), tr_id] = 1
    valid = te_id >= 0
    te_oh[np.arange(len(test))[valid], te_id[valid]] = 1
    return np.c_[x, tr_oh], np.c_[z, te_oh]


def fit_panel_hgb(train, test, hp):
    x, z = panel_hgb_design(train, test)
    fit = HistGradientBoostingRegressor(
        max_iter=hp["max_iter"], learning_rate=hp["learning_rate"],
        max_leaf_nodes=hp["max_leaf_nodes"], min_samples_leaf=hp["min_samples_leaf"],
        l2_regularization=hp["l2_regularization"], random_state=core.SEED,
    ).fit(x, train.y.to_numpy(float))
    return fit.predict(z)


def pooled_nonlinear(df: pd.DataFrame, base: pd.DataFrame):
    train = df[df.timestamp < "2023-01-01"].reset_index(drop=True)
    val = df[(df.timestamp >= "2023-01-01") & (df.timestamp < "2024-01-01")].reset_index(drop=True)
    configs = [
        {"max_iter": 150, "learning_rate": .03, "max_leaf_nodes": 7, "min_samples_leaf": 40, "l2_regularization": 3.0},
        {"max_iter": 250, "learning_rate": .03, "max_leaf_nodes": 15, "min_samples_leaf": 40, "l2_regularization": 3.0},
        {"max_iter": 200, "learning_rate": .05, "max_leaf_nodes": 7, "min_samples_leaf": 80, "l2_regularization": 10.0},
        {"max_iter": 250, "learning_rate": .03, "max_leaf_nodes": 31, "min_samples_leaf": 80, "l2_regularization": 10.0},
    ]
    grid = []
    for i, hp in enumerate(configs):
        pred = fit_panel_hgb(train, val, hp)
        grid.append({"config": i, **hp, "validation_mse": np.mean((val.y.to_numpy() - pred) ** 2)})
    grid = pd.DataFrame(grid).sort_values("validation_mse")
    grid.to_csv(OUT / "nonlinear_validation_grid.csv", index=False)
    hp = configs[int(grid.iloc[0].config)]
    rows = []
    dates = np.array(sorted(df[df.timestamp >= "2024-01-01"].timestamp.unique()))
    for start in range(0, len(dates), core.REFIT_DAYS):
        block = dates[start:start + core.REFIT_DAYS]
        cutoff = block[0]
        tr = df[df.timestamp < cutoff].reset_index(drop=True)
        te = df[df.timestamp.isin(block)].reset_index(drop=True)
        pr = fit_panel_hgb(tr, te, hp)
        rows.extend(zip(te.timestamp, te.symbol, te.y, pr))
    q = pd.DataFrame(rows, columns=["timestamp", "symbol", "y", "Pooled_HGB"])
    q = q.merge(base[["timestamp", "symbol", "HAR", "HPTS_Final", "Full_PerAsset"]],
                on=["timestamp", "symbol"], how="inner")
    q.to_csv(OUT / "pooled_nonlinear_predictions.csv", index=False)
    return pd.DataFrame([
        performance(q, "Pooled_HGB"),
        performance(q, "HPTS_Final", "Pooled_HGB"),
        performance(q, "Full_PerAsset", "Pooled_HGB"),
    ]), hp


def make_stale(df: pd.DataFrame, days: int):
    work = df.copy()
    names = []
    for c in core.IV:
        name = f"stale{days}_{c}"
        work[name] = work.groupby("symbol")[c].shift(days)
        names.append(name)
    return work.dropna(subset=names).reset_index(drop=True), names


def make_permuted(df: pd.DataFrame):
    work = df.copy()
    pure = ["log_dvol_open", "dvol_chg_1_open", "dvol_chg_5_open",
            "log_dvol_eth_open", "dvol_eth_chg_1_open"]
    names = [f"perm_{c}" for c in pure] + ["perm_iv_rv_gap_open"]
    for name in names:
        work[name] = np.nan
    for split, idx in work.groupby("split").groups.items():
        part = work.loc[idx, ["timestamp", *pure]].drop_duplicates("timestamp").sort_values("timestamp")
        source = np.arange(len(part))
        local = np.random.default_rng(core.SEED + len(split))
        local.shuffle(source)
        for c in pure:
            mapper = dict(zip(part.timestamp, part[c].to_numpy()[source]))
            work.loc[idx, f"perm_{c}"] = work.loc[idx, "timestamp"].map(mapper)
    work["perm_iv_rv_gap_open"] = work["perm_log_dvol_open"] - work["c_logrv_24"]
    return work.dropna(subset=names).reset_index(drop=True), names


def placebos(df: pd.DataFrame, base: pd.DataFrame, hp: dict):
    rows = [performance(base, "HPTS_Final") | {"variant": "current_open"}]
    pred_out = base[["timestamp", "symbol", "y", "HAR", "HPTS_Final"]].copy()
    variants = []
    for lag in (1, 2, 3, 5, 7):
        work, iv = make_stale(df, lag)
        variants.append((f"dvol_stale_{lag}d", work, core.SPOT + iv))
    work, iv = make_permuted(df)
    variants.append(("dvol_date_permuted", work, core.SPOT + iv))
    for label, work, features in variants:
        q = core.rolling_panel(work, features, features, hp, label)
        q = q[q.timestamp >= "2024-01-01"].merge(
            base[["timestamp", "symbol", "HAR"]], on=["timestamp", "symbol"], how="inner")
        rows.append(performance(q, label) | {"variant": label})
        pred_out = pred_out.merge(q[["timestamp", "symbol", label]], on=["timestamp", "symbol"], how="left")
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "dvol_placebo_results.csv", index=False)
    pred_out.to_csv(OUT / "dvol_placebo_predictions.csv", index=False)
    direct_rows = []
    joined = base[["timestamp", "symbol", "y", "HPTS_Final", "HPTS_NoIV"]].merge(
        pred_out.drop(columns=["y", "HAR", "HPTS_Final"]), on=["timestamp", "symbol"], how="inner")
    for comparator in ["HPTS_NoIV", *[c for c in joined if c.startswith("dvol_")]]:
        a = (joined.y - joined[comparator]) ** 2
        b = (joined.y - joined.HPTS_Final) ** 2
        daily = (a - b).groupby(joined.timestamp).mean()
        t = core.nw_t(daily)
        direct_rows.append({"comparator": comparator, "gain_current_pct": 100 * (1 - b.sum() / a.sum()),
                            "nw_t": t, "p_two_sided": core.two_sided_p(t)})
    direct = pd.DataFrame(direct_rows)
    direct["holm_p"] = core.adjust_p(direct.p_two_sided, "holm")
    direct.to_csv(OUT / "dvol_direct_comparisons.csv", index=False)
    perm = "dvol_date_permuted"
    a = (joined.y - joined.HPTS_NoIV) ** 2
    b = (joined.y - joined[perm]) ** 2
    daily = (a - b).groupby(joined.timestamp).mean()
    t = core.nw_t(daily)
    pd.DataFrame([{"comparison": "date-permuted DVOL vs HPTS-NoIV",
                   "gain_pct": 100 * (1 - b.sum() / a.sum()), "nw_t": t,
                   "p_two_sided": core.two_sided_p(t)}]).to_csv(OUT / "dvol_permuted_vs_noiv.csv", index=False)
    return result


def block_analysis(df: pd.DataFrame, base: pd.DataFrame):
    blocks = {"HAR": core.HAR, "XSEC": core.XSEC, "TAIL": core.TAIL, "IV": core.IV}
    rows, grids = [], []
    for omitted, cols in blocks.items():
        features = [c for c in core.FULL if c not in cols]
        label = f"HPTS_without_{omitted}"
        grid, hp = core.tune_panel(df, features, features, label)
        grids.append(grid)
        q = core.rolling_panel(df, features, features, hp, label)
        q = q[q.timestamp >= "2024-01-01"].merge(
            base[["timestamp", "symbol", "HAR", "HPTS_Final"]], on=["timestamp", "symbol"], how="inner")
        row = performance(q, "HPTS_Final", label)
        row.update({"omitted_block": omitted, "ablation_model": label,
                    "ablated_gain_vs_har_pct": performance(q, label)["mse_gain_pct"],
                    "alpha": hp["alpha"], "pool": hp["pool"]})
        rows.append(row)
    pd.concat(grids).to_csv(OUT / "block_ablation_validation_grids.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "block_ablation_results.csv", index=False)

    coef_rows = []
    dates = np.array(sorted(df[df.timestamp >= "2024-01-01"].timestamp.unique()))
    hp = {"alpha": 100.0, "pool": 10.0}
    for start in range(0, len(dates), core.REFIT_DAYS):
        cutoff = dates[start]
        tr = df[df.timestamp < cutoff].reset_index(drop=True)
        te = df[df.timestamp.isin(dates[start:start + core.REFIT_DAYS])].reset_index(drop=True)
        x, _, penalty = core.panel_design(tr, te, core.FULL, core.FULL, hp["alpha"], hp["pool"])
        beta = np.linalg.solve(x.T @ x + np.diag(penalty + 1e-10), x.T @ tr.y.to_numpy(float))
        p, a = len(core.FULL), tr.symbol.nunique()
        common = beta[1:1+p]
        dev = beta[1+p+a:].reshape(p, a)
        for block, cols in blocks.items():
            idx = [core.FULL.index(c) for c in cols]
            coef_rows.append({"cutoff": cutoff, "block": block,
                              "common_abs_mean": np.mean(np.abs(common[idx])),
                              "deviation_rms": np.sqrt(np.mean(dev[idx] ** 2)),
                              "total_rms": np.sqrt(np.mean(common[idx] ** 2) + np.mean(dev[idx] ** 2))})
    coef = pd.DataFrame(coef_rows)
    coef.to_csv(OUT / "coefficient_block_summary_by_refit.csv", index=False)
    summary = coef.groupby("block")[["common_abs_mean", "deviation_rms", "total_rms"]].agg(["mean", "std"])
    summary.to_csv(OUT / "coefficient_block_summary.csv")
    return out, coef


def economic_value(df: pd.DataFrame, base: pd.DataFrame):
    future = df.sort_values(["symbol", "timestamp"])[["timestamp", "symbol", "ret_24"]].copy()
    future["future_return"] = future.groupby("symbol").ret_24.shift(-1)
    q = base.merge(future[["timestamp", "symbol", "future_return"]], on=["timestamp", "symbol"], how="left").dropna()
    target = 0.02
    rows = []
    for cost_bps in (0, 5, 10, 20):
        strategies = {}
        for model in ("HAR", "HPTS_Final"):
            vol = np.sqrt(np.exp(np.clip(q[model].to_numpy(), -30, 10)))
            weight = np.clip(target / vol, 0, 2)
            tmp = q[["timestamp", "symbol"]].copy()
            tmp["weight"] = weight
            tmp["turnover"] = tmp.groupby("symbol").weight.diff().abs().fillna(0)
            net = weight * q.future_return.to_numpy() - cost_bps / 10000 * tmp.turnover.to_numpy()
            strategies[model] = (weight, net)
            daily = pd.DataFrame({"timestamp": q.timestamp, "net": net}).groupby("timestamp").mean().net
            mean, var = daily.mean(), daily.var(ddof=1)
            rows.append({"cost_bps": cost_bps, "model": model, "n": len(q),
                         "annual_return": 365 * mean, "annual_volatility": np.sqrt(365 * var),
                         "sharpe": np.sqrt(365) * mean / np.sqrt(var),
                         "certainty_equivalent_gamma3": 365 * (mean - 1.5 * var),
                         "mean_turnover": pd.Series(tmp.turnover).mean(),
                         "variance_target_mae": np.mean(np.abs((weight * q.future_return.to_numpy()) ** 2 - target ** 2))})
        h = strategies["HPTS_Final"][1]
        b = strategies["HAR"][1]
        util_diff = pd.Series((h - 1.5*h*h) - (b - 1.5*b*b)).groupby(q.timestamp).mean()
        t = core.nw_t(util_diff)
        rows[-1]["utility_diff_nw_t_vs_har"] = t
        rows[-1]["utility_diff_p_vs_har"] = core.two_sided_p(t)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "economic_value_volatility_timing.csv", index=False)
    return out


def figures(placebo, nonlinear, ablation, coef):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    p = placebo.sort_values("mse_gain_pct")
    ax.barh(p.variant, p.mse_gain_pct, color=["#0B6E4F" if x == "current_open" else "#8093A7" for x in p.variant])
    ax.set_xlabel("MSE improvement relative to HAR (%)")
    ax.set_title("DVOL timing and permutation diagnostics", loc="left", weight="bold")
    fig.tight_layout(); fig.savefig(FIG / "figure_E1_dvol_placebos.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    a = ablation.sort_values("mse_gain_pct")
    ax.barh(a.omitted_block, a.mse_gain_pct, color="#2E74B5")
    ax.axvline(0, color="black", lw=.8)
    ax.set_xlabel("HPTS-Final MSE gain over the ablated refit (%)")
    ax.set_title("Incremental contribution of predictor blocks", loc="left", weight="bold")
    fig.tight_layout(); fig.savefig(FIG / "figure_E2_block_ablation.png", dpi=220); plt.close(fig)

    c = coef.groupby("block").total_rms.mean().sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.barh(c.index, c.values, color="#0B6E4F")
    ax.set_xlabel("Mean standardised coefficient RMS")
    ax.set_title("Coefficient magnitude by predictor block", loc="left", weight="bold")
    fig.tight_layout(); fig.savefig(FIG / "figure_E3_coefficient_blocks.png", dpi=220); plt.close(fig)


def coefficient_only(df: pd.DataFrame):
    blocks = {"HAR": core.HAR, "XSEC": core.XSEC, "TAIL": core.TAIL, "IV": core.IV}
    coef_rows = []
    dates = np.array(sorted(df[df.timestamp >= "2024-01-01"].timestamp.unique()))
    hp = {"alpha": 100.0, "pool": 10.0}
    for start in range(0, len(dates), core.REFIT_DAYS):
        cutoff = dates[start]
        tr = df[df.timestamp < cutoff].reset_index(drop=True)
        te = df[df.timestamp.isin(dates[start:start + core.REFIT_DAYS])].reset_index(drop=True)
        x, _, penalty = core.panel_design(tr, te, core.FULL, core.FULL, hp["alpha"], hp["pool"])
        beta = np.linalg.solve(x.T @ x + np.diag(penalty + 1e-10), x.T @ tr.y.to_numpy(float))
        p, a = len(core.FULL), tr.symbol.nunique()
        common = beta[1:1+p]
        dev = beta[1+p+a:].reshape(p, a)
        for block, cols in blocks.items():
            idx = [core.FULL.index(c) for c in cols]
            coef_rows.append({"cutoff": cutoff, "block": block,
                              "common_abs_mean": np.mean(np.abs(common[idx])),
                              "deviation_rms": np.sqrt(np.mean(dev[idx] ** 2)),
                              "total_rms": np.sqrt(np.mean(common[idx] ** 2) + np.mean(dev[idx] ** 2))})
    coef = pd.DataFrame(coef_rows)
    coef.to_csv(OUT / "coefficient_block_summary_by_refit.csv", index=False)
    coef.groupby("block")[["common_abs_mean", "deviation_rms", "total_rms"]].agg(["mean", "std"]).to_csv(
        OUT / "coefficient_block_summary.csv")
    return coef


def main():
    df = core.load_and_audit()
    base = pd.read_csv(core.OUT / "holdout_predictions.csv", parse_dates=["timestamp"])
    choices = pd.read_csv(core.OUT / "selected_hyperparameters.csv").set_index("model")
    hp = {"alpha": float(choices.loc["HPTS_Final", "alpha"]), "pool": float(choices.loc["HPTS_Final", "pool"])}
    print("1/4 economic-value diagnostic", flush=True)
    econ = economic_value(df, base)
    print("2/4 pooled nonlinear benchmark", flush=True)
    nonlinear, nonlinear_hp = pooled_nonlinear(df, base)
    nonlinear.to_csv(OUT / "pooled_nonlinear_results.csv", index=False)
    print("3/4 DVOL placebos", flush=True)
    placebo = placebos(df, base, hp)
    print("4/4 block attribution", flush=True)
    ablation, coef = block_analysis(df, base)
    figures(placebo, nonlinear, ablation, coef)
    meta = {"seed": core.SEED, "holdout_start": "2024-01-01", "nonlinear_hp": nonlinear_hp,
            "note": "All nonlinear tuning used calendar 2023; stale/permuted tests reuse pre-selected HPTS hyperparameters."}
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nNONLINEAR\n", nonlinear.to_string(index=False))
    print("\nPLACEBOS\n", placebo.to_string(index=False))
    print("\nABLATIONS\n", ablation.to_string(index=False))
    print("\nECONOMIC\n", econ.to_string(index=False))


if __name__ == "__main__":
    if "--coeff-only" in sys.argv:
        data = core.load_and_audit()
        coefficients = coefficient_only(data)
        figures(pd.read_csv(OUT / "dvol_placebo_results.csv"),
                pd.read_csv(OUT / "pooled_nonlinear_results.csv"),
                pd.read_csv(OUT / "block_ablation_results.csv"), coefficients)
        print(coefficients.groupby("block")[["common_abs_mean", "deviation_rms", "total_rms"]].mean())
    elif "--placebo-only" in sys.argv:
        data = core.load_and_audit()
        saved = pd.read_csv(core.OUT / "holdout_predictions.csv", parse_dates=["timestamp"])
        choices = pd.read_csv(core.OUT / "selected_hyperparameters.csv").set_index("model")
        selected_hp = {"alpha": float(choices.loc["HPTS_Final", "alpha"]),
                       "pool": float(choices.loc["HPTS_Final", "pool"])}
        print(placebos(data, saved, selected_hp).to_string(index=False))
    else:
        main()
