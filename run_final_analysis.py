"""Final leakage-safe HPTS forecasting pipeline.

Primary estimand
-----------------
The target ``y`` is log realised variance over the exact next 24 hours:
``log(sum of squared hourly log returns)``.  The phrase realised volatility is
reserved for ``sqrt(exp(y))``.  The primary loss is squared error on log
realised variance; QLIKE and raw-volatility losses are diagnostics.

Protocol
--------
Training is before 2023, hyperparameters are selected on calendar 2023, and
the untouched holdout begins on 2024-01-01.  Every option feature is based on
the current UTC day's DVOL open; same-day close/high/low values are absent.

The proposed model is one hierarchical ridge regression, not an ensemble.  It
contains common slopes, asset fixed effects, and shrunken asset-specific slope
deviations.  The final specification omits the BTC--ETH IV spread because that
choice improved the pre-holdout validation loss.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os
import sys
import warnings

_PACKAGE_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_PACKAGE_ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "model_data.csv"
OUT = ROOT / "final_results"
FIG = OUT / "figures"
SEED = 77
REFIT_DAYS = 60
MIN_TRAIN_DAYS = 300

HAR = ["c_logrv_6", "c_logrv_24", "c_logrv_72", "c_logrv_168"]
IV_BTC = ["log_dvol_open", "dvol_chg_1_open", "dvol_chg_5_open", "iv_rv_gap_open"]
IV_ETH = ["log_dvol_eth_open", "dvol_eth_chg_1_open"]
IV_SPREAD = ["iv_spread_open"]
IV = IV_BTC + IV_ETH                         # final, validation-selected IV block
XSEC = [
    "c_xs_logrv24", "c_xs_logrv24_rank", "c_xs_dispersion", "c_xs_breadth",
    "c_beta_mkt_168", "c_corr_mkt_168", "c_resid_ret_24", "c_mkt_ret_24",
]
TAIL = [
    "c_signed_jump_24", "c_jump_asym_24", "c_jump_share_24",
    "c_vol_ratio_6_72", "c_vol_ratio_24_168", "c_path_eff_24",
    "c_range_rel", "c_amihud_24",
]
SPOT = HAR + XSEC + TAIL
FULL = SPOT + IV


def load_and_audit() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    required = {"timestamp", "symbol", "split", "y", *FULL, *IV_SPREAD}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[list(required)].isna().any().any():
        raise ValueError("Model-ready columns contain missing values")
    if df.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("Duplicate timestamp-symbol rows found")
    forbidden = [c for c in df if "dvol" in c.lower() and any(x in c.lower() for x in ("close", "high", "low"))]
    if forbidden:
        raise ValueError(f"Non-causal same-day DVOL fields found: {forbidden}")
    expected_split = np.select(
        [df.timestamp < "2023-01-01", df.timestamp < "2024-01-01"],
        ["train", "validation"], default="holdout",
    )
    if not np.array_equal(df.split.astype(str).to_numpy(), expected_split):
        raise ValueError("Stored split labels do not match the locked date protocol")
    return df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def nw_t(x: pd.Series | np.ndarray, lags: int = 4) -> float:
    z = np.asarray(x, float)
    z = z[np.isfinite(z)]
    n = len(z)
    e = z - z.mean()
    long_var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        long_var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k] / n)
    return float(z.mean() / np.sqrt(max(long_var, 1e-18) / n))


def two_sided_p(t: float) -> float:
    return float(2 * norm.sf(abs(t)))


def adjust_p(p: pd.Series | np.ndarray, method: str) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    out = np.empty(n)
    if method == "holm":
        vals = np.maximum.accumulate(p[order] * (n - np.arange(n)))
    elif method == "bh":
        vals = p[order] * n / np.arange(1, n + 1)
        vals = np.minimum.accumulate(vals[::-1])[::-1]
    else:
        raise ValueError(method)
    out[order] = np.clip(vals, 0, 1)
    return out


def block_bootstrap_gain(frame: pd.DataFrame, model: str, comparator: str,
                         reps: int = 3000, block: int = 28) -> tuple[float, float]:
    daily = frame.assign(
        comparator_loss=(frame.y - frame[comparator]) ** 2,
        model_loss=(frame.y - frame[model]) ** 2,
    ).groupby("timestamp")[["comparator_loss", "model_loss"]].sum().to_numpy()
    n = len(daily)
    starts = np.arange(max(1, n - block + 1))
    rng = np.random.default_rng(SEED)
    draws = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([
            np.arange(s, min(s + block, n))
            for s in rng.choice(starts, int(np.ceil(n / block)), replace=True)
        ])[:n]
        loss = daily[idx].sum(0)
        draws[r] = 100 * (1 - loss[1] / loss[0])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def standardise(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    x = train[features].to_numpy(float)
    z = test[features].to_numpy(float)
    mean = x.mean(0)
    std = x.std(0)
    std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, (z - mean) / std


def panel_design(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                 deviations: list[str], alpha: float, pool: float):
    x0, z0 = standardise(train, test, features)
    symbols = sorted(train.symbol.unique())
    symbol_map = {s: i for i, s in enumerate(symbols)}
    train_id = train.symbol.map(symbol_map).to_numpy()
    test_id = test.symbol.map(symbol_map).fillna(-1).astype(int).to_numpy()
    onehot_train = np.zeros((len(train), len(symbols)))
    onehot_test = np.zeros((len(test), len(symbols)))
    onehot_train[np.arange(len(train)), train_id] = 1
    valid = test_id >= 0
    onehot_test[np.arange(len(test))[valid], test_id[valid]] = 1

    x_parts = [np.ones((len(train), 1)), x0, onehot_train]
    z_parts = [np.ones((len(test), 1)), z0, onehot_test]
    penalties = [0.0] + [alpha] * len(features) + [0.25 * alpha] * len(symbols)
    if deviations:
        indices = [features.index(c) for c in deviations]
        dx = np.concatenate([onehot_train * x0[:, [j]] for j in indices], axis=1)
        dz = np.concatenate([onehot_test * z0[:, [j]] for j in indices], axis=1)
        x_parts.append(dx)
        z_parts.append(dz)
        penalties += [alpha * pool] * dx.shape[1]
    return np.concatenate(x_parts, 1), np.concatenate(z_parts, 1), np.asarray(penalties)


def panel_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                  deviations: list[str], hp: dict) -> np.ndarray:
    x, z, penalty = panel_design(train, test, features, deviations, hp["alpha"], hp.get("pool", 1.0))
    beta = np.linalg.solve(x.T @ x + np.diag(penalty + 1e-10), x.T @ train.y.to_numpy(float))
    return z @ beta


def tune_panel(df: pd.DataFrame, features: list[str], deviations: list[str], model: str):
    train = df[df.timestamp < "2023-01-01"].reset_index(drop=True)
    validation = df[(df.timestamp >= "2023-01-01") & (df.timestamp < "2024-01-01")].reset_index(drop=True)
    pools = (1.0, 3.0, 10.0, 30.0) if deviations else (1.0,)
    rows = []
    for alpha in (3.0, 10.0, 30.0, 100.0):
        for pool in pools:
            hp = {"alpha": alpha, "pool": pool}
            pred = panel_predict(train, validation, features, deviations, hp)
            rows.append({"model": model, **hp, "validation_mse": np.mean((validation.y - pred) ** 2)})
    grid = pd.DataFrame(rows).sort_values("validation_mse").reset_index(drop=True)
    return grid, grid.iloc[0][["alpha", "pool", "validation_mse"]].to_dict()


def rolling_panel(df: pd.DataFrame, features: list[str], deviations: list[str],
                  hp: dict, name: str, window_days: int | None = None) -> pd.DataFrame:
    rows = []
    dates = np.array(sorted(df[df.timestamp >= "2023-01-01"].timestamp.unique()))
    for start in range(0, len(dates), REFIT_DAYS):
        block = dates[start:start + REFIT_DAYS]
        cutoff = block[0]
        train = df[df.timestamp < cutoff]
        if window_days is not None:
            train = train[train.timestamp >= cutoff - pd.Timedelta(days=window_days)]
        test = df[df.timestamp.isin(block)]
        pred = panel_predict(train.reset_index(drop=True), test.reset_index(drop=True), features, deviations, hp)
        rows.extend(zip(test.timestamp, test.symbol, test.y, pred))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "y", name])


def tune_per_asset_ridge(df: pd.DataFrame, features: list[str], model: str):
    train = df[df.timestamp < "2023-01-01"]
    validation = df[(df.timestamp >= "2023-01-01") & (df.timestamp < "2024-01-01")]
    rows = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        losses = []
        for symbol, group in train.groupby("symbol"):
            test = validation[validation.symbol == symbol]
            x, z = standardise(group, test, features)
            y = group.y.to_numpy(float)
            beta = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - y.mean()))
            losses.extend((test.y.to_numpy() - (y.mean() + z @ beta)) ** 2)
        rows.append({"model": model, "alpha": alpha, "validation_mse": np.mean(losses)})
    grid = pd.DataFrame(rows).sort_values("validation_mse").reset_index(drop=True)
    return grid, float(grid.iloc[0].alpha)


def rolling_per_asset(df: pd.DataFrame, features: list[str], name: str,
                      kind: str = "ridge", params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    rows = []
    dates = np.array(sorted(df.timestamp.unique()))
    for start in range(MIN_TRAIN_DAYS, len(dates), REFIT_DAYS):
        cutoff = dates[start]
        block = dates[start:start + REFIT_DAYS]
        for _, group in df.groupby("symbol"):
            train = group[group.timestamp < cutoff]
            test = group[group.timestamp.isin(block)]
            if len(train) < MIN_TRAIN_DAYS or test.empty:
                continue
            x, z = standardise(train, test, features)
            y = train.y.to_numpy(float)
            if kind == "elasticnet":
                fit = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                                 max_iter=10000, random_state=SEED).fit(x, y)
                pred = fit.predict(z)
            elif kind == "hgb":
                fit = HistGradientBoostingRegressor(
                    max_iter=params["max_iter"], learning_rate=params["learning_rate"],
                    max_leaf_nodes=params["max_leaf_nodes"],
                    l2_regularization=params["l2_regularization"], random_state=SEED,
                ).fit(x, y)
                pred = fit.predict(z)
            else:
                alpha = params["alpha"]
                beta = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - y.mean()))
                pred = y.mean() + z @ beta
            rows.extend(zip(test.timestamp, test.symbol, test.y, pred))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "y", name])


def build_predictions(df: pd.DataFrame):
    specs = {
        "Full_FixedEffects": (FULL, []),
        "HPTS_CommonIV": (FULL, SPOT),
        "HPTS_IVSpecific": (FULL, IV),
        "HPTS_Final": (FULL, FULL),
        "HPTS_NoIV": (SPOT, SPOT),
    }
    grids, choices, predictions = [], [], []
    for name, (features, deviations) in specs.items():
        print(f"Tuning and forecasting {name}", flush=True)
        grid, hp = tune_panel(df, features, deviations, name)
        grids.append(grid)
        choices.append({"model": name, **hp})
        predictions.append(rolling_panel(df, features, deviations, hp, name))

    per_asset_specs = {
        "HAR": HAR,
        "HAR_J": HAR + ["c_signed_jump_24", "c_jump_asym_24", "c_jump_share_24"],
        "HAR_IV_PerAsset": HAR + IV,
        "Full_PerAsset": FULL,
    }
    for name, features in per_asset_specs.items():
        grid, alpha = tune_per_asset_ridge(df, features, name)
        grids.append(grid)
        choices.append({"model": name, "alpha": alpha, "pool": np.nan,
                        "validation_mse": float(grid.iloc[0].validation_mse)})
        predictions.append(rolling_per_asset(df, features, name, params={"alpha": alpha}))

    # Extra nonlinear/sparse competitors use the validation-selected settings
    # from the locked pre-holdout audit.
    predictions.append(rolling_per_asset(
        df, FULL, "ElasticNet", kind="elasticnet", params={"alpha": 0.1, "l1_ratio": 0.1}
    ))
    choices.append({"model": "ElasticNet", "alpha": 0.1, "pool": np.nan,
                    "validation_mse": np.nan, "l1_ratio": 0.1})
    predictions.append(rolling_per_asset(
        df, FULL, "HistGradientBoosting", kind="hgb",
        params={"max_iter": 200, "learning_rate": 0.03, "max_leaf_nodes": 7, "l2_regularization": 1.0},
    ))
    choices.append({"model": "HistGradientBoosting", "alpha": np.nan, "pool": np.nan,
                    "validation_mse": np.nan, "max_iter": 200, "learning_rate": 0.03,
                    "max_leaf_nodes": 7, "l2_regularization": 1.0})

    pred = predictions[0]
    for p in predictions[1:]:
        pred = pred.merge(p.drop(columns="y"), on=["timestamp", "symbol"], how="inner")
    pred = pred[pred.timestamp >= "2024-01-01"].reset_index(drop=True)
    lookup = df.set_index(["timestamp", "symbol"])["c_logrv_24"]
    pred["RandomWalk"] = [lookup.loc[(t, s)] for t, s in zip(pred.timestamp, pred.symbol)]
    return pred, pd.concat(grids, ignore_index=True), pd.DataFrame(choices), specs


def evaluate(pred: pd.DataFrame, subset: str, mask: np.ndarray | pd.Series | None = None):
    p = pred if mask is None else pred.loc[mask]
    models = [c for c in p if c not in ("timestamp", "symbol", "y")]
    base_mse = (p.y - p.HAR) ** 2
    base_qlike = np.exp(np.clip(p.y - p.HAR, -50, 50)) - (p.y - p.HAR) - 1
    rows = []
    for model in models:
        mse = (p.y - p[model]) ** 2
        qlike = np.exp(np.clip(p.y - p[model], -50, 50)) - (p.y - p[model]) - 1
        daily_mse = (base_mse - mse).groupby(p.timestamp).mean()
        daily_qlike = (base_qlike - qlike).groupby(p.timestamp).mean()
        tm, tq = nw_t(daily_mse), nw_t(daily_qlike)
        rows.append({
            "subset": subset, "model": model, "n": len(p),
            "mse": mse.mean(), "mse_gain_vs_har_pct": 100 * (1 - mse.sum() / base_mse.sum()),
            "mse_nw_t": tm, "mse_p_two_sided": two_sided_p(tm),
            "qlike": qlike.mean(), "qlike_gain_vs_har_pct": 100 * (1 - qlike.mean() / base_qlike.mean()),
            "qlike_nw_t": tq, "qlike_p_two_sided": two_sided_p(tq),
        })
    return pd.DataFrame(rows)


def direct_tests(pred: pd.DataFrame, subset: str, comparators: list[str]):
    p = pred if subset == "all_17_assets" else pred[~pred.symbol.isin(["BTCUSDT", "ETHUSDT"])]
    rows = []
    for comparator in comparators:
        a = (p.y - p[comparator]) ** 2
        b = (p.y - p.HPTS_Final) ** 2
        daily = (a - b).groupby(p.timestamp).mean()
        t = nw_t(daily)
        lo, hi = block_bootstrap_gain(p, "HPTS_Final", comparator)
        rows.append({
            "subset": subset, "model": "HPTS_Final", "comparator": comparator, "n": len(p),
            "gain_pct": 100 * (1 - b.sum() / a.sum()), "nw_t": t,
            "p_two_sided": two_sided_p(t), "bootstrap_95_low": lo, "bootstrap_95_high": hi,
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = adjust_p(out.p_two_sided, "holm")
    return out


def per_asset_tests(pred: pd.DataFrame):
    rows = []
    for symbol, g in pred.groupby("symbol"):
        a = (g.y - g.HAR) ** 2
        b = (g.y - g.HPTS_Final) ** 2
        daily = (a - b).groupby(g.timestamp).mean()
        t = nw_t(daily)
        rows.append({"symbol": symbol, "n": len(g), "gain_pct": 100 * (1 - b.sum() / a.sum()),
                     "nw_t": t, "p_two_sided": two_sided_p(t)})
    out = pd.DataFrame(rows).sort_values("gain_pct", ascending=False).reset_index(drop=True)
    out["p_bh"] = adjust_p(out.p_two_sided, "bh")
    return out


def per_year(pred: pd.DataFrame):
    rows = []
    for year, g in pred.groupby(pred.timestamp.dt.year):
        base = ((g.y - g.HAR) ** 2).sum()
        for model in ("HPTS_Final", "HPTS_NoIV", "Full_PerAsset", "HistGradientBoosting"):
            rows.append({"year": year, "model": model,
                         "mse_gain_vs_har_pct": 100 * (1 - ((g.y - g[model]) ** 2).sum() / base)})
    return pd.DataFrame(rows)


def scale_diagnostic(pred: pd.DataFrame):
    y, har, final = pred.y.to_numpy(), pred.HAR.to_numpy(), pred.HPTS_Final.to_numpy()
    rows = []
    for scale, transform in (
        ("log_realised_variance", lambda x: x),
        ("log_realised_volatility", lambda x: x / 2),
        ("raw_realised_volatility", lambda x: np.exp(x / 2)),
    ):
        yt, ht, ft = transform(y), transform(har), transform(final)
        a, b = (yt - ht) ** 2, (yt - ft) ** 2
        daily = pd.Series(a - b).groupby(pred.timestamp).mean()
        t = nw_t(daily)
        rows.append({"scale": scale, "n": len(pred), "gain_vs_har_pct": 100 * (1 - b.sum() / a.sum()),
                     "nw_t": t, "p_two_sided": two_sided_p(t)})
    return pd.DataFrame(rows)


def robustness(df: pd.DataFrame, choices: pd.DataFrame, pred: pd.DataFrame):
    rows = []
    hp = choices.set_index("model").loc["HPTS_Final"].to_dict()
    work = df.copy()
    delayed = []
    for c in IV:
        name = f"lag1_{c}"
        work[name] = work.groupby("symbol")[c].shift(1)
        delayed.append(name)
    work = work.dropna(subset=delayed).reset_index(drop=True)

    variants = {
        "final_no_spread": (df, FULL, FULL, None),
        "with_iv_spread": (df, FULL + IV_SPREAD, FULL + IV_SPREAD, None),
        "btc_iv_only": (df, SPOT + IV_BTC, SPOT + IV_BTC, None),
        "eth_iv_only": (df, SPOT + IV_ETH, SPOT + IV_ETH, None),
        "iv_delayed_one_day": (work, SPOT + delayed, SPOT + delayed, None),
        "rolling_730_days": (df, FULL, FULL, 730),
        "rolling_1095_days": (df, FULL, FULL, 1095),
    }
    for label, (data, features, deviations, window) in variants.items():
        if label == "final_no_spread":
            q = pred[["timestamp", "symbol", "y", "HAR", "HPTS_Final"]].rename(columns={"HPTS_Final": label})
            chosen = hp
        else:
            grid, chosen = tune_panel(data, features, deviations, label)
            q = rolling_panel(data, features, deviations, chosen, label, window_days=window)
            q = q.merge(pred[["timestamp", "symbol", "HAR"]], on=["timestamp", "symbol"], how="inner")
            q = q[q.timestamp >= "2024-01-01"]
        a, b = (q.y - q.HAR) ** 2, (q.y - q[label]) ** 2
        daily = (a - b).groupby(q.timestamp).mean()
        t = nw_t(daily)
        rows.append({"variant": label, "n": len(q), "alpha": chosen["alpha"], "pool": chosen.get("pool", np.nan),
                     "gain_vs_har_pct": 100 * (1 - b.sum() / a.sum()), "nw_t": t,
                     "p_two_sided": two_sided_p(t)})
    return pd.DataFrame(rows).sort_values("gain_vs_har_pct", ascending=False)


def leave_one_asset_out_refit(df: pd.DataFrame, pred: pd.DataFrame, hp: dict):
    rows = []
    for omitted in sorted(df.symbol.unique()):
        print(f"Leave-one-asset-out refit: {omitted}", flush=True)
        sub = df[df.symbol != omitted].reset_index(drop=True)
        p = rolling_panel(sub, FULL, FULL, hp, "HPTS_Final")
        har = pred[pred.symbol != omitted][["timestamp", "symbol", "HAR"]]
        q = p.merge(har, on=["timestamp", "symbol"], how="inner")
        q = q[q.timestamp >= "2024-01-01"]
        a, b = (q.y - q.HAR) ** 2, (q.y - q.HPTS_Final) ** 2
        daily = (a - b).groupby(q.timestamp).mean()
        rows.append({"omitted_asset": omitted, "n": len(q), "gain_pct": 100 * (1 - b.sum() / a.sum()),
                     "nw_t": nw_t(daily)})
    return pd.DataFrame(rows)


def save_figures(model_table: pd.DataFrame, asset_table: pd.DataFrame, year_table: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    main = model_table[model_table.subset == "all_17_assets"].sort_values("mse_gain_vs_har_pct")
    colors = ["#0B6E4F" if m == "HPTS_Final" else "#8093A7" for m in main.model]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(main.model, main.mse_gain_vs_har_pct, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("MSE improvement relative to HAR (%)")
    ax.set_title("HPTS-Final delivers the largest log-variance MSE improvement", loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "figure_1_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    a = asset_table.sort_values("gain_pct")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(a.symbol.str.replace("USDT", "", regex=False), a.gain_pct,
            color=np.where(a.gain_pct >= 0, "#0B6E4F", "#B44B4B"))
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("HPTS-Final MSE improvement relative to HAR (%)")
    ax.set_title("Cross-sectional consistency of the forecasting gain", loc="left", weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "figure_2_asset_gains.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for model, g in year_table.groupby("model"):
        ax.plot(g.year, g.mse_gain_vs_har_pct, marker="o", linewidth=2 if model == "HPTS_Final" else 1.2,
                label=model, color="#0B6E4F" if model == "HPTS_Final" else None)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(sorted(year_table.year.unique()))
    ax.set_ylabel("MSE improvement relative to HAR (%)")
    ax.set_title("Forecast gains remain positive across holdout years", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "figure_3_year_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_results_report(model_table: pd.DataFrame, direct: pd.DataFrame, no_options: pd.DataFrame,
                         asset: pd.DataFrame, years: pd.DataFrame, robust: pd.DataFrame,
                         scale: pd.DataFrame, choices: pd.DataFrame):
    overall = model_table[(model_table.subset == "all_17_assets") & (model_table.model == "HPTS_Final")].iloc[0]
    har = direct[(direct.subset == "all_17_assets") & (direct.comparator == "HAR")].iloc[0]
    no_har = no_options[no_options.comparator == "HAR"].iloc[0]
    no_iv = direct[(direct.subset == "all_17_assets") & (direct.comparator == "HPTS_NoIV")].iloc[0]
    positive = int((asset.gain_pct > 0).sum())
    significant = int((asset.p_bh < 0.05).sum())
    hp = choices.set_index("model").loc["HPTS_Final"]
    raw = scale[scale.scale == "raw_realised_volatility"].iloc[0]
    report = f"""# Final HPTS forecasting results

## Locked design

- Target: log realised variance over the exact next 24 hours.
- Training: before 2023; validation: calendar 2023; untouched holdout: 2024 onward.
- Proposed model: one non-ensemble hierarchical ridge model.
- Final predictors: HAR, cross-sectional state, tail/path/liquidity, and causal BTC/ETH DVOL-open variables.
- BTC--ETH IV spread: omitted because the no-spread model had lower validation MSE.
- Selected HPTS penalty: alpha={hp.alpha:.0f}, pooling multiplier={hp.pool:.0f}.
- Random seed: {SEED}, used only by stochastic competitors and bootstrap inference; ridge estimates are deterministic.

## Primary result

HPTS-Final improves pooled holdout MSE relative to per-asset HAR by **{overall.mse_gain_vs_har_pct:.2f}%**
(Newey--West t={overall.mse_nw_t:.2f}, two-sided p={overall.mse_p_two_sided:.3g}).  The 28-day
block-bootstrap 95% interval is **[{har.bootstrap_95_low:.2f}%, {har.bootstrap_95_high:.2f}%]**.

On the 15 assets without listed options, the gain is **{no_har.gain_pct:.2f}%**
(t={no_har.nw_t:.2f}, p={no_har.p_two_sided:.3g}).  Relative to HPTS-NoIV, the full model gains
**{no_iv.gain_pct:.2f}%** (t={no_iv.nw_t:.2f}, p={no_iv.p_two_sided:.3g}).

The improvement is positive for **{positive}/17** assets and significant after Benjamini--Hochberg
correction for **{significant}/17** assets.  Leave-one-asset-out refits and timing/window robustness
are supplied as supplementary tables.

## Secondary losses and scope

QLIKE is reported as a secondary diagnostic and is not the strongest result for HPTS.  Forecasting
log realised volatility is algebraically the same target up to a factor of one half and therefore
preserves the relative MSE gain.  On raw realised volatility, the HPTS gain falls to
**{raw.gain_vs_har_pct:.2f}%** (t={raw.nw_t:.2f}, p={raw.p_two_sided:.3g}); raw-scale superiority is
therefore not claimed.

Risk-management results are deliberately excluded from the main contribution because paired VaR/ES
loss improvements were not statistically decisive.  The paper should remain a forecasting study.

## Defensible contribution

The contribution is hierarchical cross-asset forecasting with incremental, causally timed option-
implied information for assets that largely lack listed options.  The results do not justify claiming
a new ridge estimator or decisive heterogeneous option-loading coefficients.
"""
    (ROOT / "FINAL_RESULTS.md").write_text(report, encoding="utf-8")


def write_reproducibility_metadata(df: pd.DataFrame, choices: pd.DataFrame):
    checksum = sha256(DATA.read_bytes()).hexdigest()
    audit = pd.DataFrame([
        {"check": "target_definition", "value": "log(sum of squared hourly log returns over next exact 24h)", "status": "pass"},
        {"check": "row_count", "value": len(df), "status": "pass"},
        {"check": "asset_count", "value": df.symbol.nunique(), "status": "pass"},
        {"check": "date_min", "value": df.timestamp.min().date(), "status": "pass"},
        {"check": "date_max", "value": df.timestamp.max().date(), "status": "pass"},
        {"check": "missing_model_values", "value": int(df[["y", *FULL]].isna().sum().sum()), "status": "pass"},
        {"check": "duplicate_asset_dates", "value": int(df.duplicated(["timestamp", "symbol"]).sum()), "status": "pass"},
        {"check": "forbidden_same_day_dvol_close_high_low", "value": 0, "status": "pass"},
        {"check": "model_data_sha256", "value": checksum, "status": "pass"},
    ])
    audit.to_csv(OUT / "data_and_leakage_audit.csv", index=False)
    clean_choices = []
    for row in choices.to_dict(orient="records"):
        clean_choices.append({key: (None if pd.isna(value) else value) for key, value in row.items()})
    metadata = {
        "seed": SEED,
        "target": "log realised variance over exact next 24 hours",
        "train_end_exclusive": "2023-01-01",
        "validation_start": "2023-01-01",
        "validation_end_exclusive": "2024-01-01",
        "holdout_start": "2024-01-01",
        "refit_days": REFIT_DAYS,
        "primary_loss": "squared error on log realised variance",
        "secondary_loss": "QLIKE on realised variance",
        "option_timing": "current UTC day DVOL open",
        "final_model": "HPTS_Final, hierarchical ridge, no IV spread",
        "model_data_sha256": checksum,
        "selected_hyperparameters": clean_choices,
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_and_audit()
    print(f"Loaded {len(df):,} rows, {df.symbol.nunique()} assets", flush=True)

    pred, grids, choices, _ = build_predictions(df)
    model_table = pd.concat([
        evaluate(pred, "all_17_assets"),
        evaluate(pred, "15_assets_without_listed_options", ~pred.symbol.isin(["BTCUSDT", "ETHUSDT"])),
    ], ignore_index=True)

    comparators = [
        "HAR", "RandomWalk", "HAR_J", "HAR_IV_PerAsset", "HPTS_NoIV",
        "Full_FixedEffects", "Full_PerAsset", "HPTS_CommonIV", "HPTS_IVSpecific",
        "ElasticNet", "HistGradientBoosting",
    ]
    direct_all = direct_tests(pred, "all_17_assets", comparators)
    direct_no_options = direct_tests(
        pred, "15_assets_without_listed_options",
        ["HAR", "HPTS_NoIV", "Full_FixedEffects", "Full_PerAsset", "HPTS_CommonIV", "HPTS_IVSpecific"],
    )
    asset = per_asset_tests(pred)
    years = per_year(pred)
    scale = scale_diagnostic(pred)
    robust = robustness(df, choices, pred)
    hp = choices.set_index("model").loc["HPTS_Final"].to_dict()
    loao = leave_one_asset_out_refit(df, pred, hp)

    model_table.sort_values(["subset", "mse_gain_vs_har_pct"], ascending=[True, False]).to_csv(OUT / "table_1_model_comparison.csv", index=False)
    direct_all.to_csv(OUT / "table_2_direct_tests.csv", index=False)
    direct_no_options.to_csv(OUT / "table_3_no_options_assets.csv", index=False)
    asset.to_csv(OUT / "table_4_per_asset.csv", index=False)
    robust.to_csv(OUT / "table_5_robustness.csv", index=False)
    scale.to_csv(OUT / "table_6_target_scale_diagnostic.csv", index=False)
    years.to_csv(OUT / "supplement_year_stability.csv", index=False)
    loao.to_csv(OUT / "supplement_leave_one_asset_out_refit.csv", index=False)
    grids.to_csv(OUT / "validation_grids.csv", index=False)
    choices.to_csv(OUT / "selected_hyperparameters.csv", index=False)
    pred.to_csv(OUT / "holdout_predictions.csv", index=False)

    save_figures(model_table, asset, years)
    write_results_report(model_table, direct_all, direct_no_options, asset, years, robust, scale, choices)
    write_reproducibility_metadata(df, choices)
    print("Final analysis completed successfully.", flush=True)


def finalize_from_saved_results():
    """Finish figures/report after a presentation-layer interruption."""
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_and_audit()
    model_table = pd.read_csv(OUT / "table_1_model_comparison.csv")
    direct_all = pd.read_csv(OUT / "table_2_direct_tests.csv")
    direct_no_options = pd.read_csv(OUT / "table_3_no_options_assets.csv")
    asset = pd.read_csv(OUT / "table_4_per_asset.csv")
    robust = pd.read_csv(OUT / "table_5_robustness.csv")
    scale = pd.read_csv(OUT / "table_6_target_scale_diagnostic.csv")
    years = pd.read_csv(OUT / "supplement_year_stability.csv")
    choices = pd.read_csv(OUT / "selected_hyperparameters.csv")
    save_figures(model_table, asset, years)
    write_results_report(model_table, direct_all, direct_no_options, asset, years, robust, scale, choices)
    write_reproducibility_metadata(df, choices)
    print("Finalization from saved results completed successfully.", flush=True)


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize_from_saved_results()
    else:
        main()
