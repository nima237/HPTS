from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT
OUT = ROOT / "results" / "benchmarks"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA

import analysis_main as core

warnings.filterwarnings("ignore")

SEED = 77
REPS = 1000
RAW_PANEL = Path(os.environ.get("HPTS_RAW_PANEL", "panel_hourly.parquet"))
PURE_IV = [
    "log_dvol_open", "dvol_chg_1_open", "dvol_chg_5_open",
    "log_dvol_eth_open", "dvol_eth_chg_1_open",
]


def nw_t(values, lags=4):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    e = x - x.mean()
    n = len(x)
    v = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag] / n)
    return float(x.mean() / np.sqrt(max(v, 1e-18) / n))


def score(frame, model, comparator="HAR", horizon_days=1):
    a = (frame.y - frame[comparator]) ** 2
    b = (frame.y - frame[model]) ** 2
    daily = (a - b).groupby(frame.timestamp).mean()
    lag = max(4, horizon_days - 1)
    t = nw_t(daily, lag)
    return {
        "model": model,
        "comparator": comparator,
        "n": len(frame),
        "mse": float(b.mean()),
        "gain_pct": float(100 * (1 - b.sum() / a.sum())),
        "nw_lags": lag,
        "nw_t": t,
        "p_two_sided": float(2 * norm.sf(abs(t))),
    }


def repeated_permutation():
    df = core.load_and_audit()
    base = pd.read_csv(ROOT / "results" / "holdout_predictions.csv",
                       parse_dates=["timestamp"])
    hold = df[df.timestamp >= "2024-01-01"].copy().reset_index(drop=True)
    hold = hold.merge(base[["timestamp", "symbol", "y", "HAR", "HPTS_Final", "HPTS_NoIV"]],
                      on=["timestamp", "symbol"], suffixes=("", "_saved"), how="inner")
    dates = np.array(sorted(hold.timestamp.unique()))
    date_id = pd.Categorical(hold.timestamp, categories=dates, ordered=True).codes
    symbols = sorted(df.symbol.unique())
    symbol_id = pd.Categorical(hold.symbol, categories=symbols, ordered=True).codes.astype(int)
    pure_cube = np.empty((len(dates), len(symbols), len(PURE_IV)))
    dated = hold.set_index(["timestamp", "symbol"])
    for j, col in enumerate(PURE_IV):
        btc = dated[col].unstack("symbol").reindex(dates)["BTCUSDT"].to_numpy(float)
        eth = dated[col].unstack("symbol").reindex(dates)["ETHUSDT"].to_numpy(float)
        pure_cube[:, :, j] = btc[:, None]
        if col in PURE_IV[:3]:
            pure_cube[:, symbols.index("ETHUSDT"), j] = eth

    correct = hold.HPTS_Final.to_numpy(float)
    y = hold.y_saved.to_numpy(float)
    noiv = hold.HPTS_NoIV.to_numpy(float)
    effects = np.zeros((len(hold), len(core.IV)))
    correct_std = np.zeros_like(effects)

    hp = {"alpha": 100.0, "pool": 10.0}
    block_dates = dates
    for start in range(0, len(block_dates), core.REFIT_DAYS):
        block = block_dates[start:start + core.REFIT_DAYS]
        cutoff = block[0]
        train = df[df.timestamp < cutoff].reset_index(drop=True)
        test_idx = np.flatnonzero(hold.timestamp.isin(block).to_numpy())
        test = hold.iloc[test_idx].copy().reset_index(drop=True)
        x, z, penalty = core.panel_design(train, test, core.FULL, core.FULL,
                                          hp["alpha"], hp["pool"])
        beta = np.linalg.solve(x.T @ x + np.diag(penalty + 1e-10),
                               x.T @ train.y.to_numpy(float))
        means = train[core.FULL].mean().to_numpy(float)
        stds = train[core.FULL].std(ddof=0).replace(0, 1).to_numpy(float)
        ns = len(symbols)
        dev0 = 1 + len(core.FULL) + ns
        for j, col in enumerate(core.IV):
            k = core.FULL.index(col)
            effective = beta[1 + k] + beta[dev0 + k * ns + symbol_id[test_idx]]
            effects[test_idx, j] = effective / stds[k]
            correct_std[test_idx, j] = test[col].to_numpy(float)

    rng = np.random.default_rng(SEED)
    observed_gain = 100 * (1 - np.sum((y - correct) ** 2) / np.sum((y - noiv) ** 2))
    perm_gain = np.empty(REPS)
    current_vs_perm = np.empty(REPS)
    correct_loss = np.sum((y - correct) ** 2)
    noiv_loss = np.sum((y - noiv) ** 2)
    for rep in range(REPS):
        order = rng.permutation(len(dates))
        vals = pure_cube[order[date_id], symbol_id, :]
        perm = np.column_stack([
            vals[:, 0], vals[:, 1], vals[:, 2],
            vals[:, 0] - hold.c_logrv_24.to_numpy(float), vals[:, 3], vals[:, 4],
        ])
        pred = correct + np.sum(effects * (perm - correct_std), axis=1)
        loss = np.sum((y - pred) ** 2)
        perm_gain[rep] = 100 * (1 - loss / noiv_loss)
        current_vs_perm[rep] = 100 * (1 - correct_loss / loss)

    summary = {
        "seed": SEED,
        "permutations": REPS,
        "test": "locked-model joint-date permutation of the public IV state on the untouched holdout",
        "observed_HPTS_Final_gain_vs_NoIV_pct": observed_gain,
        "permuted_gain_vs_NoIV_mean_pct": float(perm_gain.mean()),
        "permuted_gain_vs_NoIV_q025_pct": float(np.quantile(perm_gain, .025)),
        "permuted_gain_vs_NoIV_q975_pct": float(np.quantile(perm_gain, .975)),
        "permutation_p_one_sided": float((1 + np.sum(perm_gain >= observed_gain)) / (REPS + 1)),
        "current_gain_vs_permuted_mean_pct": float(current_vs_perm.mean()),
        "current_gain_vs_permuted_q025_pct": float(np.quantile(current_vs_perm, .025)),
        "current_gain_vs_permuted_q975_pct": float(np.quantile(current_vs_perm, .975)),
    }
    pd.DataFrame({"rep": np.arange(REPS), "permuted_gain_vs_NoIV_pct": perm_gain,
                  "current_gain_vs_permuted_pct": current_vs_perm}).to_csv(
        OUT / "permutation_draws.csv", index=False)
    (OUT / "permutation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


FRAC_LAG = 200          # fixed truncation lag of the fractional filter


def fractional_weights(d, length=FRAC_LAG):
    """Truncated binomial expansion of (1-B)^d, fixed length so that the same
    weight vector defines the transform and its inversion at every refit."""
    w = np.empty(length + 1)
    w[0] = 1.0
    for k in range(1, length + 1):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return w


def gph_d(x):
    x = np.asarray(x, float)
    n = len(x)
    m = max(12, int(n ** .55))
    ft = np.fft.fft(x - x.mean())
    periodogram = np.abs(ft) ** 2 / (2 * np.pi * n)
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    slope = np.polyfit(np.log(4 * np.sin(lam / 2) ** 2),
                       np.log(np.maximum(periodogram[1:m + 1], 1e-12)), 1)[0]
    return float(np.clip(-slope, 0.0, .49))


def arma_asset(g, fractional=False):
    x = g.c_logrv_24.to_numpy(float)
    dates = g.timestamp.to_numpy()
    out = []
    fit = None
    raw_seen = []
    weights = None
    last_fit = -10_000
    for i in range(len(g) - 1):
        date = pd.Timestamp(dates[i])
        raw_seen.append(x[i])
        if date < pd.Timestamp("2023-01-01"):
            continue
        refit = fit is None or i - last_fit >= 60
        try:
            if refit:
                history = np.asarray(raw_seen)
                if fractional:
                    if len(history) < 2 * FRAC_LAG:
                        raise ValueError("history too short for a stable fractional filter")
                    d = gph_d(history)
                    weights = fractional_weights(d)
                    # z[t] = sum_{k=0}^{FRAC_LAG} w[k] x[t-k], defined for t >= FRAC_LAG
                    z = np.convolve(history, weights)[:len(history)][FRAC_LAG:]
                    fit = ARIMA(z, order=(1, 0, 1), trend="c").fit()
                else:
                    fit = ARIMA(history, order=(1, 0, 1), trend="c").fit()
                last_fit = i
            elif fractional:
                lag = np.asarray(raw_seen[-(FRAC_LAG + 1):])[::-1]
                fit = fit.append([float(np.dot(weights[:len(lag)], lag))], refit=False)
            else:
                fit = fit.append([x[i]], refit=False)
            fc = float(fit.forecast(1)[0])
            if fractional:
                lag = np.asarray(raw_seen[-FRAC_LAG:])[::-1]
                fc -= float(np.dot(weights[1:len(lag) + 1], lag))
        except Exception:
            fc = float(np.mean(raw_seen[-60:]))
            fit = None
        out.append((date, g.symbol.iloc[0], x[i + 1], fc))
    return out


def garch_asset(g):
    r = 100 * g.ret_24.to_numpy(float)
    x = g.c_logrv_24.to_numpy(float)
    dates = g.timestamp.to_numpy()
    rows = []
    params = None
    variance = None
    last_fit = -10_000
    for i in range(len(g) - 1):
        date = pd.Timestamp(dates[i])
        if date < pd.Timestamp("2023-01-01"):
            continue
        try:
            if params is None or i - last_fit >= 60:
                res = arch_model(r[:i + 1], mean="Zero", vol="GARCH", p=1, q=1,
                                 dist="t", rescale=False).fit(disp="off", show_warning=False)
                params = res.params
                variance = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])
                last_fit = i
            else:
                variance = (float(params["omega"]) + float(params["alpha[1]"]) * r[i] ** 2
                            + float(params["beta[1]"]) * variance)
            fc = float(np.log(max(variance / 10000, 1e-12)))
        except Exception:
            fc = float(np.mean(x[max(0, i - 60):i + 1]))
            params = None
        rows.append((date, g.symbol.iloc[0], x[i + 1], fc))
    return rows


def baseline_predictions():
    df = core.load_and_audit().sort_values(["symbol", "timestamp"])
    frames = {}
    for name, func in [("ARMA_1_1", lambda g: arma_asset(g, False)),
                       ("ARFIMA_1_d_1", lambda g: arma_asset(g, True)),
                       ("GARCH_1_1_t", garch_asset)]:
        rows = []
        for symbol, g in df.groupby("symbol"):
            print(f"{name}: {symbol}", flush=True)
            rows.extend(func(g.reset_index(drop=True)))
        frames[name] = pd.DataFrame(rows, columns=["timestamp", "symbol", "y_check", name])

    saved = pd.read_csv(ROOT / "results" / "holdout_predictions.csv",
                        parse_dates=["timestamp"])
    merged = None
    calibration_rows = []
    for name, frame in frames.items():
        val = frame[(frame.timestamp >= "2023-01-01") & (frame.timestamp < "2024-01-01")]
        X = np.column_stack([np.ones(len(val)), val[name].to_numpy(float)])
        coef = np.linalg.lstsq(X, val.y_check.to_numpy(float), rcond=None)[0]
        frame[name] = coef[0] + coef[1] * frame[name]
        calibration_rows.append({"model": name, "intercept": coef[0], "slope": coef[1],
                                 "calibration_start": "2023-01-01", "calibration_end": "2023-12-31"})
        hold = frame[frame.timestamp >= "2024-01-01"][["timestamp", "symbol", name]]
        merged = hold if merged is None else merged.merge(hold, on=["timestamp", "symbol"], how="inner")
    merged = saved.merge(merged, on=["timestamp", "symbol"], how="inner")
    rows = [score(merged, "HPTS_Final")]
    for name in frames:
        rows.append(score(merged, name))
        rows.append(score(merged, "HPTS_Final", name))
    pd.DataFrame(rows).to_csv(OUT / "hoang_baur_baselines.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT / "baseline_validation_calibration.csv", index=False)
    merged[["timestamp", "symbol", "y", "HAR", "HPTS_Final", *frames]].to_csv(
        OUT / "baseline_holdout_predictions.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def horizon_data(days):
    df = core.load_and_audit().copy()
    if days == 1:
        df["target_end"] = df.timestamp + pd.Timedelta(days=1)
        return df
    target = {3: "y_3d", 7: "y_7d"}[days]
    if target in df:
        df = df.drop(columns="y").rename(columns={target: "y"})
    else:
        col = {3: "c_logrv_72", 7: "c_logrv_168"}[days]
        if not RAW_PANEL.exists():
            raise FileNotFoundError(
                f"{target} is absent from model_data.csv and the optional raw panel was not found at "
                f"{RAW_PANEL}. Set HPTS_RAW_PANEL to a compatible panel_hourly.parquet file."
            )
        raw = pd.read_parquet(RAW_PANEL, columns=["timestamp", "symbol", col])
        future = raw[raw.timestamp.dt.hour.eq(0)][["timestamp", "symbol", col]].copy()
        future["timestamp"] -= pd.Timedelta(days=days)
        future = future.rename(columns={col: "y_h"})
        df = df.drop(columns="y").merge(future, on=["timestamp", "symbol"], how="inner")
        df = df.rename(columns={"y_h": "y"})
    df["target_end"] = df.timestamp + pd.Timedelta(days=days)
    return df.dropna(subset=["y"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def rolling_panel_gap(df, features, deviations, hp, name, days):
    rows = []
    dates = np.array(sorted(df[df.timestamp >= "2024-01-01"].timestamp.unique()))
    for start in range(0, len(dates), core.REFIT_DAYS):
        block = dates[start:start + core.REFIT_DAYS]
        cutoff = pd.Timestamp(block[0])
        train = df[df.target_end <= cutoff].reset_index(drop=True)
        test = df[df.timestamp.isin(block)].reset_index(drop=True)
        pred = core.panel_predict(train, test, features, deviations, hp)
        rows.extend(zip(test.timestamp, test.symbol, test.y, pred))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "y", name])


def rolling_har_gap(df, days):
    rows = []
    dates = np.array(sorted(df[df.timestamp >= "2024-01-01"].timestamp.unique()))
    for start in range(0, len(dates), core.REFIT_DAYS):
        block = dates[start:start + core.REFIT_DAYS]
        cutoff = pd.Timestamp(block[0])
        train = df[df.target_end <= cutoff]
        test = df[df.timestamp.isin(block)]
        for symbol, tr in train.groupby("symbol"):
            te = test[test.symbol == symbol]
            if te.empty:
                continue
            x, z = core.standardise(tr, te, core.HAR)
            y = tr.y.to_numpy(float)
            beta = np.linalg.solve(x.T @ x + .1 * np.eye(x.shape[1]), x.T @ (y - y.mean()))
            pred = y.mean() + z @ beta
            rows.extend(zip(te.timestamp, te.symbol, te.y, pred))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", "y", "HAR"])


def multihorizon():
    all_scores = []
    all_pred = []
    for days in (1, 3, 7):
        print(f"multi-horizon: {days} day(s)", flush=True)
        if days == 1:
            pred = pd.read_csv(ROOT / "results" / "holdout_predictions.csv",
                               parse_dates=["timestamp"])[
                ["timestamp", "symbol", "y", "HAR", "HPTS_NoIV", "HPTS_Final"]]
        else:
            df = horizon_data(days)
            har = rolling_har_gap(df, days)
            noiv = rolling_panel_gap(df, core.SPOT, core.SPOT,
                                     {"alpha": 10.0, "pool": 30.0}, "HPTS_NoIV", days)
            final = rolling_panel_gap(df, core.FULL, core.FULL,
                                     {"alpha": 100.0, "pool": 10.0}, "HPTS_Final", days)
            pred = har.merge(noiv.drop(columns="y"), on=["timestamp", "symbol"], how="inner")
            pred = pred.merge(final.drop(columns="y"), on=["timestamp", "symbol"], how="inner")
        pred["horizon_days"] = days
        all_pred.append(pred)
        for model, comparator in [("HPTS_NoIV", "HAR"), ("HPTS_Final", "HAR"),
                                  ("HPTS_Final", "HPTS_NoIV")]:
            row = score(pred, model, comparator, days)
            row["horizon_days"] = days
            all_scores.append(row)
    pd.DataFrame(all_scores).to_csv(OUT / "multihorizon_scores.csv", index=False)
    pd.concat(all_pred, ignore_index=True).to_csv(OUT / "multihorizon_predictions.csv", index=False)
    print(pd.DataFrame(all_scores).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["permutation", "baselines", "multihorizon", "all"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.task in ("permutation", "all"):
        repeated_permutation()
    if args.task in ("baselines", "all"):
        baseline_predictions()
    if args.task in ("multihorizon", "all"):
        multihorizon()


if __name__ == "__main__":
    main()
