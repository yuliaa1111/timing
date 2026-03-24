"""
Timing system with ridge-based composite score.

This script is adapted from `1.27 final_timing.ipynb` with the same:
1) data import flow for index/open-close and money_1d,
2) open-open backtest method,
3) position construction pipeline (target -> attack mask -> slow/fast mix).

Only the composite score generation is replaced by a pluggable
double-loop ridge engine:
- outer loop: walk-forward rolling training window
- inner loop: time-series CV for alpha selection
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RUNS_DIR = PROJECT_DIR / "runs" / "timing_ridge"


# =========================
# Config
# =========================
@dataclass
class DataConfig:
    index_code: str = "000852.XSHG"  # 中证1000
    start_date: datetime = datetime(2023, 1, 1)
    end_date: datetime = datetime(2025, 6, 30)
    db_settings_path: str = str(DATA_DIR / "db_settings.json")
    db_settings_key: str = "clickhouse_read"
    use_macro_features: bool = True
    path_22: str = str(DATA_DIR / "features_ABC_zz1000_20230101_20250630.parquet")
    path_35: str = str(DATA_DIR / "market_indicators_20200101_20241231_20251109_132634.parquet")
    path_macro: str = str(DATA_DIR / "macro_2014_2026.parquet")


@dataclass
class RidgeConfig:
    # Outer loop (walk-forward)
    outer_train_window: int = 252
    outer_step: int = 1
    min_train_size: int = 180
    label_availability_lag: int = 2
    fill_between_steps: bool = True

    # Inner loop (time-series CV)
    cv_splitter: str = "expanding"  # expanding | rolling
    cv_n_splits: int = 4
    cv_train_ratio: float = 0.6
    cv_min_train_size: int = 60
    cv_min_val_size: int = 20
    cv_gap: int = 0
    cv_metric: str = "rankic"  # rankic | mse

    # Ridge
    alpha_grid: Sequence[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    )
    fit_intercept: bool = True
    max_iter: int = 20000

    # Feature preprocessing
    fill_method: str = "ffill_then_train_mean"  # ffill_then_train_mean | train_mean_only
    standardize: bool = True

    # Composite score post-transform，映射到pos target
    score_transform: str = "zscore_rolling"  # none | zscore_rolling | rank_pct
    score_z_window: int = 252
    score_z_min_periods: int = 60


@dataclass
class StrategyConfig:
    bt_start: str = "2023-07-01"
    bt_end: str = "2025-06-30"
    pos_attack: float = 0.85

    # target_pos mapping params
    target_score_clip_low: float = -3.0
    target_score_clip_high: float = 3.0
    target_pos_floor: float = 0.10
    target_pos_ceil: float = 1.00
    target_full_th: float = 0.70
    target_high_th: float = 0.55
    target_mid_th: float = 0.40
    target_low_th: float = 0.25

    # attack_B params
    attack_q: float = 0.7
    attack_q_win: int = 252
    attack_risk_win: int = 20
    attack_risk_th: float = -0.05
    attack_min_periods: int = 60

    # money step params
    money_z_win: int = 120
    money_clip_z: float = 1.0
    money_step_base: dict = field(
        default_factory=lambda: {
            "step_up_min": 0.030,
            "step_up_max": 0.100,
            "step_up_big": 0.22,
            "step_down_big": 0.50,
            "step_down_small": 0.15,
        }
    )
    money_step_amp: dict = field(
        default_factory=lambda: {
            "step_up_min": 0.020,
            "step_up_max": 0.050,
            "step_up_big": 0.06,
            "step_down_big": -0.10,
            "step_down_small": -0.05,
        }
    )


@dataclass
class TargetConfig:
    # y_t = (open_{t+2}/open_{t+1}) - 1
    y_mode: str = "open_t2_over_t1_minus1"


@dataclass
class OutputConfig:
    run_root: str = str(RUNS_DIR)
    run_name: str | None = None
    save_big_files: bool = True
    save_plots: bool = True
    show_plots: bool = False
    save_results_md: bool = True
    parquet_compression: str = "zstd"


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# =========================
# Utils: data / metrics
# =========================
def ensure_datetime_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out.sort_index()

    candidates = [c for c in out.columns if "date" in c.lower() or "trade" in c.lower()]
    if not candidates:
        raise ValueError(f"[{name}] parquet 既不是 DatetimeIndex，也找不到 date/trade 列")

    col = candidates[0]
    out[col] = pd.to_datetime(out[col])
    return out.set_index(col).sort_index()


def rolling_zscore(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    s = s.astype(float).copy()
    m = s.rolling(window=window, min_periods=min_periods).mean()
    sd = s.rolling(window=window, min_periods=min_periods).std()
    return (s - m) / (sd + 1e-8)


def rankic_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 5:
        return np.nan
    val = spearmanr(y_true[mask], y_pred[mask], nan_policy="omit")[0]
    return float(val) if np.isfinite(val) else np.nan


def metric_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "mse":
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() == 0:
            return np.nan
        return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))
    if metric == "rankic":
        return rankic_score(y_true, y_pred)
    raise ValueError(f"Unsupported cv_metric: {metric}")


def metric_better(new_val: float, best_val: float, metric: str) -> bool:
    if not np.isfinite(new_val):
        return False
    if not np.isfinite(best_val):
        return True
    if metric == "mse":
        return new_val < best_val
    if metric == "rankic":
        return new_val > best_val
    raise ValueError(f"Unsupported cv_metric: {metric}")


def _validate_ridge_config(cfg: RidgeConfig) -> None:
    if cfg.outer_train_window <= 0:
        raise ValueError("outer_train_window must be > 0")
    if cfg.outer_step <= 0:
        raise ValueError("outer_step must be > 0")
    if cfg.cv_n_splits <= 0:
        raise ValueError("cv_n_splits must be > 0")
    if cfg.cv_metric not in {"mse", "rankic"}:
        raise ValueError("cv_metric must be one of {'mse','rankic'}")
    if cfg.cv_splitter not in {"expanding", "rolling"}:
        raise ValueError("cv_splitter must be one of {'expanding','rolling'}")
    if cfg.fill_method not in {"ffill_then_train_mean", "train_mean_only"}:
        raise ValueError("fill_method must be one of {'ffill_then_train_mean','train_mean_only'}")
    if cfg.score_transform not in {"none", "zscore_rolling", "rank_pct"}:
        raise ValueError("score_transform must be one of {'none','zscore_rolling','rank_pct'}")
    if len(cfg.alpha_grid) == 0:
        raise ValueError("alpha_grid cannot be empty")


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def setup_run_artifacts(cfg: AppConfig) -> tuple[str, Path, logging.Logger]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.output.run_name or f"{ts}_{cfg.data.index_code.replace('.', '_')}"
    run_dir = Path(cfg.output.run_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"timing_ridge.{run_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return run_name, run_dir, logger


def save_series_to_parquet(series: pd.Series, path: Path, compression: str = "zstd") -> None:
    series.to_frame().to_parquet(path, compression=compression)


def save_plots(
    res: dict,
    steps_ct: pd.DataFrame,
    run_dir: Path,
    show_plots: bool = False,
) -> dict[str, str]:
    saved = {}
    nav_idx = res["nav_idx"]
    nav_str = res["nav_str"]
    nav_ex = res["nav_ex"]
    pos_bt = res["pos_bt"]

    fig = plt.figure(figsize=(12, 5))
    plt.plot(nav_idx.index, nav_idx, label="Index", linewidth=1.3)
    plt.plot(nav_str.index, nav_str, label="Strategy", linewidth=1.6)
    plt.title("Index vs Strategy NAV")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    p = run_dir / "plot_nav.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    saved["plot_nav"] = str(p)
    if show_plots:
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4))
    plt.plot(nav_ex.index, nav_ex, label="Excess NAV", linewidth=1.3)
    plt.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.6)
    plt.title("Excess NAV")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    p = run_dir / "plot_excess_nav.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    saved["plot_excess_nav"] = str(p)
    if show_plots:
        plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(12, 4))
    plt.step(pos_bt.index, pos_bt, where="post", label="Position", linewidth=1.2)
    plt.ylim(0, 1.05)
    plt.title("Position (Backtest Window)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    p = run_dir / "plot_position.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    saved["plot_position"] = str(p)
    if show_plots:
        plt.show()
    plt.close(fig)

    tail_start = pd.Timestamp("2025-02-01")
    tail_end = pd.Timestamp("2025-06-30")
    pos_tail = pos_bt.loc[tail_start:tail_end]
    fig = plt.figure(figsize=(12, 4))
    plt.step(pos_tail.index, pos_tail, where="post", label="Position Tail", linewidth=1.2)
    plt.ylim(0, 1.05)
    plt.title("Tail Position (2025-02-01 ~ 2025-06-30)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    p = run_dir / "plot_tail_position.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    saved["plot_tail_position"] = str(p)
    if show_plots:
        plt.show()
    plt.close(fig)

    liqz_bt = steps_ct["liq_z"].reindex(pos_bt.index)
    fig = plt.figure(figsize=(12, 3))
    plt.plot(liqz_bt.index, liqz_bt, label="liq_z", linewidth=1.1)
    plt.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)
    plt.title("Liquidity z (Backtest Window)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    p = run_dir / "plot_liq_z.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    saved["plot_liq_z"] = str(p)
    if show_plots:
        plt.show()
    plt.close(fig)

    return saved


def write_results_md(
    run_dir: Path,
    run_name: str,
    cfg: AppConfig,
    summary: dict,
    file_manifest: dict[str, str],
) -> None:
    bt_metrics = summary["bt_metrics"]
    lines = [
        "# timing_ridge Results",
        "",
        f"- run_id: `{run_name}`",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- index_code: `{cfg.data.index_code}`",
        f"- use_macro_features: `{cfg.data.use_macro_features}`",
        f"- bt_window: `{cfg.strategy.bt_start}` ~ `{cfg.strategy.bt_end}`",
        "",
        "## Backtest Metrics",
        "",
        f"- Strategy AnnRet: `{bt_metrics.get('AnnRet_Strategy')}`",
        f"- Strategy AnnVol: `{bt_metrics.get('AnnVol_Strategy')}`",
        f"- Strategy Sharpe: `{bt_metrics.get('Sharpe_Strategy')}`",
        f"- Strategy MaxDD: `{bt_metrics.get('MaxDD_Strategy')}`",
        f"- Strategy Hit: `{bt_metrics.get('Hit_Strategy')}`",
        f"- FullShare(bt): `{summary.get('full_share_bt')}`",
        f"- FastShare(bt): `{summary.get('fast_share_bt')}`",
        f"- Turnover(bt): `{summary.get('turnover_bt')}`",
        "",
        "## Files",
        "",
    ]

    for k in sorted(file_manifest.keys()):
        lines.append(f"- `{k}`: `{file_manifest[k]}`")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- 所有核心结果、日志与图像已落盘。")
    lines.append("- 本次配置快照见 `config_used.json`。")
    lines.append("- 详细训练窗口诊断见 `ridge_diag.csv`。")

    (run_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")

# =========================
# Inner CV splitter
# =========================
def generate_time_cv_splits(n_samples: int, cfg: RidgeConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_samples < cfg.cv_min_train_size + cfg.cv_min_val_size:
        return []

    base_train = max(cfg.cv_min_train_size, int(round(n_samples * cfg.cv_train_ratio)))
    base_train = min(base_train, n_samples - cfg.cv_min_val_size)
    if base_train < cfg.cv_min_train_size:
        return []

    usable = n_samples - base_train - cfg.cv_gap
    if usable < cfg.cv_min_val_size:
        return []

    n_splits = min(cfg.cv_n_splits, max(1, usable // cfg.cv_min_val_size))
    val_size = max(cfg.cv_min_val_size, usable // n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    if cfg.cv_splitter == "expanding":
        for k in range(n_splits):
            tr_end = base_train + k * val_size
            val_start = tr_end + cfg.cv_gap
            val_end = min(val_start + val_size, n_samples)
            if val_end - val_start < cfg.cv_min_val_size:
                continue
            tr_idx = np.arange(0, tr_end)
            va_idx = np.arange(val_start, val_end)
            if len(tr_idx) >= cfg.cv_min_train_size:
                splits.append((tr_idx, va_idx))

    else:  # rolling
        train_size = base_train
        for k in range(n_splits):
            tr_end = base_train + k * val_size
            tr_start = max(0, tr_end - train_size)
            val_start = tr_end + cfg.cv_gap
            val_end = min(val_start + val_size, n_samples)
            if val_end - val_start < cfg.cv_min_val_size:
                continue
            tr_idx = np.arange(tr_start, tr_end)
            va_idx = np.arange(val_start, val_end)
            if len(tr_idx) >= cfg.cv_min_train_size:
                splits.append((tr_idx, va_idx))

    return splits


def prepare_train_val(
    x_train_raw: pd.DataFrame,
    x_val_raw: pd.DataFrame,
    cfg: RidgeConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = x_train_raw.copy()
    x_val = x_val_raw.copy()

    if cfg.fill_method == "ffill_then_train_mean":
        x_train = x_train.ffill()
        if len(x_train) > 0:
            last_row = x_train.iloc[[-1]]
            x_val = pd.concat([last_row, x_val], axis=0).ffill().iloc[1:]

    col_mean = x_train.mean(axis=0)
    x_train = x_train.fillna(col_mean)
    x_val = x_val.fillna(col_mean)

    if cfg.standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train_np = scaler.fit_transform(x_train.values)
        x_val_np = scaler.transform(x_val.values)
    else:
        x_train_np = x_train.values
        x_val_np = x_val.values

    return x_train_np, x_val_np


def select_best_alpha_with_time_cv(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: RidgeConfig,
) -> tuple[float, float, int]:
    splits = generate_time_cv_splits(len(x_train), cfg)
    if len(splits) == 0:
        fallback_alpha = float(cfg.alpha_grid[0])
        return fallback_alpha, np.nan, 0

    best_alpha = float(cfg.alpha_grid[0])
    best_score = np.nan
    best_used_folds = 0

    y_values = y_train.values.astype(float)

    for alpha in cfg.alpha_grid:
        fold_scores: list[float] = []
        for tr_idx, va_idx in splits:
            x_tr_raw = x_train.iloc[tr_idx]
            x_va_raw = x_train.iloc[va_idx]
            y_tr = y_values[tr_idx]
            y_va = y_values[va_idx]

            x_tr_np, x_va_np = prepare_train_val(x_tr_raw, x_va_raw, cfg)
            model = Ridge(
                alpha=float(alpha),
                fit_intercept=cfg.fit_intercept,
                max_iter=cfg.max_iter,
                random_state=None,
            )
            model.fit(x_tr_np, y_tr)
            pred_va = model.predict(x_va_np)
            fold_metric = metric_score(y_va, pred_va, cfg.cv_metric)
            if np.isfinite(fold_metric):
                fold_scores.append(float(fold_metric))

        alpha_score = float(np.mean(fold_scores)) if len(fold_scores) > 0 else np.nan
        if metric_better(alpha_score, best_score, cfg.cv_metric):
            best_alpha = float(alpha)
            best_score = alpha_score
            best_used_folds = len(fold_scores)

    return best_alpha, best_score, best_used_folds


# =========================
# Ridge composite score
# =========================
def compute_target_y(index_open: pd.Series, mode: str) -> pd.Series:
    idx_open = index_open.astype(float).copy()
    if mode == "open_t2_over_t1_minus1":
        y = idx_open.shift(-2) / idx_open.shift(-1) - 1.0
        return y.rename("y_open_t2_over_t1_minus1")
    raise ValueError(f"Unsupported y_mode: {mode}")


def transform_score(raw_score: pd.Series, cfg: RidgeConfig) -> pd.Series:
    s = raw_score.astype(float).copy()
    if cfg.score_transform == "none":
        out = s
    elif cfg.score_transform == "zscore_rolling":
        out = rolling_zscore(
            s,
            window=cfg.score_z_window,
            min_periods=cfg.score_z_min_periods,
        )
    else:  # rank_pct
        rank = s.rolling(cfg.score_z_window, min_periods=cfg.score_z_min_periods).rank(pct=True)
        out = (rank - 0.5) * 2.0
    return out.replace([np.inf, -np.inf], np.nan)


def build_ridge_composite_score(
    x_all: pd.DataFrame,
    y_target: pd.Series,
    cfg: RidgeConfig,
    logger: logging.Logger | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    _validate_ridge_config(cfg)

    x_all = x_all.copy().sort_index()
    y_target = y_target.copy().reindex(x_all.index).sort_index()

    score_raw = pd.Series(index=x_all.index, dtype=float, name="score_ridge_raw")
    rows = []
    coef_rows = []

    start_pred_i = cfg.outer_train_window - 1 + cfg.label_availability_lag
    n = len(x_all)

    total_outer = 0
    skipped_outer = 0
    for pred_i in range(start_pred_i, n, cfg.outer_step):
        total_outer += 1
        train_end_i = pred_i - cfg.label_availability_lag
        if train_end_i < 0:
            skipped_outer += 1
            continue
        train_start_i = max(0, train_end_i - cfg.outer_train_window + 1)

        x_train_win = x_all.iloc[train_start_i : train_end_i + 1]
        y_train_win = y_target.iloc[train_start_i : train_end_i + 1]

        train_df = pd.concat([x_train_win, y_train_win], axis=1)
        train_df = train_df.dropna(subset=[y_target.name])

        if len(train_df) < cfg.min_train_size:
            skipped_outer += 1
            continue

        x_train = train_df[x_all.columns]
        y_train = train_df[y_target.name]

        best_alpha, cv_score, used_folds = select_best_alpha_with_time_cv(x_train, y_train, cfg)

        x_pred_raw = x_all.iloc[[pred_i]]
        x_train_np, x_pred_np = prepare_train_val(x_train, x_pred_raw, cfg)

        model = Ridge(
            alpha=float(best_alpha),
            fit_intercept=cfg.fit_intercept,
            max_iter=cfg.max_iter,
            random_state=None,
        )
        model.fit(x_train_np, y_train.values.astype(float))
        pred_val = float(model.predict(x_pred_np)[0])
        score_raw.iloc[pred_i] = pred_val

        rows.append(
            {
                "pred_date": x_all.index[pred_i],
                "train_start": x_all.index[train_start_i],
                "train_end": x_all.index[train_end_i],
                "n_train": int(len(x_train)),
                "best_alpha": float(best_alpha),
                "cv_score": float(cv_score) if np.isfinite(cv_score) else np.nan,
                "used_folds": int(used_folds),
            }
        )
        coef_row = {"pred_date": x_all.index[pred_i]}
        coef_row.update({col: float(val) for col, val in zip(x_all.columns, model.coef_)})
        coef_rows.append(coef_row)

        if logger is not None and len(rows) % 50 == 0:
            logger.info("ridge progress: done=%d latest_pred_date=%s", len(rows), x_all.index[pred_i])

    if cfg.fill_between_steps:
        score_raw = score_raw.ffill()
    score_raw = score_raw.fillna(0.0)

    composite_score = transform_score(score_raw, cfg).fillna(0.0)
    composite_score.name = "composite_score"

    diag = pd.DataFrame(rows)
    if len(coef_rows) > 0:
        coef_df = pd.DataFrame(coef_rows).set_index("pred_date").sort_index()
    else:
        coef_df = pd.DataFrame(columns=x_all.columns)
        coef_df.index.name = "pred_date"
    if logger is not None:
        logger.info(
            "ridge complete: total_outer=%d trained=%d skipped=%d",
            total_outer,
            len(diag),
            skipped_outer,
        )
    return composite_score, diag, coef_df


# =========================
# Original strategy helpers
# =========================
def get_market_money_1d(
    client,
    start_dt: datetime,
    end_dt: datetime,
    agg: str = "sum",
    fill_method: str = "ffill",
) -> pd.Series:
    money_df = client.get("money_1d", start_dt, end_dt).sort_index()
    if agg == "sum":
        m = money_df.sum(axis=1, skipna=True)
    elif agg == "median":
        m = money_df.median(axis=1, skipna=True)
    elif agg == "mean":
        m = money_df.mean(axis=1, skipna=True)
    else:
        raise ValueError("agg must be one of ['sum','median','mean']")

    m = m.replace([np.inf, -np.inf], np.nan).astype(float)
    if fill_method == "ffill":
        m = m.ffill().bfill()
    elif fill_method == "bfill":
        m = m.bfill().ffill()
    elif fill_method == "none":
        pass
    else:
        raise ValueError("fill_method must be one of ['ffill','bfill','none']")
    m.name = f"market_money_1d_{agg}"
    return m


def as_series(x, name=None) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        s = x.iloc[:, 0].copy()
    else:
        raise TypeError(f"Expect Series/DataFrame, got {type(x)}")
    s = s.sort_index().astype(float)
    if name is not None:
        s.name = name
    return s


def _max_drawdown(nav: pd.Series) -> float:
    nav = as_series(nav)
    cum_max = nav.cummax()
    dd = nav / cum_max - 1.0
    return float(dd.min())


def perf_stats_fallback_logret(r: pd.Series) -> pd.Series:
    r = as_series(r, "logret").fillna(0.0)
    ann = 252
    n = len(r)
    nav = np.exp(r.cumsum())
    out = {}
    out["AnnRet"] = float(np.exp(r.sum() * (ann / max(n, 1))) - 1.0) if n > 1 else np.nan
    out["AnnVol"] = float(r.std() * np.sqrt(ann)) if n > 1 else np.nan
    out["Sharpe"] = float(out["AnnRet"] / (out["AnnVol"] + 1e-12)) if np.isfinite(out["AnnVol"]) else np.nan
    out["MaxDD"] = _max_drawdown(nav) if n > 1 else np.nan
    out["EndNAV"] = float(nav.iloc[-1]) if n > 0 else np.nan
    return pd.Series(out)


def _as_param_series(x, index, name: str) -> pd.Series:
    if np.isscalar(x):
        return pd.Series(float(x), index=index, name=name)
    s = as_series(x, name=name).reindex(index)
    return s.ffill().bfill()


def target_pos_from_score_attackable(
    score: pd.Series,
    score_clip_low: float = -2.0,
    score_clip_high: float = 2.0,
    pos_floor: float = 0.10,
    pos_ceil: float = 1.00,
    full_th: float = 0.70,
    high_th: float = 0.55,
    mid_th: float = 0.40,
    low_th: float = 0.25,
) -> pd.Series:
    score = as_series(score, "score")
    if score_clip_high <= score_clip_low:
        raise ValueError("score_clip_high must be > score_clip_low")

    s = score.clip(score_clip_low, score_clip_high)
    strength = (s - score_clip_low) / (score_clip_high - score_clip_low)

    tgt = pd.Series(index=score.index, dtype=float)
    tgt[strength <= low_th] = pos_floor
    tgt[(strength > low_th) & (strength <= mid_th)] = pos_floor + 0.28 * (pos_ceil - pos_floor)
    tgt[(strength > mid_th) & (strength <= high_th)] = pos_floor + 0.60 * (pos_ceil - pos_floor)
    tgt[(strength > high_th) & (strength <= full_th)] = pos_floor + 0.85 * (pos_ceil - pos_floor)
    tgt[strength > full_th] = pos_ceil
    return tgt.rename("pos_target")


def follow_target_with_asym_steps_dynamic_up(
    pos_target: pd.Series,
    pos_init: float = 1.0,
    step_down_big: float = 0.50,
    step_down_small: float = 0.15,
    step_up_min: float = 0.03,
    step_up_max: float = 0.10,
    step_up_big: float = 0.22,
    up_power: float = 0.60,
    pos_floor: float = 0.10,
    pos_ceil: float = 1.00,
) -> pd.Series:
    pos_target = as_series(pos_target, "pos_target")
    idx = pos_target.index

    d_big_s = _as_param_series(step_down_big, idx, "step_down_big")
    d_small_s = _as_param_series(step_down_small, idx, "step_down_small")
    u_min_s = _as_param_series(step_up_min, idx, "step_up_min")
    u_max_s = _as_param_series(step_up_max, idx, "step_up_max")
    u_big_s = _as_param_series(step_up_big, idx, "step_up_big")

    pos_actual = pd.Series(index=idx, dtype=float)
    last = float(pos_init)

    for t in idx:
        tgt = pos_target.loc[t]
        if pd.isna(tgt):
            pos_actual.loc[t] = last
            continue

        d_big = float(d_big_s.loc[t])
        d_small = float(d_small_s.loc[t])
        u_min = float(u_min_s.loc[t])
        u_max = float(u_max_s.loc[t])
        u_big = float(u_big_s.loc[t])

        step_up_cap = float(max(u_big, u_max))

        if tgt < last:
            gap = last - tgt
            w = min(max(gap, 0.0), 1.0)
            step = d_small + w * (d_big - d_small)
            new_pos = max(last - step, tgt)
        elif tgt > last:
            gap_up = tgt - last
            w_up = min(max(gap_up, 0.0), 1.0)
            w_adj = float(w_up**up_power)
            step_up = u_min + w_adj * (step_up_cap - u_min)
            new_pos = min(last + step_up, tgt)
        else:
            new_pos = last

        new_pos = float(np.clip(new_pos, pos_floor, pos_ceil))
        pos_actual.loc[t] = new_pos
        last = new_pos

    return pos_actual.rename("pos_slow")


def follow_target_buy_attack_accelerator(
    pos_target: pd.Series,
    pos_init: float = 1.0,
    step_down_big: float = 0.50,
    step_down_small: float = 0.15,
    pos_attack: float = 0.85,
    pos_floor: float = 0.10,
    pos_ceil: float = 1.00,
    buy_mode: str = "FULL",
) -> pd.Series:
    pos_target = as_series(pos_target, "pos_target")
    idx = pos_target.index

    d_big_s = _as_param_series(step_down_big, idx, "step_down_big")
    d_small_s = _as_param_series(step_down_small, idx, "step_down_small")

    pos_actual = pd.Series(index=idx, dtype=float)
    last = float(pos_init)
    pos_attack = float(np.clip(pos_attack, pos_floor, pos_ceil))

    for t in idx:
        tgt = pos_target.loc[t]
        if pd.isna(tgt):
            pos_actual.loc[t] = last
            continue

        d_big = float(d_big_s.loc[t])
        d_small = float(d_small_s.loc[t])

        if tgt < last:
            gap = last - tgt
            w = min(max(gap, 0.0), 1.0)
            step = d_small + w * (d_big - d_small)
            new_pos = max(last - step, tgt)
        elif tgt > last:
            if buy_mode == "FULL":
                new_pos = pos_ceil
            else:
                new_pos = last
        else:
            new_pos = last

        new_pos = float(np.clip(new_pos, pos_floor, pos_ceil))
        pos_actual.loc[t] = new_pos
        last = new_pos

    return pos_actual.rename("pos_fast")


def build_attack_mask_B(
    composite_score: pd.Series,
    index_logret_open: pd.Series,
    q: float = 0.7,
    q_win: int = 252,
    risk_win: int = 20,
    risk_th: float = -0.05,
    min_periods: int = 60,
) -> pd.Series:
    s = as_series(composite_score, "score")
    r = as_series(index_logret_open, "idx_logret_open").reindex(s.index).fillna(0.0)

    thresh = s.rolling(q_win, min_periods=min_periods).quantile(q)
    cond1 = s > thresh

    roll = np.exp(r.rolling(risk_win, min_periods=max(5, risk_win // 2)).sum()) - 1.0
    cond2 = roll > risk_th

    out = (cond1 & cond2).fillna(False)
    return out.rename("attack_B")


def mix_positions_by_mask(pos_slow, pos_fast, fast_mask, pos_floor=0.10, pos_ceil=1.00):
    pos_slow = as_series(pos_slow, "pos_slow")
    pos_fast = as_series(pos_fast, "pos_fast").reindex(pos_slow.index)
    m = pd.Series(fast_mask, index=pos_slow.index).reindex(pos_slow.index).fillna(False).astype(bool)
    out = pos_slow.copy()
    out[m] = pos_fast[m]
    return out.clip(pos_floor, pos_ceil).rename("pos_mix")


def backtest_open_open(
    pos: pd.Series,
    index_logret_open: pd.Series,
    bt_start="2023-07-01",
    bt_end="2025-06-30",
    title="Strategy",
    plot=True,
    logger: logging.Logger | None = None,
):
    pos = as_series(pos, "pos")
    r = as_series(index_logret_open, "idx_logret_open")

    bt_start = pd.Timestamp(bt_start)
    bt_end = pd.Timestamp(bt_end)

    pos_bt = pos.loc[bt_start:bt_end].copy()
    r_bt = r.reindex(pos_bt.index).fillna(0.0)

    # open-open: use yesterday's pos
    pos_used = pos_bt.shift(1).ffill()
    if len(pos_used) > 0:
        pos_used.iloc[0] = pos_bt.iloc[0]
    pos_used = pos_used.fillna(pos_bt.iloc[0])

    ret_str = pos_used * r_bt

    nav_idx = np.exp(r_bt.cumsum())
    nav_str = np.exp(ret_str.cumsum())
    nav_ex = nav_str / nav_idx

    stats_idx = perf_stats_fallback_logret(r_bt)
    stats_str = perf_stats_fallback_logret(ret_str)

    comp = pd.DataFrame({"Index": stats_idx, "Strategy": stats_str})
    comp.loc["Hit", "Index"] = float((r_bt > 0).mean())
    comp.loc["Hit", "Strategy"] = float((ret_str > r_bt).mean())

    msg = f"\n=== Open-open Performance ({title}) ===\n{comp}"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(nav_idx.index, nav_idx, label="Index", linewidth=1.3)
        axes[0].plot(nav_str.index, nav_str, label=title, linewidth=1.6)
        axes[0].set_ylabel("NAV")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle="--", alpha=0.3)

        axes[1].plot(nav_ex.index, nav_ex, label="Excess NAV", linewidth=1.3)
        axes[1].axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.6)
        axes[1].set_ylabel("Excess NAV")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, linestyle="--", alpha=0.3)

        axes[2].step(pos_bt.index, pos_bt, where="post", label="Position", linewidth=1.2)
        axes[2].set_ylabel("Position")
        axes[2].set_ylim(0, 1.05)
        axes[2].legend(loc="upper left")
        axes[2].grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        "comp": comp,
        "pos_bt": pos_bt,
        "ret_idx": r_bt,
        "ret_str": ret_str,
        "nav_idx": nav_idx,
        "nav_str": nav_str,
        "nav_ex": nav_ex,
    }


def build_steps_from_money_continuous(
    money_1d: pd.Series,
    z_win: int = 120,
    clip_z: float = 1.0,
    base: dict | None = None,
    amp: dict | None = None,
):
    if base is None:
        base = {
            "step_up_min": 0.030,
            "step_up_max": 0.100,
            "step_up_big": 0.22,
            "step_down_big": 0.50,
            "step_down_small": 0.15,
        }
    if amp is None:
        amp = {
            "step_up_min": 0.020,
            "step_up_max": 0.050,
            "step_up_big": 0.06,
            "step_down_big": -0.10,
            "step_down_small": -0.05,
        }

    m = as_series(money_1d, "money_1d")
    mu = m.rolling(z_win, min_periods=max(30, z_win // 3)).mean()
    sd = m.rolling(z_win, min_periods=max(30, z_win // 3)).std().replace(0.0, np.nan)
    z = ((m - mu) / sd).clip(-clip_z, clip_z).fillna(0.0)

    cols = ["step_up_min", "step_up_max", "step_up_big", "step_down_big", "step_down_small"]
    out = pd.DataFrame(index=m.index, columns=cols, dtype=float)
    for c in cols:
        out[c] = base[c] + amp[c] * z
    out["liq_z"] = z
    return out


# =========================
# Pipeline
# =========================
def run_timing_ridge(cfg: AppConfig):
    from manage_db_read import ClickhouseReadOnly

    run_name, run_dir, logger = setup_run_artifacts(cfg)
    logger.info("run start: run_id=%s", run_name)
    logger.info("run dir: %s", run_dir)
    logger.info("config loaded")

    # ===== 1) Data source: ClickHouse (index + money) =====
    logger.info("step 1/5: loading index open/close and market money from ClickHouse")
    with open(cfg.data.db_settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)
    if cfg.data.db_settings_key not in settings:
        raise KeyError(f"db settings key not found: {cfg.data.db_settings_key}")
    db_config = settings[cfg.data.db_settings_key]
    client = ClickhouseReadOnly(
        database=db_config["database"],
        host=db_config["host"],
        port=db_config["port"],
        username=db_config["username"],
        password=db_config["password"],
    )

    close_df = client.get("index_close_1d", cfg.data.start_date, cfg.data.end_date)
    open_df = client.get("index_open_1d", cfg.data.start_date, cfg.data.end_date)

    if cfg.data.index_code not in close_df.columns or cfg.data.index_code not in open_df.columns:
        code_use = close_df.columns[0]
        logger.warning("%s 不在列中，自动改用第一列: %s", cfg.data.index_code, code_use)
    else:
        code_use = cfg.data.index_code

    index_close = close_df[code_use].astype(float).sort_index()
    index_open = open_df[code_use].astype(float).sort_index()

    index_logret = np.log(index_close / index_close.shift(1))
    index_logret_open = np.log(index_open / index_open.shift(1))
    index_logret.name = "index_logret_close"
    index_logret_open.name = "index_logret_open"

    money_1d = get_market_money_1d(client, cfg.data.start_date, cfg.data.end_date, agg="sum")

    idx = index_logret_open.index.union(index_logret.index).union(money_1d.index)
    idx = pd.DatetimeIndex(sorted(idx))
    index_logret = index_logret.reindex(idx)
    index_logret_open = index_logret_open.reindex(idx)
    money_1d = money_1d.reindex(idx).ffill().bfill()

    logger.info(
        "index series ready: %s -> %s len=%d",
        idx.min(),
        idx.max(),
        len(idx),
    )

    # ===== 2) Load features from parquet =====
    logger.info("step 2/5: loading 22/market/macro features from parquet")
    df_22 = ensure_datetime_index(pd.read_parquet(cfg.data.path_22), "22特征")
    feature_df_22 = df_22.select_dtypes(include=[np.number]).copy()

    df_35 = ensure_datetime_index(pd.read_parquet(cfg.data.path_35), "35特征")
    if "date" in df_35.columns and df_35["date"].isna().all():
        df_35 = df_35.drop(columns=["date"])
    feature_df_35_num = df_35.select_dtypes(include=[np.number]).copy()
    market_cols = [c for c in feature_df_35_num.columns if c.startswith("market_")]
    feature_df_35 = feature_df_35_num[market_cols].copy()

    if cfg.data.use_macro_features:
        df_macro = ensure_datetime_index(pd.read_parquet(cfg.data.path_macro), "macro特征")
        feature_df_macro = df_macro.select_dtypes(include=[np.number]).copy()
        x_all = (
            feature_df_22.join(feature_df_35, how="inner")
            .join(feature_df_macro, how="inner")
            .sort_index()
            .reindex(idx)
            .ffill()
        )
        n_macro_cols = int(feature_df_macro.shape[1])
    else:
        feature_df_macro = pd.DataFrame(index=feature_df_22.index)
        x_all = (
            feature_df_22.join(feature_df_35, how="inner")
            .sort_index()
            .reindex(idx)
            .ffill()
        )
        n_macro_cols = 0

    logger.info(
        "X_all ready shape=%s range=%s->%s cols_22=%d cols_market=%d cols_macro=%d",
        x_all.shape,
        x_all.index.min(),
        x_all.index.max(),
        feature_df_22.shape[1],
        feature_df_35.shape[1],
        n_macro_cols,
    )
    miss_col = x_all.isna().mean().sort_values(ascending=False).head(10)
    logger.info("X_all top missing ratios:\n%s", miss_col)

    # ===== 3) Build y and ridge composite score =====
    logger.info("step 3/5: training walk-forward ridge and building composite_score")
    y_target = compute_target_y(index_open.reindex(idx), mode=cfg.target.y_mode)
    composite_score, ridge_diag, ridge_coef = build_ridge_composite_score(
        x_all, y_target, cfg.ridge, logger=logger
    )

    logger.info("composite_score ready non_na=%d", int(composite_score.notna().sum()))
    if len(ridge_diag) > 0:
        logger.info("ridge windows=%d", len(ridge_diag))
        logger.info("alpha usage:\n%s", ridge_diag["best_alpha"].value_counts().sort_index())
        logger.info("latest windows:\n%s", ridge_diag.tail(5))
    if len(ridge_coef) > 0:
        top_mean_abs = ridge_coef.abs().mean().sort_values(ascending=False).head(20)
        logger.info("top mean-abs ridge coefficients:\n%s", top_mean_abs)

    # ===== 4) Original downstream logic unchanged =====
    logger.info("step 4/5: running downstream position pipeline")
    score_raw = as_series(composite_score, "score")
    r_open_raw = as_series(index_logret_open, "r_open")
    money_raw = as_series(money_1d, "money_1d")

    idx_master = score_raw.index.intersection(r_open_raw.index)
    score = score_raw.reindex(idx_master).ffill()
    r_open = r_open_raw.reindex(idx_master).fillna(0.0)
    money = money_raw.reindex(idx_master).ffill().bfill()

    logger.info("master idx: %s -> %s n=%d", idx_master.min(), idx_master.max(), len(idx_master))
    logger.info("money source: money_1d (ALL-market)")

    attack_B = build_attack_mask_B(
        score,
        r_open,
        q=cfg.strategy.attack_q,
        q_win=cfg.strategy.attack_q_win,
        risk_win=cfg.strategy.attack_risk_win,
        risk_th=cfg.strategy.attack_risk_th,
        min_periods=cfg.strategy.attack_min_periods,
    )

    pos_target = target_pos_from_score_attackable(
        score,
        score_clip_low=cfg.strategy.target_score_clip_low,
        score_clip_high=cfg.strategy.target_score_clip_high,
        pos_floor=cfg.strategy.target_pos_floor,
        pos_ceil=cfg.strategy.target_pos_ceil,
        full_th=cfg.strategy.target_full_th,
        high_th=cfg.strategy.target_high_th,
        mid_th=cfg.strategy.target_mid_th,
        low_th=cfg.strategy.target_low_th,
    ).clip(cfg.strategy.target_pos_floor, cfg.strategy.target_pos_ceil)

    steps_ct = build_steps_from_money_continuous(
        money,
        z_win=cfg.strategy.money_z_win,
        clip_z=cfg.strategy.money_clip_z,
        base=cfg.strategy.money_step_base,
        amp=cfg.strategy.money_step_amp,
    )

    pos_slow = follow_target_with_asym_steps_dynamic_up(
        pos_target,
        pos_init=1.0,
        step_down_big=steps_ct["step_down_big"],
        step_down_small=steps_ct["step_down_small"],
        step_up_min=steps_ct["step_up_min"],
        step_up_max=steps_ct["step_up_max"],
        step_up_big=steps_ct["step_up_big"],
        up_power=0.60,
        pos_floor=cfg.strategy.target_pos_floor,
        pos_ceil=cfg.strategy.target_pos_ceil,
    ).clip(cfg.strategy.target_pos_floor, cfg.strategy.target_pos_ceil)

    pos_fast = follow_target_buy_attack_accelerator(
        pos_target,
        pos_init=1.0,
        step_down_big=steps_ct["step_down_big"],
        step_down_small=steps_ct["step_down_small"],
        pos_attack=cfg.strategy.pos_attack,
        pos_floor=cfg.strategy.target_pos_floor,
        pos_ceil=cfg.strategy.target_pos_ceil,
        buy_mode="FULL",
    ).clip(cfg.strategy.target_pos_floor, cfg.strategy.target_pos_ceil)

    pos_mix = mix_positions_by_mask(
        pos_slow,
        pos_fast,
        attack_B,
        pos_floor=cfg.strategy.target_pos_floor,
        pos_ceil=cfg.strategy.target_pos_ceil,
    )

    bt_s = pd.Timestamp(cfg.strategy.bt_start)
    bt_e = pd.Timestamp(cfg.strategy.bt_end)
    pos_bt = pos_mix.loc[bt_s:bt_e]
    full_share_bt = float((pos_bt >= (cfg.strategy.target_pos_ceil - 1e-9)).mean())
    fast_share_bt = float(attack_B.reindex(pos_bt.index).mean())
    turnover_bt = float(pos_bt.diff().abs().sum())
    logger.info("FullShare(bt)=%.6f", full_share_bt)
    logger.info("FastShare(bt)=%.6f", fast_share_bt)
    logger.info("Turnover(bt)=%.6f", turnover_bt)

    res = backtest_open_open(
        pos_mix,
        r_open,
        bt_start=cfg.strategy.bt_start,
        bt_end=cfg.strategy.bt_end,
        title="RIDGE-INTERSECT (moneyALL, rolling z + FULL)",
        plot=False,
        logger=logger,
    )

    # ===== 5) Persist outputs =====
    logger.info("step 5/5: persisting outputs to disk")
    file_manifest: dict[str, str] = {}

    # config snapshot
    config_json = _to_jsonable(asdict(cfg))
    p = run_dir / "config_used.json"
    p.write_text(json.dumps(config_json, ensure_ascii=False, indent=2), encoding="utf-8")
    file_manifest["config_used_json"] = str(p)
    file_manifest["run_log"] = str(run_dir / "run.log")

    # key tables and series
    p = run_dir / "metrics_backtest.csv"
    res["comp"].to_csv(p, encoding="utf-8-sig")
    file_manifest["metrics_backtest_csv"] = str(p)

    p = run_dir / "ridge_diag.csv"
    ridge_diag.to_csv(p, index=False, encoding="utf-8-sig")
    file_manifest["ridge_diag_csv"] = str(p)

    p = run_dir / "ridge_coef_by_window.parquet"
    ridge_coef.to_parquet(p, compression=cfg.output.parquet_compression)
    file_manifest["ridge_coef_by_window_parquet"] = str(p)

    coef_summary = pd.DataFrame(
        {
            "coef_mean": ridge_coef.mean(axis=0),
            "coef_abs_mean": ridge_coef.abs().mean(axis=0),
            "coef_std": ridge_coef.std(axis=0),
            "coef_last": ridge_coef.iloc[-1] if len(ridge_coef) > 0 else np.nan,
        }
    ).sort_values("coef_abs_mean", ascending=False)
    p = run_dir / "ridge_coef_summary.csv"
    coef_summary.to_csv(p, encoding="utf-8-sig")
    file_manifest["ridge_coef_summary_csv"] = str(p)

    p = run_dir / "series_composite_score.parquet"
    save_series_to_parquet(composite_score.rename("composite_score"), p, compression=cfg.output.parquet_compression)
    file_manifest["series_composite_score_parquet"] = str(p)

    p = run_dir / "series_positions.parquet"
    df_positions = pd.concat(
        [
            pos_target.rename("pos_target"),
            pos_slow.rename("pos_slow"),
            pos_fast.rename("pos_fast"),
            pos_mix.rename("pos_mix"),
            attack_B.rename("attack_B"),
        ],
        axis=1,
    )
    df_positions.to_parquet(p, compression=cfg.output.parquet_compression)
    file_manifest["series_positions_parquet"] = str(p)

    p = run_dir / "series_backtest.parquet"
    df_backtest = pd.concat(
        [
            res["ret_idx"].rename("ret_idx"),
            res["ret_str"].rename("ret_str"),
            res["nav_idx"].rename("nav_idx"),
            res["nav_str"].rename("nav_str"),
            res["nav_ex"].rename("nav_ex"),
            res["pos_bt"].rename("pos_bt"),
        ],
        axis=1,
    )
    df_backtest.to_parquet(p, compression=cfg.output.parquet_compression)
    file_manifest["series_backtest_parquet"] = str(p)

    p = run_dir / "series_liquidity_steps.parquet"
    steps_ct.to_parquet(p, compression=cfg.output.parquet_compression)
    file_manifest["series_liquidity_steps_parquet"] = str(p)

    # optional big files (enabled by default)
    if cfg.output.save_big_files:
        p = run_dir / "x_all.parquet"
        x_all.to_parquet(p, compression=cfg.output.parquet_compression)
        file_manifest["x_all_parquet"] = str(p)

        p = run_dir / "y_target.parquet"
        save_series_to_parquet(y_target.rename("y_target"), p, compression=cfg.output.parquet_compression)
        file_manifest["y_target_parquet"] = str(p)

        p = run_dir / "index_inputs.parquet"
        index_inputs = pd.concat(
            [
                index_open.rename("index_open"),
                index_close.rename("index_close"),
                index_logret.rename("index_logret_close"),
                index_logret_open.rename("index_logret_open"),
                money_1d.rename("money_1d"),
            ],
            axis=1,
        )
        index_inputs.to_parquet(p, compression=cfg.output.parquet_compression)
        file_manifest["index_inputs_parquet"] = str(p)

    if cfg.output.save_plots:
        saved_plots = save_plots(res, steps_ct, run_dir, show_plots=cfg.output.show_plots)
        for k, v in saved_plots.items():
            file_manifest[k] = v

    # summary
    summary = {
        "run_id": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "index_code": cfg.data.index_code,
        "date_range_data": [str(idx.min()), str(idx.max())],
        "date_range_bt": [cfg.strategy.bt_start, cfg.strategy.bt_end],
        "x_all_shape": [int(x_all.shape[0]), int(x_all.shape[1])],
        "use_macro_features": bool(cfg.data.use_macro_features),
        "n_macro_cols_used": int(n_macro_cols),
        "ridge_windows": int(len(ridge_diag)),
        "full_share_bt": full_share_bt,
        "fast_share_bt": fast_share_bt,
        "turnover_bt": turnover_bt,
        "bt_metrics": {
            "AnnRet_Index": float(res["comp"].loc["AnnRet", "Index"]),
            "AnnVol_Index": float(res["comp"].loc["AnnVol", "Index"]),
            "Sharpe_Index": float(res["comp"].loc["Sharpe", "Index"]),
            "MaxDD_Index": float(res["comp"].loc["MaxDD", "Index"]),
            "Hit_Index": float(res["comp"].loc["Hit", "Index"]),
            "AnnRet_Strategy": float(res["comp"].loc["AnnRet", "Strategy"]),
            "AnnVol_Strategy": float(res["comp"].loc["AnnVol", "Strategy"]),
            "Sharpe_Strategy": float(res["comp"].loc["Sharpe", "Strategy"]),
            "MaxDD_Strategy": float(res["comp"].loc["MaxDD", "Strategy"]),
            "Hit_Strategy": float(res["comp"].loc["Hit", "Strategy"]),
        },
        "files": file_manifest,
    }
    p = run_dir / "summary.json"
    p.write_text(json.dumps(_to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    file_manifest["summary_json"] = str(p)

    if cfg.output.save_results_md:
        write_results_md(run_dir, run_name, cfg, summary, file_manifest)
        file_manifest["results_md"] = str(run_dir / "results.md")

    logger.info("run complete")
    logger.info("files saved:\n%s", "\n".join([f"- {k}: {v}" for k, v in sorted(file_manifest.items())]))

    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "files": file_manifest,
        "config": cfg,
        "index_open": index_open,
        "index_close": index_close,
        "index_logret": index_logret,
        "index_logret_open": index_logret_open,
        "money_1d": money_1d,
        "x_all": x_all,
        "y_target": y_target,
        "ridge_diag": ridge_diag,
        "ridge_coef": ridge_coef,
        "composite_score": composite_score,
        "attack_B": attack_B,
        "pos_target": pos_target,
        "pos_slow": pos_slow,
        "pos_fast": pos_fast,
        "pos_mix": pos_mix,
        "steps_ct": steps_ct,
        "summary": summary,
        "backtest": res,
    }


if __name__ == "__main__":
    app_cfg = AppConfig()
    _ = run_timing_ridge(app_cfg)
