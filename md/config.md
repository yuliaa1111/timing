# timing_ridge.py 配置说明

本文档对应 [`timing_ridge.py`](/Users/yulia/Desktop/projects/timing/timing_ridge.py) 中的配置类：
- `DataConfig`
- `RidgeConfig`
- `StrategyConfig`
- `TargetConfig`
- `OutputConfig`
- `AppConfig`

目标：说明每个字段的含义、如何生效、可选项。

---

## 1. DataConfig

### `index_code`
- 含义：回测和目标收益使用的指数代码。
- 如何生效：用于从 ClickHouse 读取 `index_open_1d` / `index_close_1d` 的列。
- 示例：`"000852.XSHG"`。

### `start_date` / `end_date`
- 含义：主流程取数时间范围。
- 如何生效：限制指数与全市场成交额读取区间。

### `db_settings_path`
- 含义：ClickHouse 读库配置文件路径（json）。
- 如何生效：运行时从该文件读取连接参数。

### `db_settings_key`
- 含义：配置文件中使用的连接配置键名。
- 如何生效：默认读取 `settings[db_settings_key]` 作为连接参数字典。

### `use_macro_features`
- 含义：是否使用宏观因子参与 Ridge 训练。
- 如何生效：`True` 时用 `22 + market + macro`，`False` 时仅用 `22 + market`。

### `path_22`
- 含义：22个 `vp_*` 因子 parquet 路径。
- 如何生效：作为 Ridge 特征的一部分。

### `path_35`
- 含义：市场特征 parquet 路径。
- 如何生效：只读取其中 `market_*` 数值列作为特征。

### `path_macro`
- 含义：宏观因子 parquet 路径。
- 如何生效：读取全部数值列（中文列名可直接使用）作为特征。

---

## 2. RidgeConfig

### 外层滚动训练（walk-forward）

### `outer_train_window`
- 含义：每次训练使用的历史窗口长度（天）。
- 如何生效：第 `t` 次预测只用窗口内历史样本拟合。

### `outer_step`
- 含义：外层每次向前滚动的步长（天）。
- 如何生效：`1` 表示每日重训；`N>1` 表示每 N 天重训一次。

### `min_train_size`
- 含义：允许训练的最小样本数。
- 如何生效：不足则该预测点跳过。

### `label_availability_lag`
- 含义：标签可用滞后。
- 如何生效：防未来信息泄漏。当前目标是 `y_t=(open_{t+2}/open_{t+1})-1`，默认 `2`。

### `fill_between_steps`
- 含义：当 `outer_step>1` 时，非重训日是否前向填充分数。
- 如何生效：`True` 时 `composite_score` 每天都有值；`False` 仅重训日有值。

---

### 内层时间序列 CV（选 alpha）

### `cv_splitter`
- 含义：CV切分方式。
- 可选：`"expanding"` / `"rolling"`。
- 如何生效：
- `expanding`：训练集不断扩张。
- `rolling`：训练集按固定长度滚动。

### `cv_n_splits`
- 含义：fold 数量上限。
- 如何生效：每个 alpha 会在多个时间连续 fold 上评估并取均值。

### `cv_train_ratio`
- 含义：训练窗内，CV 初始训练段占比。
- 如何生效：影响每 fold 的 train/val 切分长度。

### `cv_min_train_size`
- 含义：每个 fold 的最小训练样本数。
- 如何生效：不满足则该 fold 不参与。

### `cv_min_val_size`
- 含义：每个 fold 的最小验证样本数。
- 如何生效：验证段太短的 fold 会被跳过。

### `cv_gap`
- 含义：train 与 val 之间的间隔天数（purge gap）。
- 如何生效：降低信息穿透风险，代价是可用样本减少。

### `cv_metric`
- 含义：内层选 alpha 的评估指标。
- 可选：`"rankic"` / `"mse"`。
- 如何生效：
- `rankic`：每 fold 计算 `Spearman(y_val, y_pred)`，alpha 取平均值最大者。
- `mse`：每 fold 计算均方误差，alpha 取平均值最小者。

---

### Ridge 模型参数

### `alpha_grid`
- 含义：Ridge 正则强度候选集合。
- 如何生效：内层 CV 在该网格中选最佳 alpha。

### `fit_intercept`
- 含义：是否拟合截距。
- 如何生效：传给 `sklearn.linear_model.Ridge`。

### `max_iter`
- 含义：最大迭代次数。
- 如何生效：传给 Ridge 求解器。

---

### 特征预处理

### `fill_method`
- 含义：缺失值处理策略。
- 可选：
- `ffill_then_train_mean`
- `train_mean_only`
- 如何生效：
- `ffill_then_train_mean`：先按时间前向填充，再用训练集列均值补剩余空值。
- `train_mean_only`：直接用训练集列均值填充。

### `standardize`
- 含义：是否标准化特征。
- 如何生效：`True` 时在训练子集拟合 `StandardScaler`，并用于验证/预测。

---

### Ridge 分数后处理（生成 `composite_score`）

### `score_transform`
- 含义：将 Ridge 原始预测值转为最终 `composite_score` 的方式。
- 可选：
- `none`
- `zscore_rolling`
- `rank_pct`
- 如何生效：
- `none`：直接使用原始预测值。
- `zscore_rolling`：`(score - rolling_mean) / rolling_std`。
- `rank_pct`：滚动窗口内百分位映射到 `[-1, 1]`。

### `score_z_window`
- 含义：`zscore_rolling` 或 `rank_pct` 的滚动窗口长度。
- 如何生效：控制尺度稳定性与响应速度。

### `score_z_min_periods`
- 含义：滚动变换最小样本要求。
- 如何生效：样本不足时该变换输出 NaN（后续流程会再做填充处理）。

---

## 3. StrategyConfig

这些参数对应原 `final_timing` 下游交易链路，保持方法不变。

### `bt_start` / `bt_end`
- 含义：回测区间。
- 如何生效：`backtest_open_open` 只评估该区间。

### `pos_attack`
- 含义：快速进攻仓位参数。
- 如何生效：传入 `follow_target_buy_attack_accelerator`。
- 备注：当前 `buy_mode="FULL"` 下，触发时直接到满仓，`pos_attack` 实际不主导结果（保留兼容）。

---

### target_pos 映射参数

### `target_score_clip_low` / `target_score_clip_high`
- 含义：`composite_score` 映射前的截断上下界。
- 如何生效：先执行 `score.clip(low, high)`，再线性归一到 `[0,1]`。
- 说明：若想放宽有效区间，例如 `[-3,3]`，同时改这两个值即可。

### `target_pos_floor` / `target_pos_ceil`
- 含义：目标仓位下限和上限。
- 如何生效：分段映射输出与后续仓位路径都会受该上下限约束。

### `target_low_th` / `target_mid_th` / `target_high_th` / `target_full_th`
- 含义：归一化强度 `strength` 的四个分段阈值。
- 如何生效：决定 `target_pos` 落在哪个档位。
- 建议：保持单调 `low < mid < high < full`。

---

### attack_B 触发参数

### `attack_q`
- 含义：分位阈值 `q`。
- 如何生效：条件1 `score > rolling_quantile(score, q)`。

### `attack_q_win`
- 含义：分位计算窗口长度。
- 如何生效：窗口越短越敏感，越长越平滑。

### `attack_risk_win`
- 含义：风险过滤累计收益窗口长度。
- 如何生效：条件2中计算 `open-open` 滚动累计收益。

### `attack_risk_th`
- 含义：风险过滤阈值。
- 如何生效：条件2 `roll_return > attack_risk_th`。

### `attack_min_periods`
- 含义：attack 分位计算最小样本数。
- 如何生效：样本不足时不易触发 attack。

---

### 资金驱动动态步长参数

### `money_z_win`
- 含义：把 `money_1d` 转换为 `liq_z` 的窗口长度。
- 如何生效：`liq_z = (money - rolling_mean) / rolling_std`。

### `money_clip_z`
- 含义：`liq_z` 截断上限。
- 如何生效：`liq_z` 被限制到 `[-clip, clip]`，避免极值导致步长异常。

### `money_step_base`
- 含义：各步长参数基线值。
- 如何生效：在中性流动性下作为默认调仓速度。

### `money_step_amp`
- 含义：流动性对步长的放大系数。
- 如何生效：`step_param_t = base[param] + amp[param] * liq_z_t`。

---

## 4. TargetConfig

### `y_mode`
- 含义：训练目标定义。
- 当前支持：`"open_t2_over_t1_minus1"`。
- 如何生效：`y_t = (open_{t+2} / open_{t+1}) - 1`。

---

## 5. OutputConfig

### `run_root`
- 含义：结果根目录。
- 如何生效：每次运行会在该目录下创建独立 `run_id` 子目录。

### `run_name`
- 含义：手动指定本次运行名称。
- 如何生效：若为 `None`，自动生成时间戳+指数代码。

### `save_big_files`
- 含义：是否保存大体积中间数据。
- 如何生效：`True` 时落盘 `x_all.parquet`、`y_target.parquet`、`index_inputs.parquet`。

### `save_plots`
- 含义：是否保存图片文件。
- 如何生效：`True` 时保存 NAV/超额NAV/仓位/尾部仓位/liq_z 图。

### `show_plots`
- 含义：保存时是否同时弹窗显示图片。
- 如何生效：`False` 只落盘不显示，`True` 则额外 `plt.show()`。

### `save_results_md`
- 含义：是否生成运行结果摘要文档。
- 如何生效：`True` 时在 run 目录写入 `results.md`。

### `parquet_compression`
- 含义：Parquet 压缩方式。
- 如何生效：用于所有 parquet 输出文件的压缩参数。

---

## 6. AppConfig

`AppConfig` 是总配置容器，包含：
- `data`
- `ridge`
- `strategy`
- `target`
- `output`

主流程通过：
- `app_cfg = AppConfig()`
- `run_timing_ridge(app_cfg)`

---

## 7. 修改示例

```python
from timing.timing_ridge import AppConfig, run_timing_ridge

cfg = AppConfig()

# 数据
cfg.data.start_date = datetime(2022, 1, 1)
cfg.data.end_date = datetime(2025, 6, 30)
cfg.data.db_settings_path = "/home/quant/projects/timing/data/db_settings.json"
cfg.data.db_settings_key = "clickhouse_read"
cfg.data.use_macro_features = False  # 不用宏观时改为 False

# Ridge
cfg.ridge.cv_metric = "rankic"        # or "mse"
cfg.ridge.cv_splitter = "rolling"     # or "expanding"
cfg.ridge.outer_train_window = 252
cfg.ridge.outer_step = 1
cfg.ridge.alpha_grid = [0.1, 1, 5, 10, 20, 50]
cfg.ridge.score_transform = "zscore_rolling"  # or "none" / "rank_pct"

# strategy
cfg.strategy.target_score_clip_low = -3.0
cfg.strategy.target_score_clip_high = 3.0
cfg.strategy.attack_q = 0.7
cfg.strategy.attack_risk_th = -0.05
cfg.strategy.money_z_win = 120

# output
cfg.output.run_root = "/home/quant/projects/timing/runs/timing_ridge"
cfg.output.run_name = None
cfg.output.save_big_files = True
cfg.output.save_plots = True
cfg.output.show_plots = False
cfg.output.save_results_md = True

out = run_timing_ridge(cfg)
```
