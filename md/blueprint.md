# 择时 Ridge 化改造蓝图（blueprint.md）

## 1. 目标

将现有的“手工打分式指数择时框架”改造为“底层因子矩阵 + 宏观特征 + 滚动 Ridge 预测 + 原调仓执行层”的两层结构：

- 上游：使用底层特征直接训练 Ridge，预测下一交易日指数收益。
- 下游：保留现有的 `attack_B / slow-fast / position` 调仓与仓位映射逻辑。

本次改造**不做风格层**，不做大小盘轮动，不做额外资产配置，仅服务于以下目标：

1. 用 Ridge 替代原先的手工加权打分。
2. 把宏观数据作为慢变量特征接入模型。
3. 保留当前框架较强的下跌防守能力。
4. 在上涨阶段提升信号的进攻性和弹性。
5. 让最终输出仍然是可直接用于指数回测或股票组合总仓位控制的 `position`。

---

## 2. 当前框架与改造边界

### 2.1 当前框架（需保留的部分）

现有 notebook 的核心链路可以概括为：

1. 读取指数价格、成交额、`vp_*` 因子、`market_*` 因子。
2. 人工构造 `trend_score`、`sent_total`、`veto_total`。
3. 人工合成：
   - `composite_score = 0.6 * trend_score + 0.4 * sent_total + veto_total`
4. 将 `composite_score` 输入：
   - `build_attack_mask_B`
   - `target_pos_from_score_attackable`
   - `follow_target_with_asym_steps_dynamic_up`
   - `follow_target_buy_attack_accelerator`
   - `mix_positions_by_mask`
5. 得到最终 `position`，并进行指数层面的 open-open 回测。

### 2.2 本次改造范围

只替换**第 2、3 步**，即：

- 不再手工构造总分。
- 改为用底层因子直接进入 Ridge。
- 用 Ridge 输出接管原先 `composite_score` 的角色。

### 2.3 本次不改的部分

以下逻辑原则上保留，不重写：

- `attack_B` 的整体机制
- `slow/fast` 调仓机制
- 目标仓位映射逻辑
- 回测逻辑（仍以指数 open-open 为主）
- 原有画图与绩效评估逻辑

---

## 3. 新框架总流程

新框架应当实现如下链路：

```text
底层因子矩阵 X（trend + sentiment + veto + macro）
        ↓
滚动 Ridge 训练，预测下一日指数收益
        ↓
ridge_pred（原始预测值）
        ↓
对 ridge_pred 做滚动标准化 / 裁剪
        ↓
composite_score（新的总信号，仅作为下游接口名）
        ↓
attack_B / slow-fast / target_pos / position
        ↓
指数回测结果
```

注意：

这里的 `composite_score` 不再代表人工加权总分，只是为了兼容原有下游函数命名，实际含义是：

> “Ridge 对下一日收益的预测值，经标准化后形成的连续交易信号。”

---

## 4. 特征层设计

## 4.1 特征总原则

所有底层特征都应直接进入 Ridge，而不是先压缩成手工组内得分再组间加权。

保留“trend / sentiment / veto / macro”这四个组，仅用于：

1. 组织特征池
2. 调试与诊断
3. 分组分析 Ridge 权重

不再用于手工打分。

### 4.2 四类特征分组

#### A. trend 特征

代表趋势和价格成交方向相关特征，例如：

- `vp_turnover_weighted`
- `vp_money_ma10_over_ma60`
- `vp_price_ma10_over_ma60`
- `vp_momentum_20d`

若原 notebook 中某些因子在手工打分时使用了反号，则进入 Ridge 之前应延续相同方向约定，避免经济解释混乱。

#### B. sentiment 特征

代表市场情绪、交易活跃度、资金参与度的特征，例如：

- `vp_market_free_turnover_5d`
- `vp_etf_activity_share_5d`
- `vp_big_inflow_share_5d`
- `vp_gem_active_share_5d`
- 市场成交量 zscore
- 成交额均线比
- 市场参与度指标
- 资金分位类指标

#### C. veto / risk 特征

代表拥挤度、波动风险、执行风险等特征，例如：

- 行业集中度
- 换手一致性
- 极端换手占比
- `market_vol_percentile`
- `bb_width`
- `atr`
- `impact_cost`

注意：本组不再作为“单独扣分项”直接减去，而是作为 Ridge 的输入特征，让模型自行学习其对未来收益的负面约束作用。

#### D. macro 特征

来自本地宏观 parquet 文件：

- 路径：`/Users/yulia/Desktop/clickhouse/macro_2014_2026.parquet`

该文件中的所有数值列都可以作为候选宏观特征，统一加上 `macro_` 前缀后接入模型。

宏观特征是低频慢变量，需要先对齐到指数交易日索引，再做前向填充。

---

## 5. 特征预处理规范

## 5.1 时间索引对齐

所有特征最终必须对齐到统一的日频索引 `idx`，该索引以指数交易日为准。

### 规则：

- `vp_*` 与 `market_*` 如本身已是日频，则直接按 `idx` 对齐。
- 宏观数据若是月频或不规则频率，则先转 `DatetimeIndex`，再 `reindex(idx)`。
- 低频宏观特征对齐后做：
  - `ffill()`
  - 必要时 `bfill()` 仅用于样本最前端少量缺口

## 5.2 数值标准化

底层因子可在入模前做滚动标准化，以减轻量纲差异和 regime 切换带来的漂移。

可选方式：

1. 滚动 zscore
2. rolling percentile / rank 转换
3. 原值保留，由 sklearn `StandardScaler` 处理

推荐做法：

- 在 notebook 特征工程阶段先做轻量滚动处理，减少极端值影响。
- 在 Ridge pipeline 中再使用 `StandardScaler` 做截面统一标准化。

## 5.3 缺失值处理

Ridge 训练时不能直接接受 NaN。

推荐在 sklearn pipeline 中使用：

- `SimpleImputer(strategy="median")`

对于宏观数据和部分市场特征，允许保留前期空值，但进入模型前必须补齐。

## 5.4 极值处理

建议对最终进入 Ridge 的特征做轻度裁剪，例如：

- clip 到 rolling zscore 的 `[-5, 5]`
- 或按历史分位做 winsorize

避免极端值把 Ridge 权重拉偏。

---

## 6. 标签 y 的定义

本次 Ridge 的监督目标是：

> 使用 t 日特征，预测 t+1 日的指数收益。

推荐优先使用与当前回测一致的收益口径，例如：

- `open_t -> open_{t+1}`
- 或 notebook 当前已在使用的 `index_logret_open.shift(-1)`

注意：

- 特征必须使用 t 日收盘前可获得的信息。
- 标签是 t+1 的收益。
- 严禁未来函数。

---

## 7. Ridge 模型设计

## 7.1 模型形式

使用 sklearn 的 Ridge 回归。

推荐 pipeline：

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=8.0, fit_intercept=True)),
])
```

### 各步骤作用：

- `imputer`：补缺失值
- `scaler`：统一尺度
- `Ridge`：用 L2 正则稳定权重

## 7.2 为什么用 Ridge

与普通 OLS 相比，Ridge 更适合当前场景，因为：

1. 底层特征很多，且相关性强。
2. 宏观与市场类变量之间共线性较高。
3. 需要平滑、稳健的权重，而不是激进跳变。
4. 目标不是解释，而是构造稳定的连续交易信号。

---

## 8. 训练方式：滚动 / walk-forward

## 8.1 原则

采用滚动训练，避免全样本静态拟合导致的未来信息泄漏和参数滞后。

## 8.2 推荐方式

例如：

- 训练窗口：252 个交易日
- 每隔 20 个交易日重训一次
- 用最近一个可用模型预测当前日信号

伪代码：

```python
for t in range(train_window, len(idx)):
    if 到达重训点:
        用 [t-train_window, t) 训练 Ridge
    用当前模型预测 t 日的 ridge_pred
```

## 8.3 输出

模型输出 `ridge_pred[t]`，表示：

- 在 t 日，根据当日特征，对 t+1 收益的预测值

---

## 9. composite_score 的重新定义

## 9.1 新定义

新的 `composite_score` 不再是人工打分，而是：

```text
ridge_pred → 滚动标准化 → composite_score
```

例如：

```python
composite_score = zscore(ridge_pred, w=252, minp=80).clip(-2.0, 2.0)
```

## 9.2 为什么还保留这个变量名

因为下游大量函数已经依赖 `composite_score` 作为输入。

为了尽量少改后半段代码，直接让：

```python
composite_score = standardized_ridge_signal
```

即可。

这只是接口兼容，不代表仍然使用人工加权逻辑。

---

## 10. 下游执行层保留原则

以下函数原则上不改或仅做最小改动：

- `build_attack_mask_B`
- `target_pos_from_score_attackable`
- `follow_target_with_asym_steps_dynamic_up`
- `follow_target_buy_attack_accelerator`
- `mix_positions_by_mask`

### 新旧关系：

- 旧：`composite_score` 来自手工加权
- 新：`composite_score` 来自 Ridge 输出标准化

除信号来源变化外，下游执行层逻辑保持一致。

---

## 11. 攻击与防守逻辑

## 11.1 当前优势

现有框架的主要优势是：

- 下跌阶段防守较好
- 超额收益主要来自回撤控制

## 11.2 Ridge 化后的目标

希望在不破坏防守逻辑的前提下：

- 让上涨阶段信号更有弹性
- 让仓位在风险偏好上升时更敢于抬高

因此，本次改造中：

- 防守能力主要由 `attack_B`、`slow-fast`、目标仓位限制来保障。
- Ridge 主要负责替代手工打分，改善信号来源质量。

不要在本阶段同时大改 attack 或仓位上限规则，以免无法归因。

---

## 12. 诊断输出与解释性要求

Agent 编码时，必须额外输出以下诊断结果，方便后续分析：

### 12.1 因子权重诊断

每次重训后记录 Ridge 系数，形成：

- `coef_df`：行是重训日，列是特征名，值是系数

### 12.2 分组权重诊断

根据特征名前缀或分组映射，统计：

- trend 组绝对权重和
- sentiment 组绝对权重和
- veto 组绝对权重和
- macro 组绝对权重和

用于判断：

- 模型到底更依赖哪一组特征
- 宏观特征是否真的起作用
- 宏观特征是否挤占了情绪或趋势特征的权重

### 12.3 信号分布诊断

输出并检查：

- `ridge_pred` 的时间序列
- `composite_score` 的时间序列
- `composite_score` 的分位分布

用于判断标准化后信号是否过于扁平或过于极端。

---

## 13. 回测与验证要求

## 13.1 回测口径

保持与原 notebook 一致：

- 指数 open-open 回测
- 使用 `position.shift(1)` 或现有一致口径

## 13.2 至少对比以下版本

1. 原始手工打分版本
2. Ridge（仅 vp + market）版本
3. Ridge（vp + market + macro）版本

### 重点观察：

- 年化收益
- Sharpe
- 最大回撤
- 超额收益
- 上涨阶段跟涨能力
- 下跌阶段防守能力

## 13.3 归因建议

若宏观版在上涨阶段更好，应进一步检查：

- 是宏观特征真的提高了预测质量
- 还是宏观特征让 `composite_score` 分布更偏正，从而仓位更高

这两者都可以接受，但要分清。

---

## 14. 对 AI Agent 的具体编码要求

Agent 需要完成以下工作：

1. 读取现有 notebook 或等价 Python 代码结构。
2. 定位原手工构造 `trend_score / sent_total / veto_total / composite_score` 的代码块。
3. 删除或旁路旧的组间手工加权逻辑。
4. 保留底层因子分组与方向处理。
5. 新增宏观 parquet 读取与对齐代码。
6. 构造统一的特征矩阵 `X_all`。
7. 构造标签 `y_next_ret`。
8. 编写滚动 Ridge 训练与预测函数。
9. 将 Ridge 输出标准化后赋给新的 `composite_score`。
10. 保证后续 attack / slow-fast / position 代码仍可直接运行。
11. 增加权重、分组、信号分布等诊断输出。
12. 保证代码尽量模块化，便于后续改成 notebook cell 或函数。

---

## 15. 最终应交付的代码形态

建议交付以下模块或 notebook 代码块：

1. **数据读取块**
   - 指数
   - `vp_*`
   - `market_*`
   - `macro_*`

2. **特征工程块**
   - 对齐
   - 标准化
   - 极值处理
   - 缺失值准备

3. **标签构造块**
   - 下一日收益

4. **滚动 Ridge 训练块**
   - 输出 `ridge_pred`
   - 输出 `coef_df`

5. **信号构造块**
   - `composite_score`

6. **下游调仓块**
   - 复用原逻辑

7. **回测与诊断块**
   - 策略曲线
   - 超额曲线
   - 权重诊断
   - 分组权重诊断

---

## 16. 一句话版总结

本次改造的核心不是“把宏观手工加到总分里”，而是：

> 用底层市场因子 + 宏观慢变量共同组成特征矩阵，滚动训练 Ridge 来预测下一日指数收益，再把该预测值标准化后接入原有的 attack / slow-fast 仓位管理框架。

