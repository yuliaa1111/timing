# 数据说明文档（data.md）

## 1. 文档目的

本文档用于明确本次 Ridge 化择时改造中涉及的所有数据来源、字段角色、频率、对齐方式、标签构造口径和最终训练样本结构，供 AI Agent 在编码时严格遵守。

---

## 2. 数据总览

本项目涉及四大类输入数据：

1. 指数行情数据
2. 市场 / 情绪 / 风险因子数据（`vp_*`、`market_*`）
3. 宏观数据（本地 parquet）
4. 回测辅助数据（如成交额、仓位限制相关序列）

最终目标是构造一个以日频交易日为索引的监督学习样本：

- `X_t`：t 日可观测的全部特征
- `y_t`：t+1 日指数收益

---

## 3. 时间索引与主键

## 3.1 主时间索引

所有训练样本最终都必须对齐到统一的日频交易日索引 `idx`。

推荐定义：

- 以目标指数（当前 notebook 中使用的指数）交易日为基准
- 索引类型：`DatetimeIndex`
- 排序要求：升序
- 去重要求：必须唯一

## 3.2 时间对齐原则

所有输入特征在进入 `X` 前都要对齐到 `idx`：

- 高频日度数据：直接 `reindex(idx)`
- 低频宏观数据：`reindex(idx).ffill()`
- 对最前端少量空值，可酌情 `bfill()`

严格禁止：

- 使用未来日期信息填充过去日期
- 使用 t+1 的指标值去预测 t+1 的收益

---

## 4. 指数数据

## 4.1 用途

指数数据承担三项作用：

1. 构造监督学习标签 `y`
2. 提供回测基准收益
3. 提供部分 attack / 风险过滤辅助变量

## 4.2 必需字段

至少应包含：

- 指数开盘价 `index_open`
- 指数收盘价 `index_close`
- 指数成交额 `money_1d`（如当前 notebook 已使用）

若 notebook 现有命名不同，Agent 可沿用原命名，但需在内部统一引用。

## 4.3 标签收益口径

本次 Ridge 训练标签优先使用：

- 下一交易日 open-open 收益
- 或 notebook 当前已有的 `index_logret_open.shift(-1)`

若当前 notebook 已稳定使用某一收益口径，保持与原回测一致优先。

---

## 5. vp_* 因子数据

## 5.1 用途

`vp_*` 特征代表原框架中的微观择时信号，通常包括：

- 趋势类
- 成交活跃类
- 资金行为类
- 市场情绪类

它们是 Ridge 的核心输入之一。

## 5.2 数据结构要求

通常为：

- index：日期
- columns：单个特征或多个特征列
- 频率：日频

若来自多个 parquet 或 notebook 内部构造，最终应合并到统一特征表中。

## 5.3 特征分组建议

### trend 组候选

示例（以当前 notebook 讨论内容为准）：

- `vp_turnover_weighted`
- `vp_money_ma10_over_ma60`
- `vp_price_ma10_over_ma60`
- `vp_momentum_20d`

### sentiment 组候选

示例：

- `vp_market_free_turnover_5d`
- `vp_etf_activity_share_5d`
- `vp_big_inflow_share_5d`
- `vp_gem_active_share_5d`

注：上述分组仅用于组织和诊断，不再用于手工组间加权。

---

## 6. market_* 因子数据

## 6.1 用途

`market_*` 特征代表市场整体状态和风险环境，通常包括：

- 成交量位置
- 成交额均线关系
- 资金分位
- 参与度
- 波动率
- 冲击成本
- 布林带宽度
- ATR

它们一部分可归入 sentiment，另一部分可归入 veto / risk。

## 6.2 数据结构要求

- index：日期
- columns：不同 `market_*` 字段
- 频率：日频
- 必须对齐到 `idx`

## 6.3 分组建议

### sentiment 候选

- 市场成交量 zscore
- 成交额均线比
- 市场参与度指标
- 资金活跃度分位

### veto / risk 候选

- `market_vol_percentile`
- `bb_width`
- `atr`
- `impact_cost`
- 行业集中度 / 一致性 / 极端占比类指标

---

## 7. 宏观数据

## 7.1 数据路径

宏观数据固定来源于本地 parquet：

```text
/Users/yulia/Desktop/clickhouse/macro_2014_2026.parquet
```

Agent 编码时应将路径写成变量，便于后续修改，例如：

```python
PATH_MACRO = "/Users/yulia/Desktop/clickhouse/macro_2014_2026.parquet"
```

## 7.2 数据结构预期

根据此前讨论，该文件应大致满足：

- index：日期，范围约 2014-01-02 到 2026-02-27
- columns：若干宏观变量名称
- 频率：混合低频（日度承载的月度或周度指标也可能存在）

## 7.3 已知宏观字段示例

可能包含以下字段：

- `DR007`
- `M1(货币):同比`
- `M2(货币和准货币):同比`
- `PPI:当月同比`
- `人民币离岸价(USDCNH):收盘价`
- `制造业PMI`
- `固定资产投资(不含农户)完成额:累计值`
- `社会消费品零售总额:当月同比`
- `社会融资规模存量:期末同比`
- `规模以上工业增加值:全部:当月同比`
- `逆回购:7日:回购利率`
- `金融机构:新增人民币贷款:中长期贷款:累计`

## 7.4 宏观字段处理要求

### 7.4.1 字段筛选

- 默认读取所有数值列
- 非数值列直接剔除
- 若存在全空列、常数列，也应剔除

### 7.4.2 列名处理

建议统一加前缀：

```text
macro_原字段名
```

目的是：

- 避免与 `vp_*`、`market_*` 冲突
- 后续做分组归因更方便

### 7.4.3 对齐方法

宏观数据最终必须变成日频特征：

1. 读取原 parquet
2. 确保 index 为 `DatetimeIndex`
3. 按 `idx` 重采样 / 对齐
4. 执行 `ffill()`
5. 样本前端少量缺口可 `bfill()`

### 7.4.4 为什么允许前向填充

因为宏观变量本身是低频发布，实际交易中在下次更新前，市场只能使用最近一次已知值。因此前向填充符合真实信息可得性。

---

## 8. 特征矩阵 X 的最终结构

## 8.1 行维度

- 每一行对应一个交易日 t
- 索引为 `idx`

## 8.2 列维度

- 每一列对应一个特征
- 列集合由以下拼接而成：
  - trend 特征
  - sentiment 特征
  - veto / risk 特征
  - macro 特征

即：

```text
X_all.shape = [n_samples, n_factors]
```

## 8.3 进入 Ridge 前的数据要求

- 全部为数值型
- index 与 y 对齐
- 缺失值允许存在，但要在 sklearn pipeline 中补齐
- 不得含未来信息

---

## 9. 标签 y 的数据定义

## 9.1 目标

标签 `y_t` 表示：

> 在 t 日收盘前可得的特征条件下，对 t+1 日指数收益的监督目标。

## 9.2 推荐实现

若当前 notebook 中已有：

```python
index_logret_open
```

则可定义：

```python
y = index_logret_open.shift(-1)
```

即：

- 第 t 行特征，预测第 t+1 行收益

## 9.3 样本对齐

由于 `shift(-1)` 会导致尾部最后一天没有标签，因此：

- 训练时需要将 `X` 和 `y` 同步裁剪到标签非空区间

---

## 10. 特征预处理细则

## 10.1 预处理原则

预处理要兼顾两层：

1. notebook 侧的时间序列处理
2. Ridge pipeline 侧的截面数值处理

## 10.2 notebook 侧建议

### 可做：

- rolling zscore
- 分位标准化
- 轻度 clip / winsorize

### 不建议：

- 在 notebook 里做过强的手工组合
- 在 notebook 里把一组特征先压缩成一个单分数再入模

## 10.3 pipeline 侧必须做

- `SimpleImputer(strategy="median")`
- `StandardScaler()`

原因：

- 宏观和市场特征量纲差异大
- 某些列有缺失
- Ridge 对尺度敏感

---

## 11. 滚动训练样本切法

## 11.1 训练窗口

推荐初版：

- `train_window = 252`

即使用过去约一年交易日训练。

## 11.2 重训频率

推荐初版：

- `refit_every = 20`

即每 20 个交易日重训一次模型。

## 11.3 预测时点

在 t 日：

- 仅使用截至 t 日的特征与历史标签训练模型
- 输出对 t+1 收益的预测 `ridge_pred[t]`

---

## 12. 输出信号数据定义

## 12.1 ridge_pred

`ridge_pred` 是 Ridge 原始输出：

- index：交易日 t
- value：模型对 t+1 日收益的预测值

## 12.2 composite_score

新的 `composite_score` 定义为：

- 对 `ridge_pred` 做滚动标准化后的连续信号

例如：

```python
composite_score = zscore(ridge_pred, w=252, minp=80).clip(-2, 2)
```

这个变量只是为了兼容旧框架下游接口名。

## 12.3 position

`position` 是最终输出：

- 范围通常在 `[0, 1]` 或 notebook 当前设定区间
- 表示指数层面的目标仓位百分比
- 该值也可以后续作为全市场选股模型的总仓位控制器

---

## 13. 分组映射表要求

为了支持权重分析，Agent 应维护一个特征到分组的映射字典，例如：

```python
feature_group_map = {
    "vp_turnover_weighted": "trend",
    "vp_money_ma10_over_ma60": "trend",
    ...,
    "market_vol_percentile": "veto",
    "market_bb_width": "veto",
    ...,
    "macro_DR007": "macro",
    ...,
}
```

用途：

1. 汇总不同组的 Ridge 权重贡献
2. 检查宏观组是否有效
3. 对比趋势 / 情绪 / 风险 / 宏观的重要性变化

---

## 14. 诊断数据输出要求

## 14.1 系数表

应输出 `coef_df`：

- index：每次重训日期
- columns：特征名
- values：对应 Ridge 系数

## 14.2 分组权重表

应输出 `coef_group_abs`：

- index：重训日期
- columns：`trend`, `sentiment`, `veto`, `macro`
- values：组内系数绝对值总和或均值

## 14.3 信号表

应保留以下序列用于可视化和后续分析：

- `ridge_pred`
- `composite_score`
- `position`
- 指数收益序列
- 策略净值序列
- 超额收益序列

---

## 15. 数据质量检查清单

Agent 在正式训练前，必须至少检查以下内容：

1. `idx` 是否升序且唯一
2. `X_all` 是否全为数值列
3. 宏观列是否成功加前缀 `macro_`
4. 是否存在全空列
5. 是否存在常数列
6. `X` 与 `y` 是否严格按日期对齐
7. 是否仅使用 t 日特征预测 t+1 收益
8. `ridge_pred` 是否存在大面积 NaN
9. `composite_score` 是否存在异常极端值
10. `position` 是否仍落在策略允许范围内

---

## 16. 与原框架的兼容性说明

## 16.1 保留项

以下变量或逻辑在接口上尽量保持原样：

- `composite_score`
- `attack_B`
- `pos_target`
- `pos_slow`
- `pos_fast`
- `position`

## 16.2 变化项

原框架中的：

- `trend_score`
- `sent_total`
- `veto_total`
- `0.6 / 0.4 / veto` 的人工合成

不再作为最终信号构造主路径。

它们可以保留为：

- 特征组织参考
- 调试对照信号
- 旧版本基准对比

但不应继续作为主策略信号来源。

---

## 17. 一句话版数据定义

本项目最终要构造的是：

> 以指数交易日为主索引、由 `vp_* + market_* + macro_*` 组成的日频特征矩阵 `X_t`，配合下一日指数收益 `y_t` 进行滚动 Ridge 训练，输出标准化后的 `composite_score`，再接入原有的仓位管理模块得到最终 `position`。

