# ETF-EPO 策略详细说明

> 基于 ETF 的动量-质量轮动策略，使用 Ledoit-Wolf 最优收缩估计

## 策略概述

ETF-EPO（ETF Enhanced Portfolio Optimization）策略是一个**多因子轮动策略**，每周三调仓一次。策略核心思想是：

1. **动量因子**：选择近期表现强势的 ETF
2. **质量因子**：选择波动稳定、回撤小的 ETF
3. **EPO 权重优化**：基于 Ledoit-Wolf 协方差估计进行组合优化
4. **风控约束**：行业配置限制、个股权重上限、债券保底机制

## 策略绩效（历史回测）

| 指标 | 数值 |
|------|------|
| 年化收益 | 31%+ |
| 最大回撤 | 12.6% |
| 夏普比率 | ~1.5 |
| 调仓频率 | 每周三 |

> ⚠️ 回测结果仅供参考，过往业绩不代表未来表现

## ETF 池配置

```python
ETF_POOL = [
    "518880.XSHG",  # 黄金ETF（商品）
    "159915.XSHE",  # 创业板ETF（A股成长）
    "513100.XSHG",  # 纳指ETF（美股）
    "513980.XSHG",  # 恒指ETF（港股）
    "159980.XSHE",  # 有色ETF（周期）
    "561360.XSHG",  # 石油ETF（周期）
    "511260.XSHG",  # 十年国债ETF（债券）
    "561550.XSHG",  # 中证500ETF（A股成长）
    "159259.XSHE",  # 成长ETF（A股成长）
    "159263.XSHE",  # 价值ETF（A股价值）
]
```

### 分类配置

| 分类 | ETF | 说明 |
|------|-----|------|
| 商品 | 黄金ETF | 避险资产 |
| A股成长 | 创业板、中证500、成长、价值 | A股权益 |
| 美股 | 纳指ETF | 海外科技 |
| 港股 | 恒指ETF | 海外中国 |
| 周期 | 有色、石油 | 大宗商品 |
| 债券 | 十年国债 | 防御资产 |

## 因子模型

### 动量因子 (30%)

- **动量得分**：基于 25 日收益率的年化收益 × R²
- **放量比**：5 日均量 / 20 日均量
- **趋势判断**：价格站上 20 日均线 且 回撤 < 8%

### 质量因子 (70%)

- **Sharpe 得分**：风险调整收益
- **回撤得分**：最大回撤（越低越好）
- **波动率得分**：年化波动率（越低越好）
- **波动稳定性**：波动率的波动（越低越好）
- **成交量稳定性**：成交量的波动（越低越好）
- **对数收益**：对数收益率
- **R² 得分**：趋势拟合度

### 信号计算

```python
signal = 0.3 × momentum_score + 0.7 × quality_score
```

债券 ETF 特殊处理：仅使用动量得分

## EPO 权重优化

### Ledoit-Wolf 收缩估计

使用 Ledoit-Wolf 方法自动计算协方差矩阵的最优收缩强度：

```python
shrinkage = LedoitWolf().shrinkage_  # 自动计算
shrinkage = clip(shrinkage, 0.05, 0.3)  # 限制范围
```

### 权重计算

```python
# 1. 计算逆协方差矩阵
inv_cov = inverse(shrunk_covariance)

# 2. 基于信号计算权重
raw_weights = inv_cov × signal

# 3. 仅做多约束
raw_weights = max(0, raw_weights)

# 4. 归一化
weights = raw_weights / sum(raw_weights)
```

## 风控约束

### A 股权益限制

- 最大持仓数：1 只
- 总仓位上限：50%（动态调整 35%~50%）

### 行业限制

- 周期行业（有色/石油）：最多 1 只
- 行业总仓位：≤ 35%（动态调整 25%~35%）

### 个股权重

- 单个 ETF 权重上限：85%（动态调整 75%~85%）

### 债券保底（可选）

- 模式 1：`use_bond_floor = True`：债券最低 15% 仓位
- 模式 2：`use_bond_floor = False`：完全排除债券 ETF

### 拥挤度惩罚

- 放量比 > 1.6 时施加惩罚
- 相对拥挤度惩罚
- 回撤惩罚

## 参数配置

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `momentum_window` | 25 | 动量计算窗口 |
| `quality_window` | 25 | 质量计算窗口 |
| `cov_window` | 60 | 协方差计算窗口 |
| `score_weight_momentum` | 0.3 | 动量权重 |
| `score_weight_quality` | 0.7 | 质量权重 |
| `max_holdings` | 6 | 最大持仓数 |
| `signal_power` | 1.4 | 信号幂次 |

### EPO 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_ledoit_wolf` | True | 启用 Ledoit-Wolf 收缩 |
| `shrinkage_floor` | 0.05 | 最小收缩强度 |
| `shrinkage_cap` | 0.3 | 最大收缩强度 |
| `anchor_weight` | 0.1 | 锚定信号权重 |

### 风控参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_weight` | 0.85 | 个股权重上限 |
| `a_share_weight_cap` | 0.5 | A股权重上限 |
| `industry_weight_cap` | 0.35 | 行业权重上限 |
| `use_bond_floor` | False | 启用债券保底 |

## 使用指南

### 1. 聚宽研究环境

```python
from jqdata import *
exec(open('etf_epo_research.py').read())

# 计算指定日期
result = calculate_for_date('2025-01-15', verbose=True)
print(result['target_weights'])

# 批量计算
df = get_next_wednesday_trades('2024-01-01', '2025-01-15')
```

### 2. 聚宽回测环境

```python
# 在聚宽量化平台创建回测策略
# 将 etf_epo_backtest.py 的代码复制到策略中
```

### 3. 计算调仓方案

```python
# 输入当前持仓
current_positions = {
    '518880.XSHG': 500,   # 黄金ETF 500股
    '159915.XSHE': 300,   # 创业板ETF 300股
}
total_value = 100000       # 总资产10万元
current_prices = {
    '518880.XSHG': 4.5,
    '159915.XSHE': 2.1,
    # ... 其他价格
}

# 获取目标权重
result = calculate_for_date('2025-01-15', verbose=False)
target_weights = result['target_weights']

# 计算调仓方案
plan = calculate_rebalance_plan(
    current_positions=current_positions,
    total_value=total_value,
    target_weights=target_weights,
    current_prices=current_prices,
    min_lot=100,
    verbose=True
)

print(plan['orders'])    # 订单列表
print(plan['turnover'])  # 预估成交额
```

## 输出解读

### 因子信号排名

```
排名  ETF代码        综合信号    动量分   质量分   放量比  趋势
-----------------------------------------------------------------------
1     黄金ETF       0.7500     0.8250   0.7320   1.2    ✓
2     创业板ETF     0.6800     0.6120   0.7150   0.9    ✓
```

### 调仓方案

```
ETF代码          名称        目标权重    分类      信号
----------------------------------------------------------------------
518880.XSHG     黄金ETF     45.00%     商品      0.7500
159915.XSHE     创业板ETF   35.00%     A股       0.6800
513100.XSHG     纳指ETF     20.00%     境外      0.5500
```

## 策略优势

1. **低相关性**：多资产类别配置，分散风险
2. **自动优化**：Ledoit-Wolf 协方差估计，减少过拟合
3. **动态风控**：根据市场状态自动调整仓位
4. **简单透明**：逻辑清晰，易于理解和修改

## 风险提示

1. 回测结果不代表未来表现
2. 海外 ETF 存在汇率风险
3. 策略可能失效，需定期监控
4. 流动性风险：部分 ETF 成交不活跃

## 文件说明

| 文件 | 说明 |
|------|------|
| `etf_epo_research.py` | 研究环境版本，用于策略研究和信号计算 |
| `etf_epo_backtest.py` | 回测环境版本，用于历史回测 |
| `README.md` | 本文档 |

## 参考资料

- 聚宽文章：[年化31%+、最大回撤12.6%-复现一个ETF-EPO策略](https://www.joinquant.com/post/66279)
- Ledoit-Wolf 协方差估计：Ledoit & Wolf (2004)

---

> ⚠️ 免责声明：本策略仅供研究参考，不构成投资建议。投资有风险，入市需谨慎。
