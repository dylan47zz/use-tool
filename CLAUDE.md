# CLAUDE.md - 个人工具集合

## 项目概述

这是我的个人工具集合库，包含：

- **投资理财** (`trade_quant/`) - ETF 量化投资策略研究与回测
- **效率工具配置** - 各类效率工具的配置和脚本

## 项目结构

```
use-tool/
└── trade_quant/
    ├── etf_epo_research.py    # ETF-EPO 策略研究版本（聚宽环境）
    ├── etf_epo_backtest.py    # ETF-EPO 策略回测版本
    └── etf_epo_backtest_tmp.py # 临时回测脚本
```

## trade_quant 模块说明

### ETF-EPO 策略
基于 ETF 的轮动策略，使用多因子模型进行每周调仓。

**核心参数：**
- `ETF_POOL` - 目标 ETF 池（黄金、创业板、纳指、恒指、有色、石油、国债等）
- `ETF_NAME_MAP` - ETF 名称映射
- 策略逻辑：动量因子 + 波动率因子 + 相关性调整

### 使用环境
- 聚宽 (JoinQuant) 研究环境
- Python 3.x + pandas + numpy

## 开发约定

### Python 代码风格
- 使用 **Black** 格式化：`black *.py`
- 使用 **Ruff** 检查：`ruff check *.py`
- 类型注解：优先使用

### Git 提交规范
- 使用 GSD 工作流：`/gsd:new-project` → `/gsd:execute-phase` → `/gsd:verify-work`
- 提交信息格式：`type(scope): description`
  - `feat`: 新功能
  - `fix`: 修复
  - `refactor`: 重构
  - `chore`: 维护

### 测试
- 回测验证：确保新策略逻辑在历史数据上通过
- 单元测试：核心计算函数需要测试覆盖

## 常用命令

```bash
# 运行回测
python trade_quant/etf_epo_backtest.py

# 格式化代码
black trade_quant/

# 代码检查
ruff check trade_quant/
```

## 注意事项

- 回测代码仅供研究参考，不构成投资建议
- 实盘前需充分验证策略有效性
- 敏感信息（如 API Key）不要提交到仓库
