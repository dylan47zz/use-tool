# ETF-EPO策略 - 研究环境版本
"""
使用方法：
    # 在聚宽研究环境
    from jqdata import *
    exec(open('etf_epo_v2_research.py').read())

    # 计算指定日期的交易标的
    result = calculate_for_date('2025-01-15')
    print("目标权重:", result['target_weights'])

    # 批量计算所有周三的标的
    dates = pd.date_range('2024-01-01', '2025-01-15', freq='W-WED')
    results = []
    for date in dates:
        r = calculate_for_date(date.strftime('%Y-%m-%d'))
        if r['success']:
            results.append({
                'date': r['date'],
                'etfs': list(r['target_weights'].keys()),
                'weights': r['target_weights']
            })
"""

import numpy as np
import pandas as pd
import math

# ============ 参数配置（与v2一致）============
ETF_POOL = [
    "518880.XSHG",
    "159915.XSHE",
    "513100.XSHG",
    "513980.XSHG",
    "159980.XSHE",
    "561360.XSHG",
    "511260.XSHG",
    # v2: 新增ETF
    "561550.XSHG",  # 中证500ETF
    "159259.XSHE",  # 成长ETF
    "159263.XSHE",  # 价值ETF
]
ETF_NAME_MAP = {
    "518880.XSHG": "黄金ETF",
    "159915.XSHE": "创业板ETF",
    "513100.XSHG": "纳指ETF",
    "513980.XSHG": "恒指ETF",
    "159980.XSHE": "有色ETF大成",
    "561360.XSHG": "石油ETF",
    "511260.XSHG": "十年国债ETF",
    # v2: 新增ETF
    "561550.XSHG": "中证500ETF",
    "159259.XSHE": "成长ETF",
    "159263.XSHE": "价值ETF",
}
BOND_ETFS = {"511260.XSHG"}
A_SHARE_ETFS = {
    "159915.XSHE",  # 创业板
    "561550.XSHG",  # 中证500
    "159259.XSHE",  # 成长
    "159263.XSHE",  # 价值
}
INDUSTRY_ETFS = {"159980.XSHE", "561360.XSHG"}
PREMIUM_ETFS = {"513100.XSHG"}

# 策略参数
PARAMS = {
    "momentum_window": 25,
    "momentum_lookback": 25,
    "quality_window": 25,
    "quality_lookback": 25,
    "cov_window": 60,
    "garch_window": 120,
    "volume_short_window": 5,
    "volume_long_window": 20,
    "score_weight_momentum": 0.3,
    "score_weight_quality": 0.7,
    "max_holdings": 6,
    "use_rank_scoring": True,
    "quality_floor": 0.4,
    "momentum_floor": 0.0,
    "signal_power": 1.4,
    "use_window_mean": True,
    "max_weight": 0.85,
    "max_weight_high": 0.85,
    "max_weight_low": 0.75,
    "max_weight_dd_threshold": 0.05,
    "use_dynamic_max_weight": True,
    "max_a_share_holdings": 1,
    "a_share_weight_cap": 0.5,
    "a_share_weight_cap_high": 0.5,
    "a_share_weight_cap_low": 0.35,
    "a_share_dd_threshold": 0.05,
    "use_dynamic_a_share_cap": True,
    "min_holdings": 1,
    "min_holdings_risk": 2,
    "min_holdings_dd_threshold": 0.05,
    "use_dynamic_min_holdings": True,
    "avoid_industry": False,
    "max_industry_holdings": 1,
    "industry_weight_cap": 0.35,
    "industry_weight_cap_high": 0.35,
    "industry_weight_cap_low": 0.25,
    "industry_dd_threshold": 0.05,
    "use_dynamic_industry_cap": True,
    "industry_penalty": 0.85,
    "trend_penalty": 0.6,
    "trend_window": 20,
    "max_dd_filter": 0.08,
    "use_bond_floor": False,  # True=启用保底机制，False=完全排除债券
    "min_bond_weight": 0.15,
    "max_bond_weight": 0.40,
    # 信号筛选参数（与回测一致）
    "momentum_floor": 0.0,
    "quality_floor": 0.4,
    "use_trend_filter": False,
    "trend_hard_filter": False,
    # EPO参数
    "epo_risk_aversion": 8.0,
    "anchor_weight": 0.1,  # 锚定信号权重
    "use_ledoit_wolf": True,
    # v2: 可选，设置收缩强度的上下限（Ledoit-Wolf自动计算的结果会被限制在此范围内）
    "shrinkage_floor": 0.05,  # 最小收缩强度（0表示完全信任样本协方差）
    "shrinkage_cap": 0.3,  # 最大收缩强度（1表示完全使用目标矩阵）
    # v2: 目标波动率（0表示不使用，保持满仓）- 与回测环境一致
    "target_volatility": 0,
    "volume_ratio_threshold": 1.6,
    "volume_penalty_power": 0.8,
    "use_relative_crowding": True,
    "relative_crowding_power": 1.0,
    "relative_crowding_floor": 0.6,
    "relative_crowding_ceiling": 1.6,
    # 回撤惩罚参数（与回测一致）
    "dd_penalty_threshold": 0.05,
    "dd_penalty_power": 1.0,
    "dd_penalty_floor": 0.6,
    # v2: 信号权重模式
    "signal_weight_mode": "fixed",  # fixed/trend/strength
}


# ============ 工具函数 ============
def _print_section(title, char="="):
    """打印分隔线"""
    print(f"\n{char * 70}")
    print(f" {title}")
    print(f"{char * 70}")


def _print_table(df, title=None, float_fmt=".4f"):
    """美观的表格打印"""
    if title:
        print(f"\n{title}")
        print("-" * 70)

    if df.empty:
        print("(空数据)")
        return

    # 格式化数值
    df_str = df.copy()
    for col in df_str.columns:
        if df_str[col].dtype in ["float64", "float32"]:
            df_str[col] = df_str[col].apply(
                lambda x: f"{x:{float_fmt}}" if pd.notna(x) else "N/A"
            )

    print(df_str.to_string())
    print()


def _rank_score(series, higher_is_better=True):
    if series.empty:
        return pd.Series(0.0, index=series.index)
    rank = series.rank(pct=True, ascending=True)
    return rank if higher_is_better else 1 - rank


def _calc_sharpe(returns):
    if len(returns) < 2:
        return 0.0
    mean, std = np.mean(returns), np.std(returns)
    return (mean / std) * math.sqrt(252) if std > 0 else 0.0


def _calc_max_drawdown(prices):
    if len(prices) < 2:
        return 0.0
    running_max = np.maximum.accumulate(prices)
    drawdowns = prices / running_max - 1
    return abs(np.min(drawdowns))


def _calc_volatility(returns):
    return np.std(returns) * math.sqrt(252) if len(returns) >= 2 else 0.0


def _calc_vol_stability(returns):
    if len(returns) < 10:
        return np.std(returns)
    vol_window = max(5, len(returns) // 5)
    rolling_vol = pd.Series(returns).rolling(vol_window).std().dropna()
    return np.std(rolling_vol.values) if not rolling_vol.empty else np.std(returns)


def _calc_volume_stability(volumes):
    """计算成交量稳定性（与回测环境一致）"""
    volumes = np.array(volumes, dtype=float)
    if len(volumes) < 2:
        return 0.0
    prev = volumes[:-1]
    prev[prev == 0] = np.nan
    volume_returns = (volumes[1:] - prev) / prev
    volume_returns = volume_returns[~np.isnan(volume_returns)]
    if len(volume_returns) == 0:
        return 0.0
    return np.std(volume_returns)


def _calc_log_return(prices):
    return (
        math.log(prices[-1] / prices[0]) if len(prices) >= 2 and prices[0] > 0 else 0.0
    )


def _calc_r2_log_prices(prices):
    if len(prices) < 2 or np.any(prices <= 0):
        return 0.0
    log_prices = np.log(prices)
    x = np.arange(len(log_prices))
    slope, intercept = np.polyfit(x, log_prices, 1)
    fitted = slope * x + intercept
    ss_res = np.sum((log_prices - fitted) ** 2)
    ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _calc_momentum(prices):
    if len(prices) < 2 or np.any(prices <= 0):
        return 0.0
    log_prices = np.log(prices)
    slope = np.polyfit(np.arange(len(log_prices)), log_prices, 1)[0]
    annual_ret = math.exp(slope * 252) - 1
    r2 = _calc_r2_log_prices(prices)
    return annual_ret * r2


def _calc_volume_ratio(volume, short_win=5, long_win=20):
    if len(volume) < long_win:
        return 1.0
    short_avg = np.mean(volume[-short_win:])
    long_avg = np.mean(volume[-long_win:])
    return short_avg / long_avg if long_avg > 0 else 1.0


def _calc_trend_filter(close, trend_window=20, max_dd_filter=0.08):
    if len(close) < trend_window:
        return True, 0.0
    window = close[-trend_window:]
    ma = float(np.mean(window))
    dd = _calc_max_drawdown(window)
    return close[-1] >= ma and dd <= max_dd_filter, dd


def _ewma_volatility(returns, lam=0.94):
    """计算EWMA波动率（与回测环境一致）"""
    if len(returns) < 2:
        return 0.0
    var = returns[0] ** 2
    for r in returns[1:]:
        var = lam * var + (1 - lam) * (r**2)
    return math.sqrt(max(var, 0.0)) * math.sqrt(252)


def _forecast_volatility(returns):
    """预测波动率（与回测环境一致）"""
    returns = np.array(returns, dtype=float)
    if len(returns) < 10:
        return np.std(returns) * math.sqrt(252) if len(returns) > 1 else 0.0

    # 尝试使用arch库进行GARCH建模，失败则使用EWMA（与回测环境一致）
    try:
        from arch import arch_model

        model = arch_model(returns * 100, mean="Constant", vol="GARCH", p=1, q=1)
        res = model.fit(disp="off")
        forecast = res.forecast(horizon=1, reindex=False)
        var = float(forecast.variance.values[-1, 0])
        return math.sqrt(max(var, 0.0)) / 100 * math.sqrt(252)
    except Exception:
        # 回退到EWMA波动率（与回测环境一致）
        return _ewma_volatility(returns)


def _zscore(series):
    """计算Z-score，与回测环境一致"""
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _get_market_trend(returns_df):
    """
    基于收益率趋势判断市场状态（方案3：分层方法）

    第一层：趋势判断决定基础仓位
    - 牛市(bull)：满仓
    - 震荡市(consolidation)：7成仓
    - 熊市(bear)：3成仓

    原理：
    - 计算近期累计收益率，正向大→牛市，负向大→熊市
    - 结合趋势强度（类似Sharpe）判断趋势确定性

    Returns:
        'bull' / 'consolidation' / 'bear'
    """
    if returns_df.shape[0] < 20:
        return 'consolidation'  # 数据不足，默认震荡

    # 计算最近20日累计收益
    recent_returns = returns_df.iloc[-20:]

    # 等权组合的收益
    portfolio_returns = recent_returns.mean(axis=1)
    cum_return = (1 + portfolio_returns).prod() - 1

    # 趋势强度（类似Sharpe）
    std = portfolio_returns.std()
    trend_strength = portfolio_returns.mean() / std if std > 0 else 0
    trend_strength = trend_strength * np.sqrt(252)  # 年化

    # 判断逻辑
    if cum_return > 0.05 and trend_strength > 0.5:
        return 'bull'  # 明显上涨趋势
    elif cum_return < -0.05 and trend_strength < -0.5:
        return 'bear'  # 明显下跌趋势
    else:
        return 'consolidation'  # 震荡市




def _dynamic_a_share_cap(assets, factor_df):
    """动态调整A股权重上限"""
    if not PARAMS.get("use_dynamic_a_share_cap", False):
        return PARAMS["a_share_weight_cap"]
    if factor_df is None or factor_df.empty:
        return PARAMS["a_share_weight_cap"]
    a_assets = [a for a in assets if a in A_SHARE_ETFS]
    if not a_assets:
        return PARAMS["a_share_weight_cap"]
    dd = factor_df.loc[a_assets, "trend_dd"]
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    if dd.empty:
        return PARAMS["a_share_weight_cap"]
    dd_metric = float(np.nanpercentile(dd.values, 75))
    threshold = PARAMS.get("a_share_dd_threshold", None)
    low = PARAMS.get("a_share_weight_cap_low", PARAMS["a_share_weight_cap"])
    high = PARAMS.get("a_share_weight_cap_high", PARAMS["a_share_weight_cap"])
    if threshold is None:
        return PARAMS["a_share_weight_cap"]
    if dd_metric >= threshold:
        return low
    return high


def _dynamic_industry_cap(assets, factor_df):
    """动态调整行业权重上限"""
    if not PARAMS.get("use_dynamic_industry_cap", False):
        return PARAMS["industry_weight_cap"]
    if factor_df is None or factor_df.empty:
        return PARAMS["industry_weight_cap"]
    i_assets = [a for a in assets if a in INDUSTRY_ETFS]
    if not i_assets:
        return PARAMS["industry_weight_cap"]
    dd = factor_df.loc[i_assets, "trend_dd"]
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    if dd.empty:
        return PARAMS["industry_weight_cap"]
    dd_metric = float(np.nanpercentile(dd.values, 75))
    threshold = PARAMS.get("industry_dd_threshold", None)
    low = PARAMS.get("industry_weight_cap_low", PARAMS["industry_weight_cap"])
    high = PARAMS.get("industry_weight_cap_high", PARAMS["industry_weight_cap"])
    if threshold is None:
        return PARAMS["industry_weight_cap"]
    if dd_metric >= threshold:
        return low
    return high


def _dynamic_max_weight(assets, factor_df):
    """动态调整个股权重上限"""
    if not PARAMS.get("use_dynamic_max_weight", False):
        return PARAMS["max_weight"]
    if factor_df is None or factor_df.empty:
        return PARAMS["max_weight"]
    assets = list(assets)
    dd = factor_df.loc[assets, "trend_dd"]
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    if dd.empty:
        return PARAMS["max_weight"]
    dd_metric = float(np.nanpercentile(dd.values, 75))
    threshold = PARAMS.get("max_weight_dd_threshold", None)
    low = PARAMS.get("max_weight_low", PARAMS["max_weight"])
    high = PARAMS.get("max_weight_high", PARAMS["max_weight"])
    if threshold is None:
        return PARAMS["max_weight"]
    if dd_metric >= threshold:
        return low
    return high

def _apply_volatility_scaling(target_weights, returns_df, target_volatility):
    """
    方案3：分层方法 - 趋势判断 + 波动率微调

    第一层：趋势判断决定基础仓位
    - 牛市(bull)：满仓 100%
    - 震荡市(consolidation)：7成仓
    - 熊市(bear)：3成仓

    第二层：波动率在基础仓位上微调
    - 波动率高 → 打8折
    - 波动率低 → 不打折

    Args:
        target_weights: 目标权重字典 {etf: weight}
        returns_df: 收益矩阵 (T x n)
        target_volatility: 目标年化波动率（如0.12表示12%）

    Returns:
        缩放后的权重字典
    """
    if target_volatility <= 0:
        # 不使用波动率缩放，保持满仓
        return target_weights

    assets = list(target_weights.keys())
    if len(assets) == 0:
        return target_weights

    # 检查returns_df是否包含所有资产
    missing = set(assets) - set(returns_df.columns)
    if missing:
        print(f"【波动率缩放】缺少资产数据: {missing}，跳过缩放")
        return target_weights

    # ========== 第一层：趋势判断 ==========
    trend = _get_market_trend(returns_df)

    # 基础仓位配置
    if trend == 'bull':
        base_position = 1.0  # 牛市满仓
    elif trend == 'bear':
        base_position = 0.3  # 熊市3成仓
    else:
        base_position = 0.7  # 震荡市7成仓

    print(f"【波动率缩放】市场趋势={trend}, 基础仓位={base_position:.0%}")

    # ========== 第二层：波动率微调 ==========
    # 构建权重向量
    w = np.array([target_weights.get(a, 0) for a in assets])

    # 计算组合波动率
    try:
        cov = returns_df[assets].cov().values
        portfolio_var = w @ cov @ w
        portfolio_vol = math.sqrt(max(portfolio_var, 0)) * math.sqrt(252)
    except Exception as e:
        print(f"【波动率缩放】计算波动率失败: {e}，跳过缩放")
        return target_weights

    if portfolio_vol <= 0:
        print(f"【波动率缩放】组合波动率为0，跳过缩放")
        return target_weights

    # 波动率高时微调（打8折）
    if portfolio_vol > target_volatility * 1.2:
        vol_adjustment = 0.8
    else:
        vol_adjustment = 1.0

    # 最终仓位 = 基础仓位 × 波动率微调
    final_position = base_position * vol_adjustment
    final_position = min(final_position, 1.0)  # 最大满仓

    # 打印调试信息
    print(f"【波动率缩放】组合波动率={portfolio_vol:.2%}, 目标={target_volatility:.2%}, 波动微调={vol_adjustment:.0%}, 最终仓位={final_position:.0%}")

    # 应用缩放
    scaled_weights = {a: w[i] * final_position for i, a in enumerate(assets)}

    return scaled_weights


def _get_signal_weights(mode, returns_df=None, signals=None):
    """
    根据不同模式计算动量/质量权重

    Args:
        mode: 信号权重模式
            'fixed': 使用固定权重(score_weight_momentum, score_weight_quality)
            'trend': 根据趋势调整（牛市动量高，熊市质量高）
            'strength': 根据信号强度调整（强信号动量高，弱信号质量高）
        returns_df: 收益率矩阵（用于趋势模式）
        signals: 信号向量（用于信号强度模式）

    Returns:
        (momentum_weight, quality_weight)
    """
    if mode == 'fixed':
        # 固定权重：使用参数中的值
        return PARAMS["score_weight_momentum"], PARAMS["score_weight_quality"]

    elif mode == 'trend':
        # 方案2：根据趋势调整
        if returns_df is None:
            print("【信号权重】trend模式需要returns_df，退回固定权重")
            return PARAMS["score_weight_momentum"], PARAMS["score_weight_quality"]

        trend = _get_market_trend(returns_df)

        if trend == 'bull':
            # 牛市：动量权重高
            momentum_w, quality_w = 0.7, 0.3
        elif trend == 'bear':
            # 熊市：质量权重高
            momentum_w, quality_w = 0.3, 0.7
        else:
            # 震荡市：平衡
            momentum_w, quality_w = 0.5, 0.5

        print(f"【信号权重】趋势={trend}, 动量权重={momentum_w}, 质量权重={quality_w}")
        return momentum_w, quality_w

    elif mode == 'strength':
        # 方案3：根据信号强度调整
        if signals is None:
            print("【信号权重】strength模式需要signals，退回固定权重")
            return PARAMS["score_weight_momentum"], PARAMS["score_weight_quality"]

        avg_signal = np.mean(signals)

        if avg_signal > 0.7:
            # 信号都很强：动量权重高
            momentum_w, quality_w = 0.7, 0.3
        elif avg_signal < 0.4:
            # 信号都很弱：质量权重高
            momentum_w, quality_w = 0.3, 0.7
        else:
            # 中等信号：平衡
            momentum_w, quality_w = 0.5, 0.5

        print(f"【信号权重】信号强度={avg_signal:.3f}, 动量权重={momentum_w}, 质量权重={quality_w}")
        return momentum_w, quality_w

    else:
        print(f"【信号权重】未知模式={mode}，使用固定权重")
        return PARAMS["score_weight_momentum"], PARAMS["score_weight_quality"]


def _build_anchor_signal(etfs, price_cache):
    """构建锚定信号（基于波动率，与回测环境一致）"""
    vol_forecasts = []
    for etf in etfs:
        data = price_cache.get(etf)
        if data is None or data.empty:
            vol_forecasts.append(np.nan)
            continue
        rets = data["close"].pct_change().dropna().tail(PARAMS["garch_window"]).values
        vol = _forecast_volatility(rets)
        vol_forecasts.append(vol)

    series = pd.Series(vol_forecasts, index=etfs)
    series = series.replace([np.inf, -np.inf], np.nan)
    
    # 添加详细日志
    print(f"\n【锚定信号计算日志】")
    print(f"ETF列表: {etfs}")
    print(f"波动率预测值: {dict(zip(etfs, [f'{v:.6f}' for v in vol_forecasts]))}")
    print(f"处理后波动率: {dict(zip(series.index, [f'{v:.6f}' for v in series.values]))}")
    
    if series.isna().all():
        print("所有波动率值为NaN，返回全0数组")
        return np.zeros(len(etfs))
    
    series = series.fillna(series.median())
    print(f"填充NaN后波动率: {dict(zip(series.index, [f'{v:.6f}' for v in series.values]))}")
    
    z_scores = _zscore(-series).values
    print(f"最终锚定信号: {dict(zip(etfs, [f'{s:.6f}' for s in z_scores]))}")
    
    return z_scores


def _rolling_metric_on_prices(prices, window, func):
    prices = np.array(prices, dtype=float)
    if len(prices) < 2:
        return 0.0
    if len(prices) < window or not PARAMS["use_window_mean"]:
        return func(prices[-window:]) if len(prices) >= window else func(prices)
    values = [func(prices[i - window : i]) for i in range(window, len(prices) + 1)]
    return float(np.mean(values)) if values else func(prices)


def _compute_metrics(close, volume):
    """计算ETF指标（与回测环境完全一致）"""
    q_lookback, q_win = PARAMS["quality_lookback"], PARAMS["quality_window"]
    m_lookback = PARAMS["momentum_lookback"]

    q_prices = close[-q_lookback:]
    m_prices = close[-m_lookback:]
    q_volume = volume[-q_lookback:]

    # 质量指标
    sharpe = _rolling_metric_on_prices(
        q_prices, q_win, lambda p: _calc_sharpe(np.diff(p) / p[:-1])
    )
    max_dd = _rolling_metric_on_prices(q_prices, q_win, _calc_max_drawdown)
    volatility = _rolling_metric_on_prices(
        q_prices, q_win, lambda p: _calc_volatility(np.diff(p) / p[:-1])
    )
    vol_stability = _rolling_metric_on_prices(
        q_prices, q_win, lambda p: _calc_vol_stability(np.diff(p) / p[:-1])
    )
    # 成交量稳定性（回测环境有此指标）
    volume_stability = _calc_volume_stability(q_volume)
    log_ret = _rolling_metric_on_prices(q_prices, q_win, _calc_log_return)
    r2 = _rolling_metric_on_prices(q_prices, q_win, _calc_r2_log_prices)

    # 动量指标
    momentum = _rolling_metric_on_prices(
        m_prices, PARAMS["momentum_window"], _calc_momentum
    )
    vol_ratio = _calc_volume_ratio(
        volume, PARAMS["volume_short_window"], PARAMS["volume_long_window"]
    )
    trend_ok, trend_dd = _calc_trend_filter(
        close, PARAMS["trend_window"], PARAMS["max_dd_filter"]
    )

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "volatility": volatility,
        "vol_stability": vol_stability,
        "volume_stability": volume_stability,  # 新增：与回测一致
        "log_return": log_ret,
        "r2": r2,
        "momentum": momentum,
        "volume_ratio": vol_ratio,
        "trend_ok": trend_ok,
        "trend_dd": trend_dd,
    }


# ============ 核心计算函数 ============
def calculate_for_date(calc_date_str, verbose=True):
    """
    计算指定日期的交易标的和权重

    Args:
        calc_date_str: 日期字符串 'YYYY-MM-DD'
        verbose: 是否打印详情

    Returns:
        dict: {'success': bool, 'date': str, 'target_weights': dict, 'factor_df': DataFrame, ...}
    """
    try:
        calc_date = pd.to_datetime(
            calc_date_str
        ).normalize()  # 转换为日期，去掉时间部分

        # 获取前一交易日（严格在调仓日之前）
        # 方式：获取调仓日之前的所有交易日，取最后一个
        trade_days = get_trade_days(end_date=calc_date, count=10)
        if len(trade_days) < 2:
            return {"success": False, "message": f"无法获取{calc_date_str}的前一交易日"}

        # 确保 trade_days 是 Timestamp 类型用于比较
        trade_days = pd.to_datetime(trade_days)

        # 找到严格在 calc_date 之前的交易日
        prev_trade_days = trade_days[trade_days < calc_date]
        if len(prev_trade_days) == 0:
            return {
                "success": False,
                "message": f"无法获取{calc_date_str}之前的历史数据",
            }
        prev_day = prev_trade_days[-1]

        if verbose:
            _print_section(f"ETF-EPO策略计算: {calc_date_str}")
            print(f"调仓日期: {calc_date_str}")
            print(f"数据截止: {prev_day} (前一交易日收盘，不含调仓日)")
            print(f"债券保底: {'启用' if PARAMS['use_bond_floor'] else '禁用'}")

        # 获取价格数据
        need_days = (
            max(
                PARAMS["quality_lookback"],
                PARAMS["momentum_lookback"],
                PARAMS["cov_window"],
                PARAMS["volume_long_window"],
                PARAMS["garch_window"],
            )
            + 5
        )
        min_len = (
            max(
                PARAMS["momentum_window"],
                PARAMS["quality_window"],
                PARAMS["volume_long_window"],
            )
            + 1
        )

        price_cache = {}
        missing_etfs = []

        if verbose:
            print(f"\n数据获取: 需要{need_days}天历史数据")

        for etf in ETF_POOL:
            data = get_price(
                etf,
                count=need_days,
                end_date=prev_day,
                frequency="daily",
                fields=["close", "volume", "money"],
            )
            if data is None or data.empty:
                missing_etfs.append(etf)
                continue
            data = data.dropna()
            if len(data) >= min_len:
                price_cache[etf] = data
            else:
                missing_etfs.append(etf)

        if verbose:
            print(f"✓ 成功获取: {len(price_cache)}只ETF数据")
            if missing_etfs:
                missing_names = [str(ETF_NAME_MAP.get(e, e)) for e in missing_etfs]
                print(f"✗ 数据缺失: {', '.join(missing_names)}")

        if not price_cache:
            return {"success": False, "message": "无价格数据"}

        # 计算指标
        metrics = {}
        for etf, data in price_cache.items():
            close = data["close"].values
            turnover = (
                data["money"].fillna(0).values
                if "money" in data
                else data["volume"].fillna(0).values
            )
            metrics[etf] = _compute_metrics(close, turnover)

        df = pd.DataFrame(metrics).T
        if df.empty:
            return {"success": False, "message": "指标计算失败"}

        # 计算因子得分（与回测环境完全一致）
        if verbose:
            print("\n【诊断日志：原始指标值】")
            print(
                "ETF               momentum    sharpe     max_dd    volatility  log_return  r2"
            )
            print("-" * 90)
            for etf in df.index:
                print(
                    f"{etf:<18} {df.loc[etf, 'momentum']:>10.4f} {df.loc[etf, 'sharpe']:>10.4f} "
                    f"{df.loc[etf, 'max_drawdown']:>10.4f} {df.loc[etf, 'volatility']:>10.4f} "
                    f"{df.loc[etf, 'log_return']:>10.4f} {df.loc[etf, 'r2']:>10.4f}"
                )

        df["momentum_score"] = _rank_score(df["momentum"], True)
        df["sharpe_score"] = _rank_score(df["sharpe"], True)
        df["mdd_score"] = _rank_score(df["max_drawdown"], False)
        df["vol_score"] = _rank_score(df["volatility"], False)
        df["vol_stability_score"] = _rank_score(df["vol_stability"], False)
        df["volume_stability_score"] = _rank_score(df["volume_stability"], False)
        df["logret_score"] = _rank_score(df["log_return"], True)
        df["r2_score"] = _rank_score(df["r2"], True)

        if verbose:
            print("\n【诊断日志：百分位排名得分】")
            print(
                "ETF               momentum_s  sharpe_s   mdd_s      vol_s      logret_s   r2_s"
            )
            print("-" * 90)
            for etf in df.index:
                print(
                    f"{etf:<18} {df.loc[etf, 'momentum_score']:>10.4f} {df.loc[etf, 'sharpe_score']:>10.4f} "
                    f"{df.loc[etf, 'mdd_score']:>10.4f} {df.loc[etf, 'vol_score']:>10.4f} "
                    f"{df.loc[etf, 'logret_score']:>10.4f} {df.loc[etf, 'r2_score']:>10.4f}"
                )

        # 质量分计算（与回测完全一致，包含volume_stability_score和volume_stability_score）
        df["quality_score"] = df[
            [
                "sharpe_score",
                "mdd_score",
                "vol_score",
                "vol_stability_score",
                "volume_stability_score",
                "logret_score",
                "r2_score",
            ]
        ].mean(axis=1)

        # 信号计算 - v2: 支持动态权重模式
        signal_weight_mode = PARAMS.get("signal_weight_mode", "fixed")
        if signal_weight_mode != "fixed":
            momentum_w, quality_w = _get_signal_weights(
                signal_weight_mode, returns_df, df["momentum_score"].values
            )
            df["signal"] = momentum_w * df["momentum_score"] + quality_w * df["quality_score"]
        else:
            df["signal"] = (
                PARAMS["score_weight_momentum"] * df["momentum_score"]
                + PARAMS["score_weight_quality"] * df["quality_score"]
            )

        if verbose:
            print("\n【诊断日志：质量分和综合信号】")
            print("ETF               quality_score  momentum_s  quality_s   signal")
            print("-" * 70)
            for etf in df.index:
                print(
                    f"{etf:<18} {df.loc[etf, 'quality_score']:>12.4f}  "
                    f"{df.loc[etf, 'momentum_score']:>8.4f}  "
                    f"{df.loc[etf, 'quality_score']:>8.4f}  "
                    f"{df.loc[etf, 'signal']:>8.6f}"
                )

        # 债券特殊处理：只使用动量分
        bond_mask = df.index.isin(BOND_ETFS)
        df.loc[bond_mask, "signal"] = df.loc[bond_mask, "momentum_score"]

        # 行业惩罚
        if PARAMS["industry_penalty"] < 1:
            df.loc[df.index.isin(INDUSTRY_ETFS), "signal"] *= PARAMS["industry_penalty"]
        if PARAMS["trend_penalty"] < 1:
            df.loc[~df["trend_ok"], "signal"] *= PARAMS["trend_penalty"]

        if verbose:
            _print_section("因子信号排名", "-")
            display_cols = [
                "signal",
                "momentum_score",
                "quality_score",
                "volume_ratio",
                "trend_ok",
            ]
            display_df = df[display_cols].copy()
            display_df["name"] = [ETF_NAME_MAP.get(e, e) for e in display_df.index]
            display_df = display_df.sort_values("signal", ascending=False)

            print(
                f"{'排名':<4} {'ETF':<12} {'综合信号':<10} {'动量分':<8} {'质量分':<8} {'放量比':<8} {'趋势':<6}"
            )
            print("-" * 70)
            for i, (etf, row) in enumerate(display_df.iterrows(), 1):
                trend = "✓" if row["trend_ok"] else "✗"
                print(
                    f"{i:<4} {row['name']:<12} {row['signal']:<10.4f} {row['momentum_score']:<8.4f} "
                    f"{row['quality_score']:<8.4f} {row['volume_ratio']:<8.2f} {trend:<6}"
                )

        # 信号筛选（与回测环境完全一致）
        if PARAMS["use_rank_scoring"]:
            # 质量分筛选：质量分必须 >= quality_floor
            mask = (df["momentum"] > PARAMS["momentum_floor"]) & (
                df["quality_score"] >= PARAMS["quality_floor"]
            )
            # 可选：趋势硬过滤
            if PARAMS.get("use_trend_filter") and PARAMS.get("trend_hard_filter", True):
                mask &= df["trend_ok"]
            signal_series = df.loc[mask, "signal"]
        else:
            signal_series = df["signal"]
            signal_series = signal_series[signal_series > 0]

        # 债券开关逻辑：如果未启用保底机制，完全排除债券ETF
        if not PARAMS["use_bond_floor"] and BOND_ETFS:
            signal_series = signal_series[~signal_series.index.isin(BOND_ETFS)]

        signal_series = signal_series.sort_values(ascending=False)

        if signal_series.empty:
            return {"success": False, "message": "无有效信号"}

        # 候选选择（与回测环境一致）
        min_holdings = PARAMS["min_holdings"]
        if PARAMS.get("use_dynamic_min_holdings") and df is not None and not df.empty:
            dd = df.loc[signal_series.index, "trend_dd"]
            dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
            if not dd.empty:
                dd_metric = float(np.nanpercentile(dd.values, 75))
                if dd_metric >= PARAMS.get("min_holdings_dd_threshold", 0.05):
                    min_holdings = PARAMS.get("min_holdings_risk", 2)
        signal_series = signal_series.sort_values(ascending=False)
        selected = []
        a_count, i_count = 0, 0
        max_a = PARAMS["max_a_share_holdings"]
        max_i = PARAMS.get("max_industry_holdings", 1)
        avoid_industry = PARAMS.get("avoid_industry", False)

        for etf in signal_series.index:
            if avoid_industry and etf in INDUSTRY_ETFS:
                continue
            if max_i is not None and etf in INDUSTRY_ETFS:
                if i_count >= max_i:
                    continue
                i_count += 1
            if etf in A_SHARE_ETFS:
                if a_count >= max_a:
                    continue
                a_count += 1
            selected.append(etf)
            if PARAMS.get("max_holdings") and len(selected) >= PARAMS["max_holdings"]:
                break

        # fallback: 如果持仓数少于最小要求，补充候选
        if min_holdings and len(selected) < min_holdings:
            fallback = df["signal"].sort_values(ascending=False)
            for etf in fallback.index:
                if etf in selected:
                    continue
                if avoid_industry and etf in INDUSTRY_ETFS:
                    continue
                if max_i is not None and etf in INDUSTRY_ETFS:
                    if i_count >= max_i:
                        continue
                    i_count += 1
                if etf in A_SHARE_ETFS and a_count >= max_a:
                    continue
                if etf in A_SHARE_ETFS:
                    a_count += 1
                selected.append(etf)
                if len(selected) >= min_holdings:
                    break

        if verbose:
            print(f"\n候选池: {len(selected)}只ETF")
            for i, etf in enumerate(selected, 1):
                name = ETF_NAME_MAP.get(etf, etf)
                signal = signal_series[etf]
                print(f"  {i}. {name} (信号: {signal:.4f})")

        if not selected:
            return {"success": False, "message": "无候选标的"}

        # 构建收益矩阵
        returns_list = []
        for etf in selected:
            if etf in price_cache:
                rets = (
                    price_cache[etf]["close"]
                    .pct_change()
                    .dropna()
                    .tail(PARAMS["cov_window"])
                )
                if not rets.empty:
                    returns_list.append(rets.rename(etf))

        if not returns_list:
            return {"success": False, "message": "无法构建收益矩阵"}

        returns_df = pd.concat(returns_list, axis=1, join="inner").dropna()
        if returns_df.empty or returns_df.shape[1] == 0:
            return {"success": False, "message": "收益矩阵为空"}

        # EPO权重计算
        actual_selected = list(returns_df.columns)
        raw_signals = df.loc[actual_selected, "signal"].values

        # 【增强】与研究环境一致的详细日志
        if verbose:
            print("\n" + "=" * 70)
            print("【研究环境：信号处理详细日志】")
            print(f"候选ETF: {actual_selected}")
            print(
                f"原始信号值（来自df）: {dict(zip(actual_selected, [f'{s:.6f}' for s in raw_signals]))}"
            )

        # 【修复】添加锚定信号处理（与回测环境一致）
        signals = raw_signals.copy()
        if PARAMS["anchor_weight"] > 0:
            anchor_signal = _build_anchor_signal(actual_selected, price_cache)
            signals = (1 - PARAMS["anchor_weight"]) * signals + PARAMS[
                "anchor_weight"
            ] * anchor_signal

            if verbose:
                print(
                    f"锚定信号: {dict(zip(actual_selected, [f'{s:.6f}' for s in anchor_signal]))}"
                )
                print(
                    f"混合后信号（anchor_weight={PARAMS['anchor_weight']}）: {dict(zip(actual_selected, [f'{s:.6f}' for s in signals]))}"
                )

        clipped_signals = np.clip(signals, 0, None)
        if verbose:
            print(
                f"裁剪后信号: {dict(zip(actual_selected, [f'{s:.6f}' for s in clipped_signals]))}"
            )

        if PARAMS["signal_power"] != 1.0:
            signals = np.power(clipped_signals, PARAMS["signal_power"])
            if verbose:
                print(
                    f"幂次{PARAMS['signal_power']}后信号: {dict(zip(actual_selected, [f'{s:.6f}' for s in signals]))}"
                )
        else:
            signals = clipped_signals

        if verbose:
            print("=" * 70)

        if verbose:
            print("\n【诊断日志：EPO计算输入】")
            print(f"候选ETF: {actual_selected}")
            # 【修复】显示正确的处理阶段信息
            if PARAMS["anchor_weight"] > 0:
                print("信号值（已包含锚定信号混合）:")
            else:
                print("信号值（原始信号，无锚定混合）:")
            print(f"  {dict(zip(actual_selected, [f'{s:.6f}' for s in signals]))}")
            print(f"风险厌恶系数: {PARAMS['epo_risk_aversion']}")
            print(f"信号幂次: {PARAMS['signal_power']}")
            print(f"锚定信号权重: {PARAMS['anchor_weight']}")

        # ========== 详细调试信息 ==========
        if verbose:
            print(f"\n【EPO计算详细日志】")
            print(f"候选ETF: {actual_selected}")
            print(
                f"信号值: {dict(zip(actual_selected, [f'{s:.6f}' for s in signals]))}"
            )
            print(f"\n收益矩阵形状: {returns_df.shape}")
            print(f"收益矩阵列: {returns_df.columns.tolist()}")
            print(f"\n收益矩阵统计:")
            stats_df = pd.DataFrame(
                {
                    "mean": returns_df.mean(),
                    "std": returns_df.std(),
                    "min": returns_df.min(),
                    "max": returns_df.max(),
                }
            )
            print(stats_df.to_string())

        # Ledoit-Wolf协方差估计
        try:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf(assume_centered=True)
            lw.fit(returns_df.values)
            cov_matrix = lw.covariance_
            # 【修复】与回测环境一致的shrinkage限制
            shrinkage_floor = PARAMS.get("shrinkage_floor", 0.0)
            shrinkage_cap = PARAMS.get("shrinkage_cap", 1.0)
            shrinkage = float(np.clip(lw.shrinkage_, shrinkage_floor, shrinkage_cap))
        except:
            cov_matrix = returns_df.cov().values
            shrinkage = 0.0

        n = len(actual_selected)
        cov_matrix = cov_matrix + np.eye(n) * 1e-6

        if verbose:
            print(f"\n协方差矩阵:")
            cov_df = pd.DataFrame(
                cov_matrix, index=actual_selected, columns=actual_selected
            )
            print(cov_df.to_string())
            print(f"\n协方差矩阵对角线 (方差):")
            for etf in actual_selected:
                idx = actual_selected.index(etf)
                print(f"  {etf}: {cov_matrix[idx, idx]:.8f}")

        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except:
            inv_cov = np.linalg.pinv(cov_matrix)

        if verbose:
            print(f"\n逆协方差矩阵:")
            inv_df = pd.DataFrame(
                inv_cov, index=actual_selected, columns=actual_selected
            )
            print(inv_df.to_string())

        raw = inv_cov.dot(signals)
        if verbose:
            print(f"\n逆协方差 × 信号 (未归一化):")
            for etf, val in zip(actual_selected, raw):
                print(f"  {etf}: {val:.8f}")

        # v2: 与回测环境一致，不再使用 epo_risk_aversion（已废弃）
        # 直接返回未归一化的raw权重，让调用方统一归一化
        raw = np.maximum(0, raw)
        if verbose:
            print(f"\n仅做多约束后:")
            for etf, val in zip(actual_selected, raw):
                print(f"  {etf}: {val:.8f}")

        # 归一化
        total = np.sum(raw)
        weights = raw / total if total > 0 else np.ones(n) / n

        if verbose:
            print(f"\nEPO权重 (归一化后):")
            for etf, val in zip(actual_selected, weights):
                print(f"  {etf}: {val * 100:.4f}%")
            print(f"权重总和: {np.sum(weights) * 100:.4f}%")

        # 过滤权重太小的 ETF（模拟回测环境的 _adjust_weights_for_trading）
        # 如果某个 ETF 权重太小（< 1%），重新归一化后重新分配
        min_weight_threshold = 0.01  # 1% 以下的 ETF 被过滤
        filtered = [etf for etf, w in zip(actual_selected, weights) if w < min_weight_threshold]
        if filtered:
            if verbose:
                print(f"\n【ETF过滤】过滤掉权重<1%的ETF: {filtered}")
            mask = np.array([w >= min_weight_threshold for w in weights])
            weights = weights[mask]
            actual_selected = [etf for etf, m in zip(actual_selected, mask) if m]
            if len(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.array([1.0 / len(actual_selected)] * len(actual_selected))
            if verbose:
                print(f"过滤后剩余: {actual_selected}, 权重: {weights}")

        # 拥挤度惩罚
        # 打印关键指标用于调试
        if verbose:
            vr_dict = {etf: df.loc[etf, "volume_ratio"] for etf in actual_selected}
            dd_dict = {etf: df.loc[etf, "trend_dd"] for etf in actual_selected}
            print(f"\n【惩罚前诊断】")
            print(f"放量比(volume_ratio): {vr_dict}")
            print(f"回撤(trend_dd): {dd_dict}")
        
        penalties = []
        for etf in actual_selected:
            ratio = df.loc[etf, "volume_ratio"]
            pen = 1.0
            if ratio > PARAMS["volume_ratio_threshold"]:
                pen = ratio ** (-PARAMS["volume_penalty_power"])
            penalties.append(pen)

        if False:  # 临时禁用相对拥挤惩罚
            ratios = df.loc[actual_selected, "volume_ratio"].astype(float)
            median = np.nanmedian(ratios.values)
            if median and not np.isnan(median):
                rel = ratios.values / median
                rel = np.clip(
                    rel,
                    PARAMS["relative_crowding_floor"],
                    PARAMS["relative_crowding_ceiling"],
                )
                rel_pen = np.power(rel, -PARAMS["relative_crowding_power"])
                penalties = [p * rp for p, rp in zip(penalties, rel_pen)]

        # 回撤惩罚（与回测环境一致）
        dd = df.loc[actual_selected, "trend_dd"]
        dd = dd.replace([np.inf, -np.inf], np.nan)
        if PARAMS.get("dd_penalty_threshold") and not dd.isna().all():
            ratio = dd.values / PARAMS["dd_penalty_threshold"]
            dd_penalty = np.ones_like(ratio, dtype=float)
            mask = ratio > 1
            dd_penalty[mask] = ratio[mask] ** (-PARAMS.get("dd_penalty_power", 1.0))
            dd_penalty = np.clip(dd_penalty, PARAMS.get("dd_penalty_floor", 0.6), 1.0)
            penalties = [p * dp for p, dp in zip(penalties, dd_penalty)]

        # 调试：保存惩罚前权重用于对比
        weights_before_penalty = weights.copy()

        # 测试：打印惩罚前后的权重对比
        if verbose:
            print(f"\n【惩罚对比测试】")
            print(
                f"惩罚前权重: {dict(zip(actual_selected, [f'{w * 100:.2f}%' for w in weights]))}"
            )
            print(
                f"惩罚因子:   {dict(zip(actual_selected, [f'{p:.4f}' for p in penalties]))}"
            )

        # 临时调试：禁用所有惩罚测试纯EPO权重
        # 如需测试，取消下面这行的注释
        # penalties = [1.0] * len(penalties)

        # 打印实际应用的惩罚
        if verbose:
            print(f"\n【实际惩罚因子】（禁用相对拥挤后）: {dict(zip(actual_selected, [f'{p:.4f}' for p in penalties]))}")
        
        weights = weights * np.array(penalties)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights

        if verbose:
            print(
                f"惩罚后权重: {dict(zip(actual_selected, [f'{w * 100:.2f}%' for w in weights]))}"
            )

        # 分组权重限制 - 与回测环境完全一致
        def _normalize(weights):
            total = np.sum(weights)
            if total > 1e-12:
                return weights / total
            return np.ones_like(weights) / len(weights)

        def _apply_group_weight_cap(weights, assets, group_set, cap):
            """与回测环境完全一致的实现"""
            weights = np.array(weights, dtype=float)
            if cap is None or cap <= 0:
                return weights
            assets = list(assets)
            group_idx = [i for i, a in enumerate(assets) if a in group_set]
            if not group_idx:
                return weights
            group_weight = weights[group_idx].sum()
            if group_weight <= cap:
                return weights
            non_idx = [i for i in range(len(weights)) if i not in group_idx]
            if not non_idx:
                return weights
            scale = cap / max(group_weight, 1e-10)
            weights[group_idx] *= scale
            remaining = 1.0 - weights[group_idx].sum()
            non_weight = weights[non_idx].sum()
            if non_weight > 0:
                weights[non_idx] = weights[non_idx] / non_weight * remaining
            return weights

        def _apply_weight_cap(weights, cap):
            """与回测环境完全一致的实现"""
            weights = np.array(weights, dtype=float)
            if cap <= 0 or len(weights) == 0:
                return weights
            weights = _normalize(weights)
            n = len(weights)
            if cap * n < 1 - 1e-8:
                return weights

            capped = np.zeros_like(weights)
            remaining = 1.0
            active = np.ones(n, dtype=bool)
            base = weights.copy()
            for _ in range(n):
                if not active.any():
                    break
                total = base[active].sum()
                if total <= 0:
                    capped[active] = remaining / active.sum()
                    remaining = 0.0
                    break
                alloc = base[active] / total * remaining
                over = alloc > cap
                if not over.any():
                    capped[active] = alloc
                    remaining = 0.0
                    break
                idx = np.where(active)[0]
                over_idx = idx[over]
                capped[over_idx] = cap
                remaining -= cap * len(over_idx)
                active[over_idx] = False
            if remaining > 1e-8 and active.any():
                capped[active] += remaining / active.sum()
            return capped

        # A股权重限制（使用动态上限）
        weights_before_cap = weights.copy()
        a_share_cap = _dynamic_a_share_cap(actual_selected, df)
        if a_share_cap:
            weights = _apply_group_weight_cap(
                weights, actual_selected, A_SHARE_ETFS, a_share_cap
            )
            if verbose:
                print(f"\n【诊断日志：A股限制】")
                print(f"  A股ETF: {A_SHARE_ETFS}")
                print(f"  权重上限: {PARAMS['a_share_weight_cap'] * 100:.0f}%")
                print(
                    f"  限制前: {dict(zip(actual_selected, [f'{w * 100:.2f}%' for w in weights_before_cap]))}"
                )
                print(
                    f"  限制后: {dict(zip(actual_selected, [f'{w * 100:.2f}%' for w in weights]))}"
                )

        a_share_weight = sum(
            weights[i] for i, a in enumerate(actual_selected) if a in A_SHARE_ETFS
        )

        # 行业权重限制（使用动态上限）
        weights_before_industry = weights.copy()
        industry_cap = _dynamic_industry_cap(actual_selected, df)
        if industry_cap:
            weights = _apply_group_weight_cap(
                weights, actual_selected, INDUSTRY_ETFS, industry_cap
            )
            if verbose:
                print(f"\n【诊断日志：行业限制】")
                print(f"  行业ETF: {INDUSTRY_ETFS}")
                print(f"  权重上限: {PARAMS['industry_weight_cap'] * 100:.0f}%")
                print(
                    f"  限制前: {dict(zip(actual_selected, [f'{w * 100:.2f}%' for w in weights_before_industry]))}"
                )
                print(
                    f"  限制后: {dict(zip(actual_selected, [f'{w * 100:.2f}%' for w in weights]))}"
                )

        industry_weight = sum(
            weights[i] for i, a in enumerate(actual_selected) if a in INDUSTRY_ETFS
        )

        # 个股权重上限（使用动态上限）
        weights_before_individual = weights.copy()
        max_weight = _dynamic_max_weight(actual_selected, df)
        if max_weight:
            weights = _apply_weight_cap(weights, max_weight)

        # 债券保底
        target_weights = dict(zip(actual_selected, weights))
        if PARAMS["use_bond_floor"] and BOND_ETFS:
            bond_w = sum(target_weights.get(etf, 0) for etf in BOND_ETFS)
            min_bond, max_bond = PARAMS["min_bond_weight"], PARAMS["max_bond_weight"]

            # 如果债券权重超过上限，降低
            if bond_w > max_bond and bond_w > 0:
                scale = max_bond / bond_w
                for etf in BOND_ETFS:
                    if etf in target_weights:
                        target_weights[etf] *= scale
                # 分配多余的
                excess = bond_w - max_bond
                others = [e for e in target_weights if e not in BOND_ETFS]
                if others:
                    for etf in others:
                        target_weights[etf] += excess / len(others)

            # 如果债券权重低于保底，增加
            elif bond_w < min_bond:
                deficit = min_bond - bond_w
                others = [e for e in target_weights if e not in BOND_ETFS]
                others_w = sum(target_weights.get(e, 0) for e in others)

                if others_w > 0:
                    for etf in others:
                        target_weights[etf] = max(
                            0,
                            target_weights[etf]
                            - deficit * target_weights[etf] / others_w,
                        )

                # 分配保底给债券
                bond_signals = {
                    etf: df.loc[etf, "signal"] for etf in BOND_ETFS if etf in df.index
                }
                total_sig = sum(list(bond_signals.values()))
                for etf in BOND_ETFS:
                    if etf in df.index:
                        target_weights[etf] = min_bond * (
                            bond_signals[etf] / total_sig
                            if total_sig > 0
                            else 1 / len(BOND_ETFS)
                        )

        # 过滤接近0的并归一化
        target_weights = {k: v for k, v in target_weights.items() if v > 0.001}
        total = sum(list(target_weights.values()))

        if total > 0:
            target_weights = {k: v / total for k, v in target_weights.items()}

        # 计算分类统计
        category_weights = {}
        for etf, w in target_weights.items():
            cat = "其他"
            if etf in BOND_ETFS:
                cat = "债券"
            elif etf in A_SHARE_ETFS:
                cat = "A股"
            elif etf in INDUSTRY_ETFS:
                cat = "周期"
            elif etf == "518880.XSHG":
                cat = "商品"
            else:
                cat = "境外"
            category_weights[cat] = category_weights.get(cat, 0) + w

        if verbose:
            _print_section("最终调仓方案", "=")

            print(f"\n【候选ETF指标详情】")
            print(f"{'ETF':<15} {'原始信号':<10} {'成交量比':<10} {'趋势回撤':<10}")
            print("-" * 50)
            for etf in actual_selected:
                signal = df.loc[etf, "signal"]
                vol_ratio = df.loc[etf, "volume_ratio"]
                trend_dd = df.loc[etf, "trend_dd"]
                print(f"{etf:<15} {signal:<10.4f} {vol_ratio:<10.2f} {trend_dd:<10.4f}")

            print(f"\n【权重计算中间过程】")
            print(
                f"{'阶段':<20} {'黄金':<10} {'创业板':<10} {'纳指':<10} {'恒指':<10} {'有色':<10} {'石油':<10} {'国债':<10}"
            )
            print("-" * 110)

            # EPO原始权重（与回测环境一致，不再使用 epo_risk_aversion）
            epo_raw = inv_cov.dot(signals)
            epo_raw = np.maximum(0, epo_raw)
            epo_raw = (
                epo_raw / np.sum(epo_raw)
                if np.sum(epo_raw) > 0
                else np.ones(len(epo_raw)) / len(epo_raw)
            )
            raw_weights = dict(zip(actual_selected, epo_raw))
            print(
                f"{'1. EPO权重':<20} "
                + " ".join(
                    [f"{raw_weights.get(etf, 0) * 100:>8.2f}%" for etf in ETF_POOL]
                )
            )

            # 惩罚后权重
            pen_weights = dict(zip(actual_selected, weights_before_cap))
            print(
                f"{'2. 惩罚后权重':<20} "
                + " ".join(
                    [f"{pen_weights.get(etf, 0) * 100:>8.2f}%" for etf in ETF_POOL]
                )
            )

            # 计算并显示惩罚详情
            # 计算中位数用于显示
            ratios_arr = df.loc[actual_selected, "volume_ratio"].astype(float).values
            vol_median = np.nanmedian(ratios_arr)

            print(f"\n【惩罚计算详情】")
            print(
                f"成交量阈值: {PARAMS['volume_ratio_threshold']}, 惩罚幂: {PARAMS['volume_penalty_power']}"
            )
            print(
                f"{'ETF':<15} {'成交量比':<10} {'基础惩罚':<10} {'相对拥挤':<10} {'回撤惩罚':<10} {'总惩罚':<10}"
            )
            print("-" * 80)
            for i, etf in enumerate(actual_selected):
                vol_ratio = df.loc[etf, "volume_ratio"]
                base_pen = (
                    1.0
                    if vol_ratio <= PARAMS["volume_ratio_threshold"]
                    else vol_ratio ** (-PARAMS["volume_penalty_power"])
                )

                # 相对拥挤
                rel = vol_ratio / vol_median if vol_median > 0 else 1.0
                rel = np.clip(
                    rel,
                    PARAMS["relative_crowding_floor"],
                    PARAMS["relative_crowding_ceiling"],
                )
                rel_pen = rel ** (-PARAMS["relative_crowding_power"])

                # 回撤惩罚
                trend_dd = df.loc[etf, "trend_dd"]
                dd_thresh = PARAMS.get("dd_penalty_threshold", 0.05)
                dd_pen = 1.0
                if trend_dd > dd_thresh:
                    dd_pen = (trend_dd / dd_thresh) ** (
                        -PARAMS.get("dd_penalty_power", 1.0)
                    )
                    dd_pen = max(dd_pen, PARAMS.get("dd_penalty_floor", 0.6))

                total_pen = base_pen * rel_pen * dd_pen
                print(
                    f"{etf:<15} {vol_ratio:<10.2f} {base_pen:<10.4f} {rel_pen:<10.4f} {dd_pen:<10.4f} {total_pen:<10.4f}"
                )
            print(f"注: 相对拥挤基于成交量中位数 {vol_median:.2f} 计算")

            # A股限制后
            a_cap_weights = dict(zip(actual_selected, weights_before_industry))
            print(
                f"{'3. A股限制后':<20} "
                + " ".join(
                    [f"{a_cap_weights.get(etf, 0) * 100:>8.2f}%" for etf in ETF_POOL]
                )
                + f" (A股总: {a_share_weight * 100:.1f}%)"
            )

            # 行业限制后
            ind_weights = dict(zip(actual_selected, weights_before_individual))
            print(
                f"{'4. 行业限制后':<20} "
                + " ".join(
                    [f"{ind_weights.get(etf, 0) * 100:>8.2f}%" for etf in ETF_POOL]
                )
                + f" (行业总: {industry_weight * 100:.1f}%)"
            )

            # 个股权重上限后
            final_weights = dict(zip(actual_selected, weights))
            print(
                f"{'5. 个股权重上限后':<20} "
                + " ".join(
                    [f"{final_weights.get(etf, 0) * 100:>8.2f}%" for etf in ETF_POOL]
                )
            )

            print(f"\n【资产配置】")
            for cat, w in sorted(
                category_weights.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {cat}: {w * 100:.2f}%")

            print(f"\n【持仓明细】")
            print(f"{'ETF代码':<15} {'名称':<12} {'权重':<10} {'分类':<8} {'信号':<10}")
            print("-" * 70)
            for etf, w in sorted(
                target_weights.items(), key=lambda x: x[1], reverse=True
            ):
                name = ETF_NAME_MAP.get(etf, etf)
                signal = df.loc[etf, "signal"] if etf in df.index else 0
                cat = "其他"
                if etf in BOND_ETFS:
                    cat = "债券"
                elif etf in A_SHARE_ETFS:
                    cat = "A股"
                elif etf in INDUSTRY_ETFS:
                    cat = "周期"
                elif etf == "518880.XSHG":
                    cat = "商品"
                else:
                    cat = "境外"
                print(
                    f"{etf:<15} {name:<12} {w * 100:>6.2f}%   {cat:<8} {signal:<10.4f}"
                )

            print(f"\n【关键参数】")
            print(f"  Ledoit-Wolf收缩强度: {shrinkage:.4f}")
            print(
                f"  债券保底: {'启用' if PARAMS['use_bond_floor'] else '禁用'} (最低{PARAMS['min_bond_weight'] * 100:.0f}%)"
            )
            print(f"  A股权重上限: {PARAMS['a_share_weight_cap'] * 100:.0f}%")
            print(f"  行业权重上限: {PARAMS['industry_weight_cap'] * 100:.0f}%")
            print(f"  个股权重上限: {PARAMS['max_weight'] * 100:.0f}%")

            # v2: 目标波动率缩放
            target_volatility = PARAMS.get("target_volatility", 0.0)
            if target_volatility > 0:
                target_weights = _apply_volatility_scaling(
                    target_weights, returns_df, target_volatility
                )
                print(f"  目标波动率: {target_volatility:.2%}")

            print(f"{'=' * 70}\n")

        return {
            "success": True,
            "date": calc_date_str,
            "target_weights": target_weights,
            "factor_df": df,
            "selected_etfs": list(target_weights.keys()),
            "shrinkage": shrinkage,
            "category_weights": category_weights,
            "message": "计算成功",
        }

    except Exception as e:
        import traceback

        return {
            "success": False,
            "message": f"计算异常: {str(e)}\n{traceback.format_exc()}",
        }


# ============ 便捷函数 ============
def get_next_wednesday_trades(start_date, end_date, verbose=False):
    """
    批量获取所有周三的交易标的

    Args:
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        verbose: 是否打印详情

    Returns:
        DataFrame: 包含日期、ETF、权重的表格
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="W-WED")
    results = []

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        result = calculate_for_date(date_str, verbose=verbose)

        if result["success"]:
            for etf, weight in result["target_weights"].items():
                results.append(
                    {
                        "date": date_str,
                        "etf_code": etf,
                        "etf_name": ETF_NAME_MAP.get(etf, etf),
                        "weight": weight,
                        "weight_pct": f"{weight * 100:.2f}%",
                    }
                )

    return pd.DataFrame(results)


def calculate_rebalance_plan(
    current_positions,
    total_value,
    target_weights,
    current_prices,
    min_lot=100,
    verbose=True,
):
    """
    计算具体调仓方案

    Args:
        current_positions: 当前持仓 dict {etf: shares}
        total_value: 总资产
        target_weights: 目标权重 dict {etf: weight}
        current_prices: 当前价格 dict {etf: price}
        min_lot: 最小交易手数（默认100股）
        verbose: 是否打印详情

    Returns:
        dict: {
            'orders': 订单列表,
            'target_shares': 目标股数,
            'changes': 变更类型 dict {etf: 'BUY'/'SELL'/'HOLD'},
            'turnover': 预估成交金额
        }
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(" 调仓方案计算")
        print(f"{'=' * 70}")
        print(f"总资产: ¥{total_value:,.2f}")
        print(f"当前持仓: {len(current_positions)}只")
        print(f"目标持仓: {len(target_weights)}只")
        print(f"最小手数: {min_lot}股")

    # 1. 计算目标股数
    target_shares = {}
    for etf, weight in target_weights.items():
        price = current_prices.get(etf)
        if price is None or np.isnan(price) or price <= 0:
            continue
        target_value = total_value * weight
        # 计算股数（取整到最小手数）
        shares = int(target_value / price // min_lot) * min_lot
        if shares >= min_lot:
            target_shares[etf] = shares

    # 2. 生成调仓指令
    orders = []
    changes = {}
    turnover = 0.0

    # 2.1 卖出不在目标列表中的持仓
    for etf in current_positions:
        if etf not in target_shares:
            price = current_prices.get(etf, 0)
            current_shares = current_positions[etf]
            order_value = current_shares * price
            orders.append(
                {
                    "etf": etf,
                    "name": ETF_NAME_MAP.get(etf, etf),
                    "action": "SELL",
                    "current_shares": current_shares,
                    "target_shares": 0,
                    "price": price,
                    "value": order_value,
                }
            )
            changes[etf] = "SELL"
            turnover += order_value

    # 2.2 调整现有持仓
    for etf, target in target_shares.items():
        price = current_prices.get(etf, 0)
        current = current_positions.get(etf, 0)

        if target > current:
            # 买入
            order_value = (target - current) * price
            orders.append(
                {
                    "etf": etf,
                    "name": ETF_NAME_MAP.get(etf, etf),
                    "action": "BUY",
                    "current_shares": current,
                    "target_shares": target,
                    "price": price,
                    "value": order_value,
                }
            )
            changes[etf] = "BUY" if current == 0 else "ADD"
            turnover += order_value
        elif target < current:
            # 卖出
            order_value = (current - target) * price
            orders.append(
                {
                    "etf": etf,
                    "name": ETF_NAME_MAP.get(etf, etf),
                    "action": "SELL",
                    "current_shares": current,
                    "target_shares": target,
                    "price": price,
                    "value": order_value,
                }
            )
            changes[etf] = "SELL" if target == 0 else "REDUCE"
            turnover += order_value
        else:
            # 持有不变
            changes[etf] = "HOLD"

    # 3. 打印调仓方案
    if verbose and orders:
        print(f"\n【调仓指令】")
        print(
            f"{'操作':<6} {'ETF代码':<15} {'名称':<12} {'当前':>8} {'目标':>8} {'价格':>10} {'金额':>12}"
        )
        print("-" * 85)

        # 先打印卖出
        for order in [o for o in orders if o["action"] == "SELL"]:
            print(
                f"{'卖出':<6} {order['etf']:<15} {order['name']:<12} "
                f"{order['current_shares']:>8} {order['target_shares']:>8} "
                f"¥{order['price']:>8.2f} ¥{order['value']:>10,.0f}"
            )

        # 再打印买入
        for order in [o for o in orders if o["action"] == "BUY"]:
            print(
                f"{'买入':<6} {order['etf']:<15} {order['name']:<12} "
                f"{order['current_shares']:>8} {order['target_shares']:>8} "
                f"¥{order['price']:>8.2f} ¥{order['value']:>10,.0f}"
            )

        print(f"\n预估成交总额: ¥{turnover:,.2f}")
        print(f"换手率: {turnover / total_value * 100:.2f}%")

    # 4. 计算调仓后持仓
    new_positions = current_positions.copy()
    for order in orders:
        if order["action"] == "SELL" and order["target_shares"] == 0:
            new_positions.pop(order["etf"], None)
        else:
            new_positions[order["etf"]] = order["target_shares"]

    if verbose:
        print(f"\n【调仓后持仓】")
        new_total = sum(
            new_positions.get(e, 0) * current_prices.get(e, 0) for e in new_positions
        )
        print(f"持仓市值: ¥{new_total:,.2f}")
        print(f"现金预估: ¥{total_value - new_total:,.2f}")
        print(f"{'=' * 70}\n")

    return {
        "orders": orders,
        "target_shares": target_shares,
        "changes": changes,
        "turnover": turnover,
        "new_positions": new_positions,
    }


def generate_trade_summary(results_list, show_details=True):
    """
    生成详细的交易汇总报告

    Args:
        results_list: calculate_for_date返回的结果列表
        show_details: 是否打印详细表格

    Returns:
        DataFrame: 汇总统计
    """
    summary = []

    for r in results_list:
        if not r["success"]:
            continue

        date = r["date"]
        factor_df = r.get("factor_df", pd.DataFrame())
        target_weights = r.get("target_weights", {})
        shrinkage = r.get("shrinkage", 0)
        category_weights = r.get("category_weights", {})

        summary.append(
            {
                "date": date,
                "num_etfs": len(r["selected_etfs"]),
                "etfs": ", ".join(
                    [str(ETF_NAME_MAP.get(e, e)) for e in r["selected_etfs"]]
                ),
                "shrinkage": f"{shrinkage:.4f}" if shrinkage else "N/A",
            }
        )

        if show_details and not factor_df.empty:
            print(f"\n{'=' * 80}")
            print(f"【{date}】交易详情")
            print(f"{'=' * 80}")

            # 1. 因子信号排名表
            print(f"\n📊 因子信号排名 (Ledoit-Wolf收缩强度: {shrinkage:.4f})")
            print("-" * 80)

            # 准备显示数据
            display_df = factor_df.copy()
            display_df["name"] = [ETF_NAME_MAP.get(e, e) for e in display_df.index]
            display_df = display_df.sort_values("signal", ascending=False)

            # 打印表头
            print(
                f"{'排名':<4} {'ETF代码':<15} {'名称':<12} {'综合信号':<10} {'动量分':<8} {'质量分':<8} {'趋势':<6}"
            )
            print("-" * 80)

            # 打印数据行
            for i, (etf, row) in enumerate(display_df.iterrows(), 1):
                trend = "✓" if row.get("trend_ok", True) else "✗"
                print(
                    f"{i:<4} {etf:<15} {row['name']:<12} "
                    f"{row['signal']:<10.4f} {row['momentum_score']:<8.4f} "
                    f"{row['quality_score']:<8.4f} {trend:<6}"
                )

            # 2. 调仓方案表
            print(f"\n📈 调仓方案")
            print("-" * 80)

            if target_weights:
                # 打印表头
                print(
                    f"{'ETF代码':<15} {'名称':<12} {'目标权重':<12} {'分类':<10} {'信号':<10}"
                )
                print("-" * 80)

                # 按权重排序打印
                for etf, weight in sorted(
                    target_weights.items(), key=lambda x: x[1], reverse=True
                ):
                    name = ETF_NAME_MAP.get(etf, etf)
                    signal = (
                        factor_df.loc[etf, "signal"] if etf in factor_df.index else 0
                    )

                    # 确定分类
                    cat = "其他"
                    if etf in BOND_ETFS:
                        cat = "债券"
                    elif etf in A_SHARE_ETFS:
                        cat = "A股"
                    elif etf in INDUSTRY_ETFS:
                        cat = "周期"
                    elif etf == "518880.XSHG":
                        cat = "商品"
                    else:
                        cat = "境外"

                    print(
                        f"{etf:<15} {name:<12} {weight * 100:>8.2f}%   {cat:<10} {signal:<10.4f}"
                    )

                # 3. 资产配置分类
                print(f"\n💼 资产配置")
                print("-" * 80)
                if category_weights:
                    for cat, w in sorted(
                        category_weights.items(), key=lambda x: x[1], reverse=True
                    ):
                        bar = "█" * int(w * 20)  # 20个字符宽度
                        print(f"  {cat:<8}: {w * 100:>6.2f}% {bar}")

            print(f"{'=' * 80}")

    return pd.DataFrame(summary)


# 如果直接运行此文件，显示使用说明
if __name__ == "__main__":
    print("""
======================================================================
ETF-EPO策略研究环境版本（优化版）
======================================================================

使用方法:
---------
1. 在聚宽研究环境导入：
   from jqdata import *
   exec(open('etf_epo_v2_research.py').read())

2. 计算指定日期交易标的：
   result = calculate_for_date('2025-01-15', verbose=True)
   print(result['target_weights'])
   
   # 返回结果包含：
   # - target_weights: 目标权重
   # - factor_df: 因子数据
   # - selected_etfs: 选中ETF列表
   # - shrinkage: Ledoit-Wolf收缩强度
   # - category_weights: 分类权重统计

3. 批量获取所有周三的标的：
   df = get_next_wednesday_trades('2024-01-01', '2025-01-15')
   print(df)

4. 生成交易汇总（带详细表格）：
   results = []
   for date in pd.date_range('2024-01-01', '2025-01-15', freq='W-WED'):
       r = calculate_for_date(date.strftime('%Y-%m-%d'), verbose=False)
       if r['success']:
           results.append(r)
   
   # 方式A: 打印详细表格（默认）
   summary = generate_trade_summary(results, show_details=True)
   
   # 方式B: 只返回DataFrame，不打印
   summary = generate_trade_summary(results, show_details=False)
   print(summary)

5. 计算具体调仓方案（输入持仓和资金）：
   # 第一步：计算目标权重
   result = calculate_for_date('2025-01-15', verbose=False)
   target_weights = result['target_weights']
   
   # 第二步：准备当前持仓和价格
   current_positions = {
       '518880.XSHG': 500,   # 当前持有500股黄金ETF
       '159980.XSHE': 300,   # 当前持有300股有色ETF
   }
   total_value = 100000     # 总资产10万元
   
   # 获取当前价格（实际交易中从行情获取）
   current_prices = {
       '518880.XSHG': 4.5,
       '159980.XSHE': 3.2,
       '513100.XSHG': 1.8,
       # ... 其他ETF价格
   }
   
   # 第三步：计算调仓方案
   plan = calculate_rebalance_plan(
       current_positions=current_positions,
       total_value=total_value,
       target_weights=target_weights,
       current_prices=current_prices,
       min_lot=100,           # 最小交易100股
       verbose=True
   )
   
   # 输出结果：
   # - plan['orders']: 订单列表
   # - plan['target_shares']: 目标股数
   # - plan['changes']: 变更类型
   # - plan['turnover']: 预估成交额
   # - plan['new_positions']: 调仓后持仓
   
   # 输出示例：
   # ================================================================================
   # 【2025-01-15】交易详情
   # ================================================================================
   # 
   # 📊 因子信号排名 (Ledoit-Wolf收缩强度: 0.2847)
   # --------------------------------------------------------------------------------
   # 排名  ETF代码          名称         综合信号    动量分    质量分    趋势  
   # --------------------------------------------------------------------------------
   # 1    518880.XSHG     黄金ETF      0.7500    0.8250   0.7320   ✓    
   # 2    159915.XSHE     创业板ETF    0.6800    0.6120   0.7150   ✓    
   # ...
   # 
   # 📈 调仓方案
   # --------------------------------------------------------------------------------
   # ETF代码          名称         目标权重      分类       信号      
   # --------------------------------------------------------------------------------
   # 518880.XSHG     黄金ETF       25.00%     商品       0.7500    
   # 159915.XSHE     创业板ETF     20.00%     A股        0.6800    
   # ...
   # 
   # 💼 资产配置
   # --------------------------------------------------------------------------------
   #   商品    :  25.00% █████
   #   A股     :  20.00% ████
   #   周期    :  30.00% ██████
   #   债券    :  15.00% ███
   #   境外    :  10.00% ██
   # ================================================================================

当前ETF池（7只，已剔除沪深300）:
----------------------------------------------------------------------
- 518880.XSHG: 黄金ETF     (商品)
- 159915.XSHE: 创业板ETF   (A股成长)
- 513100.XSHG: 纳指ETF     (美股)
- 513980.XSHG: 恒指ETF     (港股)
- 159980.XSHE: 有色ETF大成 (周期)
- 561360.XSHG: 石油ETF     (周期)
- 511260.XSHG: 十年国债ETF (债券)

债券开关逻辑:
----------------------------------------------------------------------
- use_bond_floor = True:  启用保底机制 (最低15%仓位)
- use_bond_floor = False: 完全排除债券 (纯权益配置)

日志输出优化：
----------------------------------------------------------------------
- 详细的因子信号排名表
- 候选池筛选过程
- 资产配置分类统计
- 持仓明细表（含权重、分类、信号）
- 关键参数显示（收缩强度、权重限制等）

======================================================================
""")
