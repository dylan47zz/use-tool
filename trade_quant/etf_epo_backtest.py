# 克隆自聚宽文章：https://www.joinquant.com/post/66279
# 标题：年化31%+、最大回撤12.6%-复现一个ETF-EPO策略
# 作者：estivation
#
# v2版本：使用Ledoit-Wolf统计最优收缩估计替代经验收缩
# 升级内容：
#   1. 将协方差估计从"经验shrinkage"升级为"统计最优shrinkage"
#   2. ETF池优化：豆粕->石油ETF，新增十年国债
#   3. 配置逻辑：A股只有创业板（剔除沪深300后表现更好），行业竞争选1只（有色vs石油）
#   4. 债券保底：国债最低15%仓位，提供防御性配置

# 动量 + 质量因子 -> EPO 权重（Ledoit-Wolf收缩 + GARCH 锚定）
# -> 成交拥挤度约束 + 滑动窗口指标 + 债券保底机制
# -> 增强日志输出

from jqdata import *
import numpy as np
import pandas as pd
import math
from prettytable import PrettyTable
import prettytable

# v2: 引入Ledoit-Wolf统计最优收缩估计
try:
    from sklearn.covariance import LedoitWolf

    _LEDOIT_WOLF_AVAILABLE = True
except Exception:
    LedoitWolf = None
    _LEDOIT_WOLF_AVAILABLE = False

try:
    from arch import arch_model

    _ARCH_AVAILABLE = True
except Exception:
    arch_model = None
    _ARCH_AVAILABLE = False


# ============ 工具函数 ============


def _format_etf_code(etf_code):
    """美化ETF代码显示"""
    name = ETF_NAME_MAP.get(etf_code, "ETF")
    code = etf_code.split(".")[0]
    return f"{code}({name})"


def _get_etf_name(etf_code):
    """获取ETF名称"""
    return ETF_NAME_MAP.get(etf_code, etf_code)


def _get_etf_category(etf_code):
    """获取ETF分类"""
    return ETF_CATEGORY_MAP.get(etf_code, "其他")


def _get_trend_emoji(trend_ok):
    """获取趋势状态表情"""
    return "✅ 上涨趋势" if trend_ok else "⚠️ 趋势不佳"


def _get_signal_bar(signal, max_signal=1.0):
    """生成信号强度进度条"""
    if pd.isna(signal) or signal <= 0:
        return "░░░░░░░░░░"
    ratio = min(signal / max_signal, 1.0) if max_signal > 0 else 0
    filled = int(ratio * 10)
    return "█" * filled + "░" * (10 - filled)


def _get_pnl_emoji(pnl_ratio):
    """根据盈亏比例返回表情"""
    if pnl_ratio > 0.1:
        return "🤑 大涨"
    elif pnl_ratio > 0.05:
        return "😄 上涨"
    elif pnl_ratio > 0:
        return "📈 小涨"
    elif pnl_ratio > -0.05:
        return "📉 小跌"
    elif pnl_ratio > -0.1:
        return "😟 大跌"
    else:
        return "🤬 暴跌"


def _print_header(title, width=70):
    """打印分隔标题"""
    log.info("")
    log.info(f"{'=' * width}")
    log.info(f"  {title}")
    log.info(f"{'=' * width}")


def _print_subheader(subtitle):
    """打印副标题"""
    log.info(f"─── {subtitle} ───")


def _print_factor_table(factor_df, top_n=7):
    """使用PrettyTable打印因子得分排名"""
    if factor_df is None or factor_df.empty:
        return

    _print_header("📊 因子信号排名")

    table = PrettyTable(
        [
            "排名",
            "ETF代码",
            "ETF名称",
            "综合信号",
            "动量分",
            "质量分",
            "放量比",
            "趋势",
        ]
    )
    table.hrules = prettytable.ALL
    table.align = "r"
    table.align = "l"

    sorted_df = factor_df.sort_values("signal", ascending=False).head(top_n)

    for rank, (etf, row) in enumerate(sorted_df.iterrows(), 1):
        momentum_score = row.get("momentum_score", row.get("momentum_z", 0))
        quality_score = row.get("quality_score", 0)
        signal = row.get("signal", 0)
        volume_ratio = row.get("volume_ratio", 1.0)
        trend_ok = row.get("trend_ok", True)

        trend_emoji = "✅" if trend_ok else "⚠️"

        table.add_row(
            [
                f"#{rank}",
                _format_etf_code(etf),
                _get_etf_name(etf),
                f"{signal:.4f}",
                f"{momentum_score:.3f}",
                f"{quality_score:.3f}",
                f"{volume_ratio:.2f}x",
                trend_emoji,
            ]
        )

    log.info(f"\n{table}\n")


def _print_weight_table(target_weights, factor_df, total_value):
    """使用PrettyTable打印目标权重"""
    if not target_weights or factor_df is None:
        return

    _print_header("🎯 调仓目标")

    table = PrettyTable(
        [
            "ETF代码",
            "ETF名称",
            "分类",
            "目标权重",
            "预估金额",
            "信号得分",
            "动量分",
            "质量分",
            "放量比",
            "趋势",
        ]
    )
    table.hrules = prettytable.ALL
    table.align = "l"
    table.align = "r"

    sorted_weights = dict(
        sorted(target_weights.items(), key=lambda x: x[1], reverse=True)
    )

    for etf, weight in sorted_weights.items():
        if etf not in factor_df.index:
            continue

        row = factor_df.loc[etf]
        est_amount = total_value * weight
        momentum_score = row.get("momentum_score", row.get("momentum_z", 0))
        quality_score = row.get("quality_score", 0)
        signal = row.get("signal", 0)
        volume_ratio = row.get("volume_ratio", 1.0)
        trend_ok = row.get("trend_ok", True)
        trend_emoji = "✅" if trend_ok else "⚠️"

        table.add_row(
            [
                _format_etf_code(etf),
                _get_etf_name(etf),
                _get_etf_category(etf),
                f"{weight * 100:.1f}%",
                f"¥{est_amount:,.0f}",
                f"{signal:.4f}",
                f"{momentum_score:.3f}",
                f"{quality_score:.3f}",
                f"{volume_ratio:.2f}x",
                trend_emoji,
            ]
        )

    log.info(f"\n📈 总资产: ¥{total_value:,.2f}")
    log.info(f"\n{table}\n")

    weight_summary = ", ".join(
        [f"{_get_etf_name(k)}: {v * 100:.1f}%" for k, v in sorted_weights.items()]
    )
    log.info(f"📋 权重分配: {weight_summary}")


def _print_metrics_summary(metrics_df):
    """打印关键指标汇总"""
    if metrics_df is None or metrics_df.empty:
        return

    _print_header("📈 策略指标监控")

    top_etf = metrics_df["signal"].idxmax()
    best_signal = metrics_df.loc[top_etf, "signal"]
    best_momentum = metrics_df.loc[top_etf, "momentum"]
    best_quality = metrics_df.loc[top_etf, "quality_score"]

    worst_etf = metrics_df["signal"].idxmin()
    worst_signal = metrics_df.loc[worst_etf, "signal"]

    avg_signal = metrics_df["signal"].mean()
    avg_momentum = metrics_df["momentum"].mean()
    avg_quality = metrics_df["quality_score"].mean()

    log.info(f"🏆 最佳信号: {_get_etf_name(top_etf)} = {best_signal:.4f}")
    log.info(f"   动量得分: {best_momentum:.4f} | 质量得分: {best_quality:.4f}")
    log.info(f"")
    log.info(f"📉 最差信号: {_get_etf_name(worst_etf)} = {worst_signal:.4f}")
    log.info(f"")
    log.info(f"📊 平均信号: {avg_signal:.4f}")
    log.info(f"   平均动量: {avg_momentum:.4f} | 平均质量: {avg_quality:.4f}")


def _print_daily_summary(context, positions):
    """每日收盘后打印持仓汇总"""
    _print_header(f"📅 每日收盘汇总 - {context.current_dt.strftime('%Y-%m-%d')}")

    total_value = context.portfolio.total_value

    if not positions:
        log.info(f"🚤 当前总资产: ¥{total_value:,.2f}  (空仓)")
        return

    table = PrettyTable(
        [
            "ETF代码",
            "ETF名称",
            "持仓数量",
            "持仓成本",
            "当前价格",
            "盈亏比例",
            "盈亏金额",
            "市值",
            "仓位占比",
        ]
    )
    table.hrules = prettytable.ALL
    table.align = "l"
    table.align = "r"

    total_market_value = 0
    for etf, pos in positions.items():
        current_price = pos.price
        avg_cost = pos.avg_cost
        shares = pos.total_amount
        market_value = shares * current_price
        total_market_value += market_value

        pnl_ratio = (current_price - avg_cost) / avg_cost if avg_cost > 0 else 0
        pnl_amount = (current_price - avg_cost) * shares
        weight = market_value / total_value

        pnl_emoji = _get_pnl_emoji(pnl_ratio)
        pnl_str = f"{pnl_emoji} {pnl_ratio * 100:+.2f}%"

        table.add_row(
            [
                _format_etf_code(etf),
                _get_etf_name(etf),
                f"{shares:,}",
                f"¥{avg_cost:.3f}",
                f"¥{current_price:.3f}",
                pnl_str,
                f"¥{pnl_amount:+,.0f}",
                f"¥{market_value:,.0f}",
                f"{weight * 100:.1f}%",
            ]
        )

    log.info(f"\n💰 总资产: ¥{total_value:,.2f}")
    log.info(f"📊 持仓市值: ¥{total_market_value:,.2f}")
    log.info(f"\n{table}\n")

    # v2: 计算并记录累计收益（当前持仓的未实现盈亏）
    current_pnl = {}  # 当前持仓盈亏
    for etf, pos in positions.items():
        current_price = pos.price
        avg_cost = pos.avg_cost
        shares = pos.total_amount
        pnl_amount = (current_price - avg_cost) * shares
        current_pnl[etf] = pnl_amount

    # 计算总已实现盈亏累计收益 = 历史 + 当前持仓盈亏
    # g.cumulative_pnl 存储的是历史已实现盈亏
    total_pnl = sum(list(g.cumulative_pnl.values())) + sum(list(current_pnl.values()))

    # 打印累计收益表格
    all_pnl = {**{k: v for k, v in g.cumulative_pnl.items()}, **current_pnl}
    if all_pnl:
        _print_cumulative_pnl(positions, all_pnl)

    # 使用record记录每个ETF的累计收益（用于收益曲线）
    for etf in g.etf_pool:
        # 累计收益 = 历史已实现 + 当前持仓
        realized = g.cumulative_pnl.get(etf, 0)
        unrealized = current_pnl.get(etf, 0)
        pnl = realized + unrealized
        # 将ETF代码转换为可用的record名称
        record_name = "etf_" + etf.replace('.', '_').replace('-', '_')
        # 使用关键字参数
        record(**{record_name: pnl})
    # 记录总累计收益
    record(total_pnl=total_pnl)

    category_weights = {}
    for etf in positions.keys():
        category = _get_etf_category(etf)
        weight = positions[etf].total_amount * positions[etf].price / total_value
        category_weights[category] = category_weights.get(category, 0) + weight

    cat_str = ", ".join(
        [
            f"{k}: {v * 100:.1f}%"
            for k, v in sorted(
                category_weights.items(), key=lambda x: x[1], reverse=True
            )
        ]
    )
    log.info(f"📈 分类配置: {cat_str}")


def _print_cumulative_pnl(positions, cumulative_pnl):
    """打印累计收益表格"""
    if not positions:
        return

    table = PrettyTable(
        [
            "ETF代码",
            "ETF名称",
            "累计收益",
            "当前持仓盈亏",
            "占比",
        ]
    )
    table.hrules = prettytable.ALL
    table.align = "l"
    table.align = "r"

    total_pnl = sum([v for k, v in cumulative_pnl.items() if not k.endswith('_last')])

    for etf in positions.keys():
        cum_pnl = cumulative_pnl.get(etf, 0)
        current_pnl = cumulative_pnl.get(etf + '_last', 0)
        pnl_ratio = cum_pnl / total_pnl if total_pnl != 0 else 0

        pnl_emoji = "📈" if cum_pnl > 0 else "📉" if cum_pnl < 0 else "➡️"

        table.add_row(
            [
                _format_etf_code(etf),
                _get_etf_name(etf),
                f"{pnl_emoji} ¥{cum_pnl:+,.0f}",
                f"¥{current_pnl:+,.0f}",
                f"{pnl_ratio * 100:.1f}%" if total_pnl != 0 else "N/A",
            ]
        )

    log.info(f"\n📊 累计收益:\n{table}\n")
    log.info(f"💎 总累计收益: ¥{total_pnl:+,.0f}")


def _print_rebalance_summary(changes):
    """打印调仓变更汇总"""
    if not changes:
        log.info("\n📋 本次调仓: 无变更")
        return

    buys = [etf for etf, change in changes.items() if change == "BUY"]
    sells = [etf for etf, change in changes.items() if change == "SELL"]
    holds = [etf for etf, change in changes.items() if change == "HOLD"]

    _print_header("🔄 调仓执行摘要")

    if buys:
        log.info(f"\n🟢 买入 ({len(buys)}只):")
        for etf in buys:
            log.info(f"   + {_format_etf_code(etf)} ({_get_etf_name(etf)})")

    if sells:
        log.info(f"\n🔴 卖出 ({len(sells)}只):")
        for etf in sells:
            log.info(f"   - {_format_etf_code(etf)} ({_get_etf_name(etf)})")

    if holds:
        log.info(f"\n🟡 持有 ({len(holds)}只):")
        hold_infos = [f"{_get_etf_name(e)}" for e in holds]
        hold_str = ", ".join(hold_infos)
        log.info(f"   = {hold_str}")


# ============ ETF名称映射 ============

ETF_NAME_MAP = {
    "518880.XSHG": "黄金ETF",
    "159915.XSHE": "创业板ETF",
    "513100.XSHG": "纳指ETF",
    "513980.XSHG": "恒指ETF",
    "159980.XSHE": "有色ETF大成",
    "561360.XSHG": "石油ETF",  # 新增：替换豆粕
    "511260.XSHG": "十年国债ETF",  # 新增：债券保底
    # v2: 新增ETF
    "561550.XSHG": "中证500ETF",
    "159259.XSHE": "成长ETF",
    "159263.XSHE": "价值ETF",
}

ETF_CATEGORY_MAP = {
    "518880.XSHG": "商品",
    "159915.XSHE": "A股成长",
    "513100.XSHG": "美股",
    "513980.XSHG": "港股",
    "159980.XSHE": "周期",
    "561360.XSHG": "周期",  # 新增：石油（也是周期）
    "511260.XSHG": "债券",  # 新增：国债
    # v2: 新增ETF
    "561550.XSHG": "A股成长",  # 中证500：中盘成长
    "159259.XSHE": "A股成长",  # 成长ETF
    "159263.XSHE": "A股价值",  # 价值ETF
}


# ============ 策略初始化 ============


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(3 / 10000))
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=2.5 / 10000,
            close_commission=2.5 / 10000,
            min_commission=0.2,
        ),
        type="fund",
    )
    log.set_level("system", "error")
    log.set_level("order", "error")

    # ETF池：10只ETF，覆盖商品、A股、美股、港股、周期、债券
    g.etf_pool = [
        "518880.XSHG",  # 黄金（商品）
        "159915.XSHE",  # 创业板（A股成长）
        "513100.XSHG",  # 纳指（美股）
        "513980.XSHG",  # 恒指（港股）
        "159980.XSHE",  # 有色（周期）
        "561360.XSHG",  # 石油（周期）
        "511260.XSHG",  # 十年国债（债券）
        # v2: 新增ETF
        "561550.XSHG",  # 中证500（中盘）
        "159259.XSHE",  # 成长ETF
        "159263.XSHE",  # 价值ETF
    ]

    # 窗口参数
    g.momentum_window = 25
    g.momentum_lookback = 25
    g.quality_window = 25
    g.quality_lookback = 25
    g.cov_window = 60
    g.garch_window = 120
    g.volume_short_window = 5
    g.volume_long_window = 20

    # 因子混合参数
    g.score_weight_momentum = 0.3
    g.score_weight_quality = 0.7
    # v2: 信号权重模式（方案2/方案3）
    # 'fixed': 固定权重（使用上面的momentum/quality参数）
    # 'trend': 趋势调整（牛市动量高，熊市质量高）
    # 'strength': 信号强度调整（强信号动量高，弱信号质量高）
    g.signal_weight_mode = 'fixed'
    g.anchor_weight = 0.1
    g.max_holdings = 6
    g.use_rank_scoring = True
    g.quality_floor = 0.4
    g.momentum_floor = 0.0
    g.signal_power = 1.4
    g.use_window_mean = True
    g.min_holdings = 1
    g.min_holdings_risk = 2
    g.min_holdings_dd_threshold = 0.05
    g.use_dynamic_min_holdings = True
    g.max_weight = 0.85
    g.max_weight_high = 0.85
    g.max_weight_low = 0.75
    g.max_weight_dd_threshold = 0.05
    g.use_dynamic_max_weight = True
    g.use_risk_parity = False
    g.use_trend_filter = False
    g.trend_window = 20
    g.max_dd_filter = 0.08
    g.trend_penalty = 0.6
    g.trend_hard_filter = False
    g.min_lot = 100
    g.fallback_etf = "518880.XSHG"
    g.max_a_share_holdings = 1  # A股只有创业板（剔除沪深300后回测表现更好）
    g.a_share_weight_cap = 0.5  # A股总仓位上限50%
    g.a_share_weight_cap_high = 0.5
    g.a_share_weight_cap_low = 0.35
    g.a_share_dd_threshold = 0.05
    g.use_dynamic_a_share_cap = True
    g.industry_penalty = 0.85
    g.industry_weight_cap_high = 0.35
    g.industry_weight_cap_low = 0.25
    g.industry_dd_threshold = 0.05
    g.use_dynamic_industry_cap = True
    g.max_industry_holdings = 1  # 行业竞争选1只（有色 vs 石油）
    g.industry_weight_cap = 0.35
    g.avoid_industry = False
    g.premium_filter_enabled = True
    g.premium_threshold = 5.0
    g.premium_penalty = 0.5
    g.premium_hard_filter = False
    g.premium_etfs = {"513100.XSHG"}
    # A股ETF：创业板、中证500、成长、价值
    g.a_share_etfs = {
        "159915.XSHE",  # 创业板（成长）
        "561550.XSHG",  # 中证500（中盘）
        "159259.XSHE",  # 成长ETF
        "159263.XSHE",  # 价值ETF
    }
    # 行业ETF：有色和石油都是周期，竞争选1只
    g.industry_etfs = {
        "159980.XSHE",  # 有色
        "561360.XSHG",  # 石油
    }
    # 债券ETF配置：保底机制
    g.bond_etfs = {"511260.XSHG"}  # 十年国债
    g.use_bond_floor = False  # 债券保底开关（True=启用保底机制，False=纯动量竞争）
    g.min_bond_weight = 0.15  # 债券最低15%仓位（保底，仅在use_bond_floor=True时生效）
    g.max_bond_weight = 0.40  # 债券最高40%仓位（防止过度保守）

    # EPO参数 - v2: 使用Ledoit-Wolf统计最优收缩
    # g.epo_risk_aversion = 8.0  # v2: 已废弃，归一化后该参数无效
    # v2: 移除了 g.epo_shrinkage，改为自动计算
    g.use_ledoit_wolf = True  # v2: 新增开关，启用Ledoit-Wolf收缩估计
    # v2: 可选，设置收缩强度的上下限（Ledoit-Wolf自动计算的结果会被限制在此范围内）
    g.shrinkage_floor = 0.05  # 最小收缩强度（0表示完全信任样本协方差）
    g.shrinkage_cap = 0.3  # 最大收缩强度（1表示完全使用目标矩阵）
    # v2: 目标波动率版本 - 控制仓位而不是永远满仓
    g.target_volatility = 0.15  # 目标年化波动率（12%），0表示不使用波动率缩放（满仓）

    # 成交拥挤度惩罚
    g.volume_ratio_threshold = 1.6
    g.volume_penalty_power = 0.8
    g.use_relative_crowding = True
    g.relative_crowding_power = 1.0
    g.relative_crowding_floor = 0.6
    g.relative_crowding_ceiling = 1.6
    g.dd_penalty_threshold = 0.05
    g.dd_penalty_power = 1.0
    g.dd_penalty_floor = 0.6

    # 每周调仓（周三）
    g.rebalance_weekday = 3
    g.factor_time = "09:20"
    g.rebalance_time = "11:15"

    g.price_cache = {}
    g.factor_df = None
    # v2: 记录Ledoit-Wolf收缩强度用于监控
    g.last_shrinkage = None
    # v2: 累计收益记录
    g.cumulative_pnl = {}  # {etf: cumulative_pnl_amount}
    g.total_pnl = 0  # 总累计收益

    run_weekly(calc_factors, weekday=g.rebalance_weekday, time=g.factor_time)
    run_weekly(rebalance, weekday=g.rebalance_weekday, time=g.rebalance_time)

    # 每日打印持仓汇总
    run_daily(daily_summary, "15:00")


# ============ 每日收盘汇总 ============


def daily_summary(context):
    """每日收盘后打印持仓汇总"""
    positions = context.portfolio.positions
    _print_daily_summary(context, positions)


# ============ 因子计算 ============


def calc_factors(context):
    g.price_cache = {}
    g.factor_df = None

    prev_day = _previous_trade_day(context)
    if prev_day is None:
        log.warn("no previous trade day")
        return

    need_days = (
        max(
            g.momentum_lookback,
            g.quality_lookback,
            g.cov_window,
            g.volume_long_window,
            g.garch_window,
        )
        + 5
    )
    min_len = max(g.momentum_window, g.quality_window, g.volume_long_window) + 1

    for etf in g.etf_pool:
        data = get_price(
            etf,
            count=need_days,
            end_date=prev_day,
            frequency="daily",
            fields=["close", "volume", "money"],
        )
        if data is None or data.empty:
            continue
        data = data.dropna()
        if len(data) < min_len:
            continue
        g.price_cache[etf] = data

    if not g.price_cache:
        log.warn("no price data in cache")
        return

    metrics = {}
    for etf, data in g.price_cache.items():
        close = data["close"].values
        if "money" in data:
            turnover = data["money"].fillna(0).values
        else:
            turnover = data["volume"].fillna(0).values
        metrics[etf] = _compute_metrics(close, turnover)

    metrics_df = pd.DataFrame(metrics).T
    if metrics_df.empty:
        log.warn("empty metrics")
        return

    if g.use_rank_scoring:
        metrics_df["momentum_score"] = _rank_score(metrics_df["momentum"], True)
        metrics_df["sharpe_score"] = _rank_score(metrics_df["sharpe"], True)
        metrics_df["mdd_score"] = _rank_score(metrics_df["max_drawdown"], False)
        metrics_df["vol_score"] = _rank_score(metrics_df["volatility"], False)
        metrics_df["vol_stability_score"] = _rank_score(
            metrics_df["vol_stability"], False
        )
        metrics_df["volume_stability_score"] = _rank_score(
            metrics_df["volume_stability"], False
        )
        metrics_df["logret_score"] = _rank_score(metrics_df["log_return"], True)
        metrics_df["r2_score"] = _rank_score(metrics_df["r2"], True)

        metrics_df["quality_score"] = metrics_df[
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

        # 方案B：债券只使用动量分，不参与质量分计算
        bond_etfs = getattr(g, "bond_etfs", set())
        metrics_df["signal"] = (
            g.score_weight_momentum * metrics_df["momentum_score"]
            + g.score_weight_quality * metrics_df["quality_score"]
        )
        # 债券：只使用动量分（忽略质量分）
        bond_mask = metrics_df.index.isin(bond_etfs)
        metrics_df.loc[bond_mask, "signal"] = metrics_df.loc[
            bond_mask, "momentum_score"
        ]

        if g.industry_penalty < 1:
            metrics_df.loc[metrics_df.index.isin(g.industry_etfs), "signal"] *= (
                g.industry_penalty
            )
        if g.trend_penalty < 1:
            metrics_df.loc[~metrics_df["trend_ok"], "signal"] *= g.trend_penalty
    else:
        metrics_df["momentum_z"] = _zscore(metrics_df["momentum"])
        metrics_df["sharpe_z"] = _zscore(metrics_df["sharpe"])
        metrics_df["mdd_z"] = _zscore(-metrics_df["max_drawdown"])
        metrics_df["vol_z"] = _zscore(-metrics_df["volatility"])
        metrics_df["vol_stability_z"] = _zscore(-metrics_df["vol_stability"])
        metrics_df["volume_stability_z"] = _zscore(-metrics_df["volume_stability"])
        metrics_df["logret_z"] = _zscore(metrics_df["log_return"])
        metrics_df["r2_z"] = _zscore(metrics_df["r2"])
        metrics_df["quality_score"] = metrics_df[
            [
                "sharpe_z",
                "mdd_z",
                "vol_z",
                "vol_stability_z",
                "volume_stability_z",
                "logret_z",
                "r2_z",
            ]
        ].mean(axis=1)

        # 方案B：债券只使用动量分，不参与质量分计算
        bond_etfs = getattr(g, "bond_etfs", set())
        metrics_df["signal"] = (
            g.score_weight_momentum * metrics_df["momentum_z"]
            + g.score_weight_quality * metrics_df["quality_score"]
        )
        # 债券：只使用动量分（忽略质量分）
        bond_mask = metrics_df.index.isin(bond_etfs)
        metrics_df.loc[bond_mask, "signal"] = metrics_df.loc[bond_mask, "momentum_z"]

        if g.industry_penalty < 1:
            metrics_df.loc[metrics_df.index.isin(g.industry_etfs), "signal"] *= (
                g.industry_penalty
            )
        if g.trend_penalty < 1:
            metrics_df.loc[~metrics_df["trend_ok"], "signal"] *= g.trend_penalty

    metrics_df["premium"] = 0.0
    metrics_df["premium_ok"] = True
    if g.premium_filter_enabled:
        for etf in g.premium_etfs:
            if etf not in metrics_df.index:
                continue
            close = g.price_cache.get(etf)
            if close is None or close.empty:
                continue
            nav = _get_unit_nav(etf, prev_day)
            if nav is None or nav <= 0:
                continue
            premium = (close["close"].values[-1] - nav) / nav * 100
            metrics_df.at[etf, "premium"] = premium
            metrics_df.at[etf, "premium_ok"] = premium <= g.premium_threshold
        if g.premium_penalty < 1:
            metrics_df.loc[~metrics_df["premium_ok"], "signal"] *= g.premium_penalty

    g.factor_df = metrics_df

    # 打印因子计算结果
    _print_header(f"📊 因子计算完成 - {context.current_dt.strftime('%Y-%m-%d')}")
    _print_factor_table(metrics_df, top_n=len(metrics_df))
    _print_metrics_summary(metrics_df)


# ============ 调仓逻辑 ============


def rebalance(context):
    if g.factor_df is None or g.factor_df.empty:
        log.warn("no factor data")
        return

    factor_df = g.factor_df

    # 【修复】在信号筛选前就排除国债，确保不使用债券
    if not getattr(g, "use_bond_floor", False):
        bond_etfs = getattr(g, "bond_etfs", set())
        if bond_etfs:
            factor_df = factor_df[~factor_df.index.isin(bond_etfs)]
            log.info(f"【债券排除】已排除债券ETF: {bond_etfs}")

    if g.use_rank_scoring:
        mask = (factor_df["momentum"] > g.momentum_floor) & (
            factor_df["quality_score"] >= g.quality_floor
        )
        if g.use_trend_filter and getattr(g, "trend_hard_filter", True):
            mask &= factor_df["trend_ok"]
        if (
            g.premium_filter_enabled
            and getattr(g, "premium_hard_filter", True)
            and "premium_ok" in factor_df.columns
        ):
            mask &= factor_df["premium_ok"]
        signal_series = factor_df.loc[mask, "signal"]
    else:
        signal_series = factor_df["signal"]
        signal_series = signal_series[signal_series > 0]

    if signal_series.empty:
        _print_header("⚠️ 无有效信号 - 清仓")
        log.info("🚫 所有ETF信号为负，执行清仓")
        _execute_orders(context, {})
        return

    candidates = _select_candidates(signal_series, factor_df)
    if not candidates:
        _print_header("⚠️ 无候选标的 - 清仓")
        log.info("🚫 约束条件过滤后无候选ETF，执行清仓")
        _execute_orders(context, {})
        return
    returns_df = _build_returns_df(candidates)
    if returns_df.empty or returns_df.shape[1] == 0:
        log.warn("no returns for EPO")
        return

    # 信号处理 - v2: 支持动态信号权重
    # 根据模式获取动量/质量权重
    signal_weight_mode = getattr(g, 'signal_weight_mode', 'fixed')
    if signal_weight_mode != 'fixed':
        # 获取原始动量分数和质量分数
        momentum_scores = g.factor_df.loc[returns_df.columns, "momentum_score"].values
        quality_scores = g.factor_df.loc[returns_df.columns, "quality_score"].values

        # 获取动态权重
        momentum_w, quality_w = _get_signal_weights(
            signal_weight_mode, returns_df, None
        )

        # 重新计算信号
        raw_signals = momentum_w * momentum_scores + quality_w * quality_scores
    else:
        raw_signals = g.factor_df.loc[returns_df.columns, "signal"].values

    signals = raw_signals.copy()
    if g.anchor_weight > 0:
        anchor_signal = _build_anchor_signal(returns_df.columns)
        signals = (1 - g.anchor_weight) * signals + g.anchor_weight * anchor_signal

    if g.signal_power != 1.0:
        signals = np.power(signals, g.signal_power)

    # v2: 使用Ledoit-Wolf统计最优收缩估计
    weights = _epo_weights(returns_df, signals)

    # ====== 调试日志：EPO权重（未归一化）======
    log.info(f"【LW调试】EPO原始权重(raw): {dict(zip(list(returns_df.columns), weights))}")

    if g.use_risk_parity:
        weights = _apply_risk_parity(weights, returns_df)
        log.info(f"【LW调试】RiskParity后权重: {dict(zip(list(returns_df.columns), weights))}")

    # 成交拥挤度惩罚
    penalties = []
    for etf in returns_df.columns:
        ratio = g.factor_df.loc[etf, "volume_ratio"]
        penalty = 1.0
        if ratio > g.volume_ratio_threshold:
            penalty = ratio ** (-g.volume_penalty_power)
        penalties.append(penalty)

    penalties = np.array(penalties, dtype=float)
    if g.use_relative_crowding:
        ratios = g.factor_df.loc[returns_df.columns, "volume_ratio"]
        ratios = ratios.replace([np.inf, -np.inf], np.nan)
        median = np.nanmedian(ratios.values)
        if median and not np.isnan(median):
            rel = ratios.values / median
            rel = np.clip(rel, g.relative_crowding_floor, g.relative_crowding_ceiling)
            rel_penalty = np.power(rel, -g.relative_crowding_power)
            penalties = penalties * rel_penalty

    dd = g.factor_df.loc[returns_df.columns, "trend_dd"]
    dd = dd.replace([np.inf, -np.inf], np.nan)
    if g.dd_penalty_threshold and not dd.isna().all():
        ratio = dd.values / g.dd_penalty_threshold
        dd_penalty = np.ones_like(ratio, dtype=float)
        mask = ratio > 1
        dd_penalty[mask] = ratio[mask] ** (-g.dd_penalty_power)
        dd_penalty = np.clip(dd_penalty, g.dd_penalty_floor, 1.0)
        penalties = penalties * dd_penalty

    weights = _normalize(np.array(weights) * penalties)
    log.info(f"【LW调试】Penalty归一化后权重: {dict(zip(list(returns_df.columns), weights))}")

    a_share_cap = g.a_share_weight_cap
    if g.use_dynamic_a_share_cap:
        a_share_cap = _dynamic_a_share_cap(returns_df.columns, factor_df)

    if a_share_cap is not None:
        weights = _apply_group_weight_cap(
            weights, list(returns_df.columns), g.a_share_etfs, a_share_cap
        )

    industry_cap = g.industry_weight_cap
    if g.use_dynamic_industry_cap:
        industry_cap = _dynamic_industry_cap(returns_df.columns, factor_df)

    if industry_cap is not None:
        weights = _apply_group_weight_cap(
            weights, list(returns_df.columns), g.industry_etfs, industry_cap
        )

    max_weight = g.max_weight
    if g.use_dynamic_max_weight:
        max_weight = _dynamic_max_weight(list(returns_df.columns), factor_df)

    if max_weight:
        weights = _apply_weight_cap(weights, max_weight)
    log.info(f"【LW调试】MaxWeight限制后权重: {dict(zip(list(returns_df.columns), weights))}")
    target_weights = dict(zip(returns_df.columns, weights))

    # 债券保底配置：确保债券有最低仓位
    target_weights = _apply_bond_floor(target_weights, factor_df)
    log.info(f"【LW调试】BondFloor后最终权重: {target_weights}")

    # v2: 目标波动率版本 - 缩放仓位
    if g.target_volatility > 0:
        target_weights = _apply_volatility_scaling(target_weights, returns_df, g.target_volatility)
        log.info(f"【LW调试】波动率缩放后权重: {target_weights}")

    _print_header(f"🔄 调仓执行 - {context.current_dt.strftime('%Y-%m-%d')}")
    _print_weight_table(target_weights, factor_df, context.portfolio.total_value)

    # v2: 打印Ledoit-Wolf收缩强度信息
    if g.use_ledoit_wolf and g.last_shrinkage is not None:
        log.info(f"📊 Ledoit-Wolf收缩强度: {g.last_shrinkage:.4f}")

    # 计算调仓变更
    current_positions = context.portfolio.positions
    changes = {}
    for etf in target_weights:
        if etf in current_positions:
            changes[etf] = "HOLD"
    for etf in current_positions:
        if etf not in target_weights:
            changes[etf] = "SELL"
    for etf in target_weights:
        if etf not in current_positions:
            changes[etf] = "BUY"

    _print_rebalance_summary(changes)
    _execute_orders(context, target_weights)


# ============ 订单执行 ============


def _execute_orders(context, target_weights):
    current_data = get_current_data()
    total_value = context.portfolio.total_value

    target_weights = _adjust_weights_for_trading(
        target_weights, current_data, total_value
    )

    if not target_weights:
        return

    target_shares = {}
    for etf, weight in target_weights.items():
        price = current_data[etf].last_price
        if price is None or np.isnan(price) or price <= 0:
            continue
        target_value = total_value * weight
        shares = int(target_value / price // g.min_lot) * g.min_lot
        if shares < g.min_lot:
            shares = 0
        target_shares[etf] = shares

    # 卖出不在目标列表中的持仓
    for etf in list(context.portfolio.positions.keys()):
        if etf not in target_shares and not current_data[etf].paused:
            # v2: 记录已实现收益
            pos = context.portfolio.positions.get(etf)
            if pos:
                realized_pnl = (pos.price - pos.avg_cost) * pos.total_amount
                g.cumulative_pnl[etf] = g.cumulative_pnl.get(etf, 0) + realized_pnl
                log.info(f"📊 已实现收益: {_format_etf_code(etf)} = ¥{realized_pnl:+,.0f}")
            log.info(f"🔴 卖出 {_format_etf_code(etf)} ({_get_etf_name(etf)})")
            order_target(etf, 0)

    # 先减仓释放资金
    for etf, shares in target_shares.items():
        if current_data[etf].paused:
            continue
        current_pos = context.portfolio.positions.get(etf)
        current_shares = current_pos.total_amount if current_pos else 0
        if shares < current_shares:
            # v2: 记录已实现收益（减仓部分）
            sold_shares = current_shares - shares
            price = current_data[etf].last_price
            realized_pnl = (price - current_pos.avg_cost) * sold_shares
            g.cumulative_pnl[etf] = g.cumulative_pnl.get(etf, 0) + realized_pnl
            log.info(f"📊 已实现收益: {_format_etf_code(etf)} = ¥{realized_pnl:+,.0f} (减仓{sold_shares}股)")
            log.info(f"🔴 减仓 {_format_etf_code(etf)}: {current_shares} -> {shares}股")
            order_target(etf, shares)

    # 卖出后再加仓
    for etf, shares in target_shares.items():
        if current_data[etf].paused:
            continue
        current_pos = context.portfolio.positions.get(etf)
        current_shares = current_pos.total_amount if current_pos else 0
        if shares > current_shares:
            log.info(f"🟢 加仓 {_format_etf_code(etf)}: {current_shares} -> {shares}股")
            order_target(etf, shares)


# ============ 因子计算辅助函数 ============


def _build_returns_df(etfs):
    series_list = []
    for etf in etfs:
        data = g.price_cache.get(etf)
        if data is None or data.empty:
            continue
        rets = data["close"].pct_change().dropna().tail(g.cov_window)
        if rets.empty:
            continue
        series_list.append(rets.rename(etf))
    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1, join="inner").dropna()


# v2: 使用Ledoit-Wolf统计最优收缩估计
def _epo_weights(returns_df, signals):
    """
    计算EPO权重，使用Ledoit-Wolf统计最优收缩估计

    Args:
        returns_df: 收益矩阵 (T x n)
        signals: 信号向量 (n,)

    Returns:
        weights: 权重向量 (n,)
    """
    n = returns_df.shape[1]
    if returns_df.shape[0] < 2 or n == 0:
        return np.ones(n) / max(n, 1)

    assets = list(returns_df.columns)

    # v2: 使用Ledoit-Wolf统计最优收缩估计
    # 保存样本协方差用于对比调试
    sample_cov = returns_df.cov().values

    if g.use_ledoit_wolf and _LEDOIT_WOLF_AVAILABLE:
        try:
            # Ledoit-Wolf自动计算最优收缩强度
            lw = LedoitWolf(assume_centered=True)  # 收益序列均值≈0
            lw.fit(returns_df.values)

            # v2修复：获取LW计算的收缩系数，并应用范围限制
            raw_shrinkage = lw.shrinkage_
            limited_shrinkage = float(
                np.clip(raw_shrinkage, g.shrinkage_floor, g.shrinkage_cap)
            )
            g.last_shrinkage = limited_shrinkage

            # 手动应用收缩：shrunk = (1-s)*样本协方差 + s*目标矩阵（单位矩阵）
            # 这样才能让shrinkage_floor和shrinkage_cap真正生效
            target = np.eye(n)  # 目标矩阵
            shrunk = (1 - limited_shrinkage) * sample_cov + limited_shrinkage * target

        except Exception as e:
            log.warn(f"Ledoit-Wolf估计失败，回退到样本协方差: {e}")
            # 回退到样本协方差
            shrunk = sample_cov
            g.last_shrinkage = 0.0
    else:
        # 不使用Ledoit-Wolf时使用样本协方差
        shrunk = sample_cov
        g.last_shrinkage = 0.0

    # ====== 调试日志：对比协方差矩阵差异 ======
    # 不加正则化的对比（加正则化后差异会被稀释）
    cov_diff = shrunk - sample_cov
    cov_diff_norm = np.linalg.norm(cov_diff, 'fro')
    cov_diff_max = np.max(np.abs(cov_diff))
    log.info(f"【LW调试】收缩强度={g.last_shrinkage:.4f}, 协方差矩阵Frobenius范数差异={cov_diff_norm:.6f}, 最大元素差异={cov_diff_max:.6f}")

    # 样本协方差逆矩阵（用于对比）
    sample_cov_stable = sample_cov + np.eye(n) * 1e-6
    try:
        inv_sample_cov = np.linalg.inv(sample_cov_stable)
    except:
        inv_sample_cov = np.pinv(sample_cov_stable)

    # 确保正定性（数值稳定性）
    shrunk = shrunk + np.eye(n) * 1e-6

    try:
        inv_cov = np.linalg.inv(shrunk)
    except np.linalg.LinAlgError:
        inv_cov = np.pinv(shrunk)

    # ====== 调试日志：对比逆矩阵差异 ======
    inv_diff = inv_cov - inv_sample_cov
    inv_diff_norm = np.linalg.norm(inv_diff, 'fro')
    log.info(f"【LW调试】逆矩阵Frobenius范数差异={inv_diff_norm:.6f}")

    raw = inv_cov.dot(signals)

    # 仅做多约束
    raw = np.maximum(0, raw)

    # ====== 调试日志：对比raw权重差异 ======
    raw_sample = inv_sample_cov.dot(signals)
    raw_sample = np.maximum(0, raw_sample)

    raw_diff = raw - raw_sample
    raw_diff_norm = np.linalg.norm(raw_diff, 1)  # L1范数更能反映绝对差异
    raw_diff_max = np.max(np.abs(raw_diff))
    log.info(f"【LW调试】Raw权重L1范数差异={raw_diff_norm:.6f}, 最大元素差异={raw_diff_max:.6f}")
    # 打印各资产权重对比
    weight_diff = {}
    for i, asset in enumerate(assets):
        diff = raw[i] - raw_sample[i]
        if abs(diff) > 1e-6:  # 只打印有差异的
            weight_diff[asset] = diff
    if weight_diff:
        log.info(f"【LW调试】权重差异: {weight_diff}")

    # v2优化：返回未归一化的raw权重，让调用方统一归一化
    # 这样Ledoit-Wolf的协方差矩阵优化效果能更直接地体现在最终权重中
    return raw


def _compute_metrics(close, volume):
    quality_prices = close[-g.quality_lookback :]
    momentum_prices = close[-g.momentum_lookback :]
    quality_volume = volume[-g.quality_lookback :]

    sharpe = _rolling_metric_on_prices(
        quality_prices,
        g.quality_window,
        lambda p: _calc_sharpe(np.diff(p) / p[:-1]),
    )
    max_dd = _rolling_metric_on_prices(
        quality_prices, g.quality_window, _calc_max_drawdown
    )
    volatility = _rolling_metric_on_prices(
        quality_prices,
        g.quality_window,
        lambda p: _calc_volatility(np.diff(p) / p[:-1]),
    )
    vol_stability = _rolling_metric_on_prices(
        quality_prices,
        g.quality_window,
        lambda p: _calc_vol_stability(np.diff(p) / p[:-1]),
    )
    log_return = _rolling_metric_on_prices(
        quality_prices, g.quality_window, _calc_log_return
    )
    r2_q = _rolling_metric_on_prices(
        quality_prices, g.quality_window, _calc_r2_log_prices
    )
    volume_stability = _rolling_metric_on_volume(
        quality_volume, g.quality_window, _calc_volume_stability
    )

    momentum = _rolling_metric_on_prices(
        momentum_prices, g.momentum_window, _calc_momentum
    )

    volume_ratio = _calc_volume_ratio(volume)
    trend_ok, trend_dd = _calc_trend_filter(close)

    return {
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "volatility": volatility,
        "vol_stability": vol_stability,
        "volume_stability": volume_stability,
        "log_return": log_return,
        "r2": r2_q,
        "momentum": momentum,
        "volume_ratio": volume_ratio,
        "trend_ok": trend_ok,
        "trend_dd": trend_dd,
    }


def _calc_sharpe(returns):
    if len(returns) < 2:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(252)


def _calc_max_drawdown(prices):
    if len(prices) < 2:
        return 0.0
    running_max = np.maximum.accumulate(prices)
    drawdowns = prices / running_max - 1
    return abs(np.min(drawdowns))


def _calc_vol_stability(returns):
    if len(returns) < 2:
        return 0.0
    vol_window = max(5, int(len(returns) / 5))
    if len(returns) < vol_window + 1:
        return np.std(returns)
    rolling_vol = pd.Series(returns).rolling(vol_window).std().dropna()
    if rolling_vol.empty:
        return np.std(returns)
    return np.std(rolling_vol.values)


def _calc_volatility(returns):
    if len(returns) < 2:
        return 0.0
    return np.std(returns) * math.sqrt(252)


def _calc_volume_stability(volumes):
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


def _calc_r2(log_prices):
    if len(log_prices) < 2:
        return 0.0
    x = np.arange(len(log_prices))
    slope, intercept = np.polyfit(x, log_prices, 1)
    fitted = slope * x + intercept
    ss_res = np.sum((log_prices - fitted) ** 2)
    ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def _calc_annualized_return(log_prices):
    if len(log_prices) < 2:
        return 0.0
    slope = np.polyfit(np.arange(len(log_prices)), log_prices, 1)[0]
    return math.exp(slope * 252) - 1


def _calc_volume_ratio(volume):
    if len(volume) < g.volume_long_window:
        return 1.0
    short_avg = np.mean(volume[-g.volume_short_window :])
    long_avg = np.mean(volume[-g.volume_long_window :])
    if long_avg <= 0:
        return 1.0
    return short_avg / long_avg


def _calc_trend_filter(close):
    if len(close) < g.trend_window:
        return True, 0.0
    window = close[-g.trend_window :]
    ma = float(np.mean(window))
    dd = _calc_max_drawdown(window)
    if ma <= 0:
        return False, dd
    return close[-1] >= ma and dd <= g.max_dd_filter, dd


def _calc_log_return(prices):
    if len(prices) < 2 or prices[0] <= 0:
        return 0.0
    return math.log(prices[-1] / prices[0])


def _calc_r2_log_prices(prices):
    prices = np.array(prices, dtype=float)
    if len(prices) < 2 or np.any(prices <= 0):
        return 0.0
    return _calc_r2(np.log(prices))


def _calc_momentum(prices):
    prices = np.array(prices, dtype=float)
    if len(prices) < 2 or np.any(prices <= 0):
        return 0.0
    log_prices = np.log(prices)
    return _calc_annualized_return(log_prices) * _calc_r2(log_prices)


def _rolling_metric_on_prices(prices, window, func):
    prices = np.array(prices, dtype=float)
    if len(prices) < 2:
        return 0.0
    if len(prices) < window:
        return func(prices)
    if not g.use_window_mean:
        return func(prices[-window:])
    values = []
    for i in range(window, len(prices) + 1):
        values.append(func(prices[i - window : i]))
    return float(np.mean(values)) if values else func(prices)


def _rolling_metric_on_volume(volumes, window, func):
    volumes = np.array(volumes, dtype=float)
    if len(volumes) < 2:
        return 0.0
    if len(volumes) < window:
        return func(volumes)
    if not g.use_window_mean:
        return func(volumes[-window:])
    values = []
    for i in range(window, len(volumes) + 1):
        values.append(func(volumes[i - window : i]))
    return float(np.mean(values)) if values else func(volumes)


def _normalize(values):
    total = np.sum(values)
    if total <= 0:
        return np.ones(len(values)) / max(len(values), 1)
    return values / total


def _apply_risk_parity(weights, returns_df):
    vols = returns_df.std() * math.sqrt(252)
    vols = vols.replace([np.inf, -np.inf], np.nan)
    if vols.isna().all():
        return weights
    vols = vols.fillna(vols.median())
    inv_vol = 1.0 / (vols.values + 1e-10)
    scaled = np.array(weights) * inv_vol
    return _normalize(scaled)


def _apply_weight_cap(weights, cap):
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


def _apply_group_weight_cap(weights, assets, group_set, cap):
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


def _dynamic_a_share_cap(assets, factor_df):
    if not getattr(g, "use_dynamic_a_share_cap", False):
        return g.a_share_weight_cap
    if factor_df is None or factor_df.empty:
        return g.a_share_weight_cap
    a_assets = [a for a in assets if a in g.a_share_etfs]
    if not a_assets:
        return g.a_share_weight_cap
    dd = factor_df.loc[a_assets, "trend_dd"]
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    if dd.empty:
        return g.a_share_weight_cap
    dd_metric = float(np.nanpercentile(dd.values, 75))
    threshold = getattr(g, "a_share_dd_threshold", None)
    low = getattr(g, "a_share_weight_cap_low", g.a_share_weight_cap)
    high = getattr(g, "a_share_weight_cap_high", g.a_share_weight_cap)
    if threshold is None:
        return g.a_share_weight_cap
    if dd_metric >= threshold:
        return low
    return high


def _dynamic_industry_cap(assets, factor_df):
    if not getattr(g, "use_dynamic_industry_cap", False):
        return g.industry_weight_cap
    if factor_df is None or factor_df.empty:
        return g.industry_weight_cap
    i_assets = [a for a in assets if a in g.industry_etfs]
    if not i_assets:
        return g.industry_weight_cap
    dd = factor_df.loc[i_assets, "trend_dd"]
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    if dd.empty:
        return g.industry_weight_cap
    dd_metric = float(np.nanpercentile(dd.values, 75))
    threshold = getattr(g, "industry_dd_threshold", None)
    low = getattr(g, "industry_weight_cap_low", g.industry_weight_cap)
    high = getattr(g, "industry_weight_cap_high", g.industry_weight_cap)
    if threshold is None:
        return g.industry_weight_cap
    if dd_metric >= threshold:
        return low
    return high


def _dynamic_max_weight(assets, factor_df):
    if not getattr(g, "use_dynamic_max_weight", False):
        return g.max_weight
    if factor_df is None or factor_df.empty:
        return g.max_weight
    assets = list(assets)
    dd = factor_df.loc[assets, "trend_dd"]
    dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
    if dd.empty:
        return g.max_weight
    dd_metric = float(np.nanpercentile(dd.values, 75))
    threshold = getattr(g, "max_weight_dd_threshold", None)
    low = getattr(g, "max_weight_low", g.max_weight)
    high = getattr(g, "max_weight_high", g.max_weight)
    if threshold is None:
        return g.max_weight
    if dd_metric >= threshold:
        return low
    return high


def _select_candidates(signal_series, factor_df):
    if signal_series.empty:
        return []

    # 债券开关逻辑：如果未启用保底机制，完全排除债券ETF
    bond_etfs = getattr(g, "bond_etfs", set())
    use_bond = getattr(g, "use_bond_floor", True)
    if not use_bond and bond_etfs:
        # 过滤掉债券ETF，不参与任何竞争
        signal_series = signal_series[~signal_series.index.isin(bond_etfs)]

    min_holdings = g.min_holdings
    if g.use_dynamic_min_holdings and factor_df is not None and not factor_df.empty:
        dd = factor_df.loc[signal_series.index, "trend_dd"]
        dd = dd.replace([np.inf, -np.inf], np.nan).dropna()
        if not dd.empty:
            dd_metric = float(np.nanpercentile(dd.values, 75))
            if dd_metric >= g.min_holdings_dd_threshold:
                min_holdings = g.min_holdings_risk
    signal_series = signal_series.sort_values(ascending=False)
    selected = []
    a_share_count = 0
    industry_count = 0
    max_industry = getattr(g, "max_industry_holdings", None)

    for etf in signal_series.index:
        if g.avoid_industry and etf in g.industry_etfs:
            continue
        if max_industry is not None and etf in g.industry_etfs:
            if industry_count >= max_industry:
                continue
            industry_count += 1
        if etf in g.a_share_etfs:
            if a_share_count >= g.max_a_share_holdings:
                continue
            a_share_count += 1
        selected.append(etf)
        if g.max_holdings and len(selected) >= g.max_holdings:
            break

    if min_holdings and len(selected) < min_holdings:
        fallback = factor_df["signal"].sort_values(ascending=False)
        for etf in fallback.index:
            if etf in selected:
                continue
            if g.avoid_industry and etf in g.industry_etfs:
                continue
            if max_industry is not None and etf in g.industry_etfs:
                if industry_count >= max_industry:
                    continue
                industry_count += 1
            if etf in g.a_share_etfs and a_share_count >= g.max_a_share_holdings:
                continue
            if etf in g.a_share_etfs:
                a_share_count += 1
            selected.append(etf)
            if len(selected) >= min_holdings:
                break

    return selected


def _adjust_weights_for_trading(target_weights, current_data, total_value):
    """
    调整权重以适应交易约束（最小交易量、权重限制等）

    重要：保留原始仓位比例，支持波动率缩放后的非满仓场景
    """
    if not target_weights:
        return {}

    # v2: 保存原始仓位比例，用于保留波动率缩放效果
    original_total = float(sum(list(target_weights.values())))

    assets = []
    weights = []

    for etf, weight in target_weights.items():
        price = current_data[etf].last_price
        if price is None or np.isnan(price) or price <= 0:
            continue
        target_value = total_value * weight
        shares = int(target_value / price // g.min_lot) * g.min_lot
        min_value = g.min_lot * price
        if shares < g.min_lot:
            continue
        assets.append(etf)
        weights.append(weight)

    if not assets:
        return {}

    weights = _normalize(np.array(weights))

    # 应用 A股 权重限制
    a_share_cap = g.a_share_weight_cap
    if g.use_dynamic_a_share_cap and g.factor_df is not None:
        a_share_cap = _dynamic_a_share_cap(assets, g.factor_df)
    if a_share_cap is not None:
        weights = _apply_group_weight_cap(
            weights, list(assets), g.a_share_etfs, a_share_cap
        )

    # 应用行业 权重限制
    industry_cap = g.industry_weight_cap
    if g.use_dynamic_industry_cap and g.factor_df is not None:
        industry_cap = _dynamic_industry_cap(assets, g.factor_df)
    if industry_cap is not None:
        weights = _apply_group_weight_cap(
            weights, list(assets), g.industry_etfs, industry_cap
        )

    # 应用个股权重上限
    max_weight = g.max_weight
    if g.use_dynamic_max_weight and g.factor_df is not None:
        max_weight = _dynamic_max_weight(list(assets), g.factor_df)
    if max_weight:
        weights = _apply_weight_cap(weights, max_weight)

    # v2: 恢复原始仓位比例，保留波动率缩放效果
    weights = np.array(weights, dtype=float) * original_total
    # 再次归一化确保总和正确
    weights = _normalize(weights)

    result = dict(zip(assets, weights))

    return result


def _zscore(series):
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _rank_score(series, higher_is_better=True):
    if series.empty:
        return pd.Series(0.0, index=series.index)
    rank = series.rank(pct=True, ascending=True)
    if higher_is_better:
        return rank
    return 1 - rank


def _previous_trade_day(context):
    trade_days = get_trade_days(end_date=context.current_dt, count=2)
    if len(trade_days) < 2:
        return None
    return trade_days[-2]


def _get_unit_nav(etf, ref_date):
    try:
        nav = get_extras("unit_net_value", etf, end_date=ref_date, count=1)
        if nav is None or nav.empty:
            nav = get_extras(
                "unit_net_value", etf, start_date=ref_date, end_date=ref_date
            )
        if nav is None or nav.empty:
            return None
        if etf in nav.columns:
            value = nav[etf].iloc[-1]
        else:
            value = nav.iloc[-1, 0]
        if value is None or np.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def _build_anchor_signal(etfs):
    vol_forecasts = []
    for etf in etfs:
        data = g.price_cache.get(etf)
        if data is None or data.empty:
            vol_forecasts.append(np.nan)
            continue
        rets = data["close"].pct_change().dropna().tail(g.garch_window).values
        vol_forecasts.append(_forecast_volatility(rets))

    series = pd.Series(vol_forecasts, index=etfs)
    series = series.replace([np.inf, -np.inf], np.nan)
    if series.isna().all():
        return np.zeros(len(etfs))
    series = series.fillna(series.median())
    return _zscore(-series).values


def _forecast_volatility(returns):
    returns = np.array(returns, dtype=float)
    if len(returns) < 10:
        return np.std(returns) * math.sqrt(252) if len(returns) > 1 else 0.0

    if _ARCH_AVAILABLE:
        try:
            model = arch_model(returns * 100, mean="Constant", vol="GARCH", p=1, q=1)
            res = model.fit(disp="off")
            forecast = res.forecast(horizon=1, reindex=False)
            var = float(forecast.variance.values[-1, 0])
            return math.sqrt(max(var, 0.0)) / 100 * math.sqrt(252)
        except Exception:
            pass

    return _ewma_volatility(returns)


def _ewma_volatility(returns, lam=0.94):
    if len(returns) < 2:
        return 0.0
    var = returns[0] ** 2
    for r in returns[1:]:
        var = lam * var + (1 - lam) * (r**2)
    return math.sqrt(max(var, 0.0)) * math.sqrt(252)


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


def _get_signal_weights(mode, returns_df=None, signals=None):
    """
    根据不同模式计算动量/质量权重

    Args:
        mode: 信号权重模式
            'fixed': 使用固定权重(g.score_weight_momentum, g.score_weight_quality)
            'trend': 根据趋势调整（牛市动量高，熊市质量高）
            'strength': 根据信号强度调整（强信号动量高，弱信号质量高）
        returns_df: 收益率矩阵（用于趋势模式）
        signals: 信号向量（用于信号强度模式）

    Returns:
        (momentum_weight, quality_weight)
    """
    if mode == 'fixed':
        # 固定权重：使用参数中的值
        return g.score_weight_momentum, g.score_weight_quality

    elif mode == 'trend':
        # 方案2：根据趋势调整
        if returns_df is None:
            log.warn("【信号权重】trend模式需要returns_df，退回固定权重")
            return g.score_weight_momentum, g.score_weight_quality

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

        log.info(f"【信号权重】趋势={trend}, 动量权重={momentum_w}, 质量权重={quality_w}")
        return momentum_w, quality_w

    elif mode == 'strength':
        # 方案3：根据信号强度调整
        if signals is None:
            log.warn("【信号权重】strength模式需要signals，退回固定权重")
            return g.score_weight_momentum, g.score_weight_quality

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

        log.info(f"【信号权重】信号强度={avg_signal:.3f}, 动量权重={momentum_w}, 质量权重={quality_w}")
        return momentum_w, quality_w

    else:
        log.warn(f"【信号权重】未知模式={mode}，使用固定权重")
        return g.score_weight_momentum, g.score_weight_quality


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
        log.warn(f"【波动率缩放】缺少资产数据: {missing}，跳过缩放")
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

    log.info(f"【波动率缩放】市场趋势={trend}, 基础仓位={base_position:.0%}")

    # ========== 第二层：波动率微调 ==========
    # 构建权重向量
    w = np.array([target_weights.get(a, 0) for a in assets])

    # 计算组合波动率
    try:
        cov = returns_df[assets].cov().values
        portfolio_var = w @ cov @ w
        portfolio_vol = math.sqrt(max(portfolio_var, 0)) * math.sqrt(252)
    except Exception as e:
        log.warn(f"【波动率缩放】计算波动率失败: {e}，跳过缩放")
        return target_weights

    if portfolio_vol <= 0:
        log.warn(f"【波动率缩放】组合波动率为0，跳过缩放")
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
    log.info(f"【波动率缩放】组合波动率={portfolio_vol:.2%}, 目标={target_volatility:.2%}, 波动微调={vol_adjustment:.0%}, 最终仓位={final_position:.0%}")

    # 应用缩放
    scaled_weights = {a: w[i] * final_position for i, a in enumerate(assets)}

    return scaled_weights


def _apply_bond_floor(target_weights, factor_df):
    """
    债券保底配置：确保债券仓位不低于 min_bond_weight
    如果债券当前权重低于保底值，从其他资产中按比例扣减，补足债券仓位
    """
    # 检查开关：如果未启用保底机制，直接返回原权重
    if not getattr(g, "use_bond_floor", False):
        return target_weights

    if not hasattr(g, "bond_etfs") or not g.bond_etfs:
        return target_weights

    if not hasattr(g, "min_bond_weight") or g.min_bond_weight <= 0:
        return target_weights

    # 计算当前债券总权重
    bond_weight = sum(target_weights.get(etf, 0) for etf in g.bond_etfs)
    min_bond = g.min_bond_weight
    max_bond = getattr(g, "max_bond_weight", 0.40)

    # 如果债券权重已经达标，但超过上限，需要限制
    if bond_weight > max_bond:
        # 债券仓位过高，需要降低
        excess = bond_weight - max_bond
        bond_etfs_in_portfolio = [etf for etf in g.bond_etfs if etf in target_weights]

        if bond_etfs_in_portfolio and bond_weight > 0:
            # 按比例降低债券权重
            for etf in bond_etfs_in_portfolio:
                target_weights[etf] = target_weights[etf] * max_bond / bond_weight

            # 释放的权重分配给其他资产
            other_etfs = [etf for etf in target_weights if etf not in g.bond_etfs]
            if other_etfs:
                add_per_etf = excess / len(other_etfs)
                for etf in other_etfs:
                    target_weights[etf] += add_per_etf

        return target_weights

    # 如果债券权重低于保底值，需要补足
    if bond_weight < min_bond:
        deficit = min_bond - bond_weight

        # 从非债券资产中扣除权重
        non_bond_etfs = [etf for etf in target_weights if etf not in g.bond_etfs]
        non_bond_weight = sum(target_weights[etf] for etf in non_bond_etfs)

        if non_bond_weight > 0 and deficit > 0:
            # 按比例扣除非债券资产的权重
            for etf in non_bond_etfs:
                reduction = target_weights[etf] * (deficit / non_bond_weight)
                target_weights[etf] = max(0, target_weights[etf] - reduction)

            # 补足债券权重
            bond_etfs_in_pool = [etf for etf in g.bond_etfs if etf in factor_df.index]
            if bond_etfs_in_pool:
                # 根据信号强度分配债券权重
                bond_signals = {}
                for etf in bond_etfs_in_pool:
                    signal_val = factor_df.loc[etf, "signal"]
                    # 确保是标量值（处理可能是Series的情况）
                    if hasattr(signal_val, "item"):
                        signal_val = signal_val.item()
                    bond_signals[etf] = float(signal_val)

                total_signal = sum(list(bond_signals.values()))

                if total_signal > 0:
                    for etf in bond_etfs_in_pool:
                        target_weights[etf] = (
                            min_bond * bond_signals[etf] / total_signal
                        )
                else:
                    # 如果信号都为0，平均分配
                    for etf in bond_etfs_in_pool:
                        target_weights[etf] = min_bond / len(bond_etfs_in_pool)

            # 归一化确保总和为1
            total = sum(list(target_weights.values()))
            if total > 0:
                target_weights = {k: v / total for k, v in target_weights.items()}

    return target_weights


def handle_data(context, data):
    pass
