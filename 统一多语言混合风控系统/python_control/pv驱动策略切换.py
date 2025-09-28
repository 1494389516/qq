#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PV驱动策略切换风控系统
基于PV异常强度、趋势斜率、漏斗一致性等多维信号，实现动态策略切换
结合窗口抖动测试确保策略切换的稳定性

作者: 风控算法专家
用途: 企业级动态风控策略引擎
"""

import numpy as np
import pandas as pd
import datetime
import time
import logging
import json
from typing import Dict, List, Tuple, Optional, NamedTuple
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import threading
import math


class StrategyLevel(Enum):
    """策略强度等级"""
    BASIC = 1      # 基础策略
    ENHANCED = 2   # 增强策略  
    STRICT = 3     # 严格策略
    EMERGENCY = 4  # 紧急策略


class SignalType(Enum):
    """信号类型"""
    UP_TICK = "up_tick"       # 上行信号
    DOWN_TICK = "down_tick"   # 下行信号
    STABLE = "stable"         # 稳定信号
    INVALID = "invalid"       # 无效信号


@dataclass
class PVMetrics:
    """PV指标数据"""
    timestamp: datetime.datetime
    pv_count: int
    detail_pv: int
    order_count: int
    is_bot_like: bool = False
    quality_flag: bool = True
    prerender_flag: bool = False


@dataclass
class CoreSignals:
    """核心信号计算结果"""
    timestamp: datetime.datetime
    z_c: float              # PV异常强度(z分)
    slope_c: float          # PV近5分钟线性斜率
    r_ca: float             # 漏斗一致性(PV→下单)
    is_bot_like: bool       # 机器人标识
    quality_flag: bool      # 质量标识


@dataclass
class WindowStability:
    """窗口稳定性检测结果"""
    theta_up: float         # 上行一致率
    theta_down: float       # 下行一致率
    up_ticks: int          # 上行tick数量
    down_ticks: int        # 下行tick数量
    total_samples: int     # 总样本数
    window_seconds: int    # 窗口秒数


@dataclass
class StrategyDecision:
    """策略决策结果"""
    timestamp: datetime.datetime
    current_strategy: StrategyLevel
    signal_type: SignalType
    stability: WindowStability
    core_signals: CoreSignals
    decision_reason: str
    emergency_override: bool = False


class PVDrivenStrategyEngine:
    """PV驱动策略切换引擎"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # 配置参数
        self.config = self._load_default_config()
        
        # 数据存储
        self.pv_history: deque = deque(maxlen=500)  # 存储最近500个数据点
        self.signal_history: deque = deque(maxlen=100)
        self.decision_history: deque = deque(maxlen=50)
        
        # 状态管理
        self.current_strategy = StrategyLevel.BASIC
        self.last_strategy_change = datetime.datetime.now()
        self.cooling_until = None
        
        # 基线统计
        self.pv_baseline_mean = 1000.0
        self.pv_baseline_std = 200.0
        self.baseline_window = deque(maxlen=60)  # 1小时基线窗口
        
        self.logger.info("PV驱动策略引擎初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('PVStrategyEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_default_config(self) -> Dict:
        """加载默认配置参数"""
        return {
            # 核心阈值
            'tau_z': 2.0,           # z分上行阈值
            'tau_z_low': 0.8,       # z分下行阈值
            'tau_s': 0.1,           # 斜率阈值(每分钟相对基线10%提升)
            'gamma': 2.0,           # 漏斗失配上限
            
            # 一致率阈值
            'theta_up_threshold': 0.8,    # 进入策略的一致率阈值
            'theta_down_threshold': 0.9,  # 退出策略的一致率阈值
            
            # 窗口参数
            'tick_seconds': 25,           # 路由tick周期
            'window_seconds': 150,        # 滑动窗口时长(2.5分钟)
            'slope_window_minutes': 5,    # 斜率计算窗口
            
            # 冷却时间
            'cooling_minutes': 15,        # 策略切换冷却时间
            
            # 紧急规则
            'emergency_z_threshold': 0.9, # 紧急策略z分阈值
            
            # 漏斗护栏
            'funnel_check_minutes': 2,    # 失配检查时长
            'pv_weight_decay': 0.5,       # PV权重衰减系数
        }
    
    def update_baseline(self, pv_metrics: PVMetrics):
        """更新PV基线统计"""
        if not pv_metrics.is_bot_like and pv_metrics.quality_flag:
            self.baseline_window.append(pv_metrics.pv_count)
            
            if len(self.baseline_window) >= 30:  # 至少30个样本
                baseline_data = list(self.baseline_window)
                self.pv_baseline_mean = np.mean(baseline_data)
                self.pv_baseline_std = max(np.std(baseline_data), 1.0)  # 避免除零
    
    def calculate_core_signals(self, pv_metrics: PVMetrics) -> CoreSignals:
        """计算核心信号"""
        current_time = pv_metrics.timestamp
        
        # 1. 计算PV异常强度 z_C(t)
        z_c = (pv_metrics.pv_count - self.pv_baseline_mean) / self.pv_baseline_std
        
        # 2. 计算PV近5分钟线性斜率 slope_C(t)
        slope_c = self._calculate_slope(current_time)
        
        # 3. 计算漏斗一致性 r_CA(t)
        r_ca = self._calculate_funnel_consistency(pv_metrics)
        
        return CoreSignals(
            timestamp=current_time,
            z_c=float(z_c),
            slope_c=float(slope_c),
            r_ca=float(r_ca),
            is_bot_like=pv_metrics.is_bot_like,
            quality_flag=pv_metrics.quality_flag
        )
    
    def _calculate_slope(self, current_time: datetime.datetime) -> float:
        """计算PV线性斜率"""
        # 获取近5分钟的数据
        cutoff_time = current_time - datetime.timedelta(minutes=self.config['slope_window_minutes'])
        recent_data = [
            (metric.timestamp, metric.pv_count) 
            for metric in self.pv_history 
            if metric.timestamp >= cutoff_time and not metric.is_bot_like
        ]
        
        if len(recent_data) < 3:
            return 0.0
        
        # 转换为时间序列数据
        times = [(t - recent_data[0][0]).total_seconds() / 60.0 for t, _ in recent_data]
        values = [v for _, v in recent_data]
        
        # 计算线性回归斜率
        if len(times) > 1:
            slope = np.polyfit(times, values, 1)[0]
            # 归一化为相对基线的变化率
            return slope / max(self.pv_baseline_mean, 1.0)
        
        return 0.0
    
    def _calculate_funnel_consistency(self, pv_metrics: PVMetrics) -> float:
        """计算漏斗一致性"""
        if pv_metrics.detail_pv == 0:
            return 0.0
        
        # 简化的漏斗一致性：订单数/详情页PV
        consistency = pv_metrics.order_count / max(pv_metrics.detail_pv, 1)
        return consistency
    
    def evaluate_tick_conditions(self, signals: CoreSignals) -> Tuple[bool, bool]:
        """评估上行/下行条件"""
        # 应用z分变换函数 f(z_C)
        f_zc = self._transform_z_score(signals.z_c)
        
        # 上行条件：f(z_C) >= tau_z 且 slope_C >= tau_s 且 r_CA 未严重失配
        up_condition = (
            f_zc >= self.config['tau_z'] and 
            signals.slope_c >= self.config['tau_s'] and 
            signals.r_ca <= self.config['gamma'] and
            not signals.is_bot_like and
            signals.quality_flag
        )
        
        # 下行条件：f(z_C) <= tau_z_low 或 slope_C <= -tau_s
        down_condition = (
            f_zc <= self.config['tau_z_low'] or 
            signals.slope_c <= -self.config['tau_s']
        )
        
        return up_condition, down_condition
    
    def _transform_z_score(self, z_score: float) -> float:
        """z分变换函数，将z分映射到[0,1]区间"""
        # 使用sigmoid变换：f(z) = 1 / (1 + exp(-z/2))
        return 1.0 / (1.0 + math.exp(-z_score / 2.0))
    
    def calculate_window_stability(self, current_time: datetime.datetime) -> WindowStability:
        """计算窗口稳定性"""
        window_start = current_time - datetime.timedelta(seconds=self.config['window_seconds'])
        
        # 获取窗口内的信号
        window_signals = [
            s for s in self.signal_history 
            if window_start <= s.timestamp <= current_time
        ]
        
        if not window_signals:
            return WindowStability(0.0, 0.0, 0, 0, 0, self.config['window_seconds'])
        
        up_ticks = 0
        down_ticks = 0
        
        for signals in window_signals:
            up_tick, down_tick = self.evaluate_tick_conditions(signals)
            if up_tick:
                up_ticks += 1
            if down_tick:
                down_ticks += 1
        
        total_samples = len(window_signals)
        theta_up = up_ticks / total_samples if total_samples > 0 else 0.0
        theta_down = down_ticks / total_samples if total_samples > 0 else 0.0
        
        return WindowStability(
            theta_up=theta_up,
            theta_down=theta_down,
            up_ticks=up_ticks,
            down_ticks=down_ticks,
            total_samples=total_samples,
            window_seconds=self.config['window_seconds']
        )
    
    def check_emergency_conditions(self, signals: CoreSignals) -> bool:
        """检查紧急条件"""
        f_zc = self._transform_z_score(signals.z_c)
        return f_zc >= self.config['emergency_z_threshold']
    
    def check_funnel_guard(self, current_time: datetime.datetime) -> bool:
        """检查漏斗护栏"""
        cutoff_time = current_time - datetime.timedelta(minutes=self.config['funnel_check_minutes'])
        
        recent_signals = [
            s for s in self.signal_history 
            if s.timestamp >= cutoff_time
        ]
        
        if len(recent_signals) < 2:
            return False
        
        # 检查r_CA是否持续异常
        abnormal_count = sum(1 for s in recent_signals if s.r_ca > self.config['gamma'])
        abnormal_ratio = abnormal_count / len(recent_signals)
        
        return abnormal_ratio > 0.8  # 80%以上样本异常
    
    def make_strategy_decision(self, pv_metrics: PVMetrics) -> StrategyDecision:
        """做出策略决策"""
        current_time = pv_metrics.timestamp
        
        # 更新基线
        self.update_baseline(pv_metrics)
        
        # 存储PV数据
        self.pv_history.append(pv_metrics)
        
        # 计算核心信号
        signals = self.calculate_core_signals(pv_metrics)
        self.signal_history.append(signals)
        
        # 检查冷却时间
        if self.cooling_until and current_time < self.cooling_until:
            return StrategyDecision(
                timestamp=current_time,
                current_strategy=self.current_strategy,
                signal_type=SignalType.STABLE,
                stability=self.calculate_window_stability(current_time),
                core_signals=signals,
                decision_reason="冷却期内，保持当前策略"
            )
        
        # 检查紧急条件
        if self.check_emergency_conditions(signals):
            self.current_strategy = StrategyLevel.EMERGENCY
            self.last_strategy_change = current_time
            self.cooling_until = current_time + datetime.timedelta(
                minutes=self.config['cooling_minutes']
            )
            
            return StrategyDecision(
                timestamp=current_time,
                current_strategy=self.current_strategy,
                signal_type=SignalType.UP_TICK,
                stability=self.calculate_window_stability(current_time),
                core_signals=signals,
                decision_reason="触发紧急条件，立即升级策略",
                emergency_override=True
            )
        
        # 检查漏斗护栏
        funnel_abnormal = self.check_funnel_guard(current_time)
        if funnel_abnormal:
            self.logger.warning("检测到漏斗失配异常，PV权重降权")
        
        # 计算窗口稳定性
        stability = self.calculate_window_stability(current_time)
        
        # 策略切换决策
        signal_type = SignalType.STABLE
        decision_reason = "保持当前策略"
        strategy_changed = False
        
        # 上行决策：需要升级策略
        if (stability.theta_up >= self.config['theta_up_threshold'] and 
            self.current_strategy != StrategyLevel.EMERGENCY):
            
            next_level = min(StrategyLevel.EMERGENCY.value, self.current_strategy.value + 1)
            new_strategy = StrategyLevel(next_level)
            
            if new_strategy != self.current_strategy:
                self.current_strategy = new_strategy
                signal_type = SignalType.UP_TICK
                decision_reason = f"上行一致率{stability.theta_up:.3f}达到阈值，升级策略"
                strategy_changed = True
        
        # 下行决策：需要降级策略
        elif (stability.theta_down >= self.config['theta_down_threshold'] and 
              self.current_strategy != StrategyLevel.BASIC):
            
            next_level = max(StrategyLevel.BASIC.value, self.current_strategy.value - 1)
            new_strategy = StrategyLevel(next_level)
            
            if new_strategy != self.current_strategy:
                self.current_strategy = new_strategy
                signal_type = SignalType.DOWN_TICK
                decision_reason = f"下行一致率{stability.theta_down:.3f}达到阈值，降级策略"
                strategy_changed = True
        
        # 如果策略发生变化，设置冷却时间
        if strategy_changed:
            self.last_strategy_change = current_time
            self.cooling_until = current_time + datetime.timedelta(
                minutes=self.config['cooling_minutes']
            )
        
        decision = StrategyDecision(
            timestamp=current_time,
            current_strategy=self.current_strategy,
            signal_type=signal_type,
            stability=stability,
            core_signals=signals,
            decision_reason=decision_reason
        )
        
        # 记录决策历史
        self.decision_history.append(decision)
        
        # 日志记录
        self.logger.info(
            f"策略决策 - 时间: {current_time.strftime('%H:%M:%S')}, "
            f"策略: {self.current_strategy.name}, 信号: {signal_type.value}, "
            f"z分: {signals.z_c:.3f}, 斜率: {signals.slope_c:.3f}, "
            f"一致率↑: {stability.theta_up:.3f}, 一致率↓: {stability.theta_down:.3f}, "
            f"原因: {decision_reason}"
        )
        
        return decision
    
    def get_current_status(self) -> Dict:
        """获取当前状态信息"""
        current_time = datetime.datetime.now()
        
        status = {
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_strategy": self.current_strategy.name,
            "cooling_until": self.cooling_until.strftime("%H:%M:%S") if self.cooling_until else None,
            "pv_baseline": {
                "mean": round(self.pv_baseline_mean, 2),
                "std": round(self.pv_baseline_std, 2),
                "samples": len(self.baseline_window)
            },
            "config": self.config,
            "data_points": {
                "pv_history": len(self.pv_history),
                "signal_history": len(self.signal_history),
                "decision_history": len(self.decision_history)
            }
        }
        
        return status
    
    def generate_audit_report(self, hours: int = 1) -> Dict:
        """生成审计报告"""
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=hours)
        
        # 筛选时间范围内的决策
        period_decisions = [
            d for d in self.decision_history
            if start_time <= d.timestamp <= end_time
        ]
        
        if not period_decisions:
            return {"message": "指定时间范围内无决策记录"}
        
        # 统计分析
        strategy_distribution = {}
        signal_distribution = {}
        
        for decision in period_decisions:
            strategy_distribution[decision.current_strategy.name] = \
                strategy_distribution.get(decision.current_strategy.name, 0) + 1
            signal_distribution[decision.signal_type.value] = \
                signal_distribution.get(decision.signal_type.value, 0) + 1
        
        # 计算平均指标
        avg_z_score = np.mean([d.core_signals.z_c for d in period_decisions])
        avg_slope = np.mean([d.core_signals.slope_c for d in period_decisions])
        avg_theta_up = np.mean([d.stability.theta_up for d in period_decisions])
        avg_theta_down = np.mean([d.stability.theta_down for d in period_decisions])
        
        return {
            "report_period": f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}",
            "total_decisions": len(period_decisions),
            "strategy_distribution": strategy_distribution,
            "signal_distribution": signal_distribution,
            "average_metrics": {
                "z_score": round(avg_z_score, 3),
                "slope": round(avg_slope, 3),
                "theta_up": round(avg_theta_up, 3),
                "theta_down": round(avg_theta_down, 3)
            },
            "recent_decisions": [
                {
                    "time": d.timestamp.strftime("%H:%M:%S"),
                    "strategy": d.current_strategy.name,
                    "signal": d.signal_type.value,
                    "z_score": round(d.core_signals.z_c, 3),
                    "theta_up": round(d.stability.theta_up, 3),
                    "reason": d.decision_reason
                }
                for d in period_decisions[-10:]  # 最近10个决策
            ]
        }


def demo_pv_strategy_engine():
    """演示PV驱动策略引擎"""
    print("=== PV驱动策略切换系统演示 ===\n")
    
    # 创建策略引擎
    engine = PVDrivenStrategyEngine()
    
    # 显示当前状态
    status = engine.get_current_status()
    print("系统状态:")
    print(f"  当前策略: {status['current_strategy']}")
    print(f"  PV基线: 均值={status['pv_baseline']['mean']}, 标准差={status['pv_baseline']['std']}")
    print(f"  配置参数: tau_z={status['config']['tau_z']}, tau_s={status['config']['tau_s']}")
    print()
    
    # 模拟PV数据流
    base_time = datetime.datetime.now()
    
    # 模拟不同场景的PV数据
    test_scenarios = [
        # 正常流量
        {"pv": 800, "detail": 400, "orders": 40, "desc": "正常低峰流量"},
        {"pv": 1200, "detail": 600, "orders": 60, "desc": "正常高峰流量"},
        
        # PV突增场景
        {"pv": 2500, "detail": 1000, "orders": 80, "desc": "PV异常突增"},
        {"pv": 3000, "detail": 1200, "orders": 90, "desc": "PV持续高位"},
        {"pv": 3500, "detail": 1400, "orders": 100, "desc": "PV进一步上升"},
        
        # 机器人流量
        {"pv": 4000, "detail": 200, "orders": 5, "bot": True, "desc": "疑似机器人流量"},
        
        # 流量回落
        {"pv": 2000, "detail": 800, "orders": 70, "desc": "流量开始回落"},
        {"pv": 1000, "detail": 500, "orders": 50, "desc": "流量恢复正常"},
    ]
    
    print("模拟PV数据流处理:")
    decisions = []
    
    for i, scenario in enumerate(test_scenarios):
        # 创建PV指标
        pv_metrics = PVMetrics(
            timestamp=base_time + datetime.timedelta(minutes=i*3),
            pv_count=scenario["pv"],
            detail_pv=scenario["detail"],
            order_count=scenario["orders"],
            is_bot_like=scenario.get("bot", False),
            quality_flag=not scenario.get("bot", False)
        )
        
        # 进行策略决策
        decision = engine.make_strategy_decision(pv_metrics)
        decisions.append(decision)
        
        print(f"  场景{i+1}: {scenario['desc']}")
        print(f"    PV={scenario['pv']}, 详情PV={scenario['detail']}, 订单={scenario['orders']}")
        print(f"    z分={decision.core_signals.z_c:.3f}, 斜率={decision.core_signals.slope_c:.3f}")
        print(f"    策略={decision.current_strategy.name}, 信号={decision.signal_type.value}")
        print(f"    决策原因: {decision.decision_reason}")
        print()
    
    # 生成审计报告
    print("=" * 60)
    report = engine.generate_audit_report(1)
    print("\n审计报告:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo_pv_strategy_engine()