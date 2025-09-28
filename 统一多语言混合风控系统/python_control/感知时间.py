#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
感知时间风控算法
实现基于时间的风控策略切换，根据白天/晚上时段触发不同的风控算法
适用于电商、金融等对时间敏感的风控场景

作者: 风控算法专家
用途: 时间感知的智能风控决策系统
"""

import datetime
import time
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import json


class TimeRiskLevel(Enum):
    """时间风险等级枚举"""
    LOW = "low"           # 低风险时段
    MEDIUM = "medium"     # 中等风险时段  
    HIGH = "high"         # 高风险时段
    CRITICAL = "critical" # 极高风险时段


class RiskStrategy(Enum):
    """风控策略枚举"""
    BASIC = "basic"               # 基础策略
    ENHANCED = "enhanced"         # 增强策略
    STRICT = "strict"            # 严格策略
    EMERGENCY = "emergency"       # 紧急策略


@dataclass
class TimeRiskConfig:
    """时间风险配置"""
    start_hour: int
    end_hour: int
    risk_level: TimeRiskLevel
    strategy: RiskStrategy
    description: str


@dataclass
class RiskEvent:
    """风控事件"""
    timestamp: datetime.datetime
    user_id: str
    action: str
    risk_score: float
    time_risk_factor: float
    final_decision: str


class TimePerceptionRiskEngine:
    """时间感知风控引擎"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.risk_configs = self._initialize_risk_configs()
        self.strategy_rules = self._initialize_strategy_rules()
        self.event_history: List[RiskEvent] = []
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('TimeRiskEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_risk_configs(self) -> List[TimeRiskConfig]:
        """初始化时间风险配置"""
        configs = [
            # 深夜高风险时段 (00:00-06:00)
            TimeRiskConfig(0, 6, TimeRiskLevel.HIGH, RiskStrategy.STRICT,
                          "深夜时段，异常活动频发，启用严格策略"),
            
            # 早晨低风险时段 (06:00-09:00)  
            TimeRiskConfig(6, 9, TimeRiskLevel.LOW, RiskStrategy.BASIC,
                          "早晨时段，用户活动正常，使用基础策略"),
            
            # 上午中等风险时段 (09:00-12:00)
            TimeRiskConfig(9, 12, TimeRiskLevel.MEDIUM, RiskStrategy.ENHANCED,
                          "上午工作时段，中等风险，使用增强策略"),
            
            # 下午低风险时段 (12:00-18:00)
            TimeRiskConfig(12, 18, TimeRiskLevel.LOW, RiskStrategy.BASIC,
                          "下午时段，正常业务高峰，使用基础策略"),
            
            # 晚上中等风险时段 (18:00-22:00)
            TimeRiskConfig(18, 22, TimeRiskLevel.MEDIUM, RiskStrategy.ENHANCED,
                          "晚上时段，娱乐消费高峰，使用增强策略"),
            
            # 深夜极高风险时段 (22:00-24:00)
            TimeRiskConfig(22, 24, TimeRiskLevel.CRITICAL, RiskStrategy.EMERGENCY,
                          "深夜时段，欺诈高发期，启用紧急策略")
        ]
        
        self.logger.info(f"已加载 {len(configs)} 个时间风险配置")
        return configs
    
    def _initialize_strategy_rules(self) -> Dict[RiskStrategy, Dict]:
        """初始化策略规则"""
        rules = {
            RiskStrategy.BASIC: {
                "risk_threshold": 0.3,
                "verification_required": False,
                "max_attempts": 10,
                "block_duration": 300,  # 5分钟
                "description": "基础风控策略"
            },
            RiskStrategy.ENHANCED: {
                "risk_threshold": 0.2,
                "verification_required": True,
                "max_attempts": 5,
                "block_duration": 900,  # 15分钟
                "description": "增强风控策略"
            },
            RiskStrategy.STRICT: {
                "risk_threshold": 0.1,
                "verification_required": True,
                "max_attempts": 3,
                "block_duration": 1800,  # 30分钟
                "description": "严格风控策略"
            },
            RiskStrategy.EMERGENCY: {
                "risk_threshold": 0.05,
                "verification_required": True,
                "max_attempts": 1,
                "block_duration": 3600,  # 1小时
                "description": "紧急风控策略"
            }
        }
        
        self.logger.info("策略规则初始化完成")
        return rules
    
    def get_current_time_config(self, current_time: Optional[datetime.datetime] = None) -> TimeRiskConfig:
        """获取当前时间的风险配置"""
        if current_time is None:
            current_time = datetime.datetime.now()
            
        current_hour = current_time.hour
        
        for config in self.risk_configs:
            if config.start_hour <= current_hour < config.end_hour:
                return config
        
        # 处理跨零点的情况 (22:00-24:00 对应 22-0)
        for config in self.risk_configs:
            if config.start_hour > config.end_hour:  # 跨零点
                if current_hour >= config.start_hour or current_hour < config.end_hour:
                    return config
        
        # 默认返回中等风险配置
        return TimeRiskConfig(0, 24, TimeRiskLevel.MEDIUM, RiskStrategy.ENHANCED, "默认配置")
    
    def is_daytime(self, current_time: Optional[datetime.datetime] = None) -> bool:
        """判断是否为白天"""
        if current_time is None:
            current_time = datetime.datetime.now()
            
        hour = current_time.hour
        # 定义白天时间为 6:00-18:00
        return 6 <= hour < 18
    
    def calculate_time_risk_factor(self, current_time: Optional[datetime.datetime] = None) -> float:
        """计算时间风险因子"""
        config = self.get_current_time_config(current_time)
        
        # 根据风险等级计算风险因子
        risk_factors = {
            TimeRiskLevel.LOW: 0.8,
            TimeRiskLevel.MEDIUM: 1.0,
            TimeRiskLevel.HIGH: 1.5,
            TimeRiskLevel.CRITICAL: 2.0
        }
        
        return risk_factors.get(config.risk_level, 1.0)
    
    def evaluate_risk(self, user_id: str, action: str, base_risk_score: float,
                     current_time: Optional[datetime.datetime] = None) -> RiskEvent:
        """评估风险并做出决策"""
        if current_time is None:
            current_time = datetime.datetime.now()
            
        # 获取时间配置和风险因子
        time_config = self.get_current_time_config(current_time)
        time_risk_factor = self.calculate_time_risk_factor(current_time)
        
        # 计算最终风险分数
        final_risk_score = base_risk_score * time_risk_factor
        
        # 获取策略规则
        strategy_rule = self.strategy_rules[time_config.strategy]
        
        # 做出决策
        if final_risk_score >= strategy_rule["risk_threshold"]:
            if strategy_rule["verification_required"]:
                decision = "VERIFY"
            else:
                decision = "BLOCK"
        else:
            decision = "ALLOW"
        
        # 创建风控事件
        event = RiskEvent(
            timestamp=current_time,
            user_id=user_id,
            action=action,
            risk_score=final_risk_score,
            time_risk_factor=time_risk_factor,
            final_decision=decision
        )
        
        # 记录事件
        self.event_history.append(event)
        
        # 日志记录
        self.logger.info(
            f"风控决策 - 用户: {user_id}, 动作: {action}, "
            f"基础风险: {base_risk_score:.3f}, 时间因子: {time_risk_factor:.2f}, "
            f"最终风险: {final_risk_score:.3f}, 决策: {decision}, "
            f"策略: {time_config.strategy.value}, 时段: {time_config.description}"
        )
        
        return event
    
    def get_current_strategy_info(self) -> Dict:
        """获取当前策略信息"""
        current_time = datetime.datetime.now()
        config = self.get_current_time_config(current_time)
        is_day = self.is_daytime(current_time)
        
        return {
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_daytime": is_day,
            "time_period": "白天" if is_day else "夜晚",
            "risk_level": config.risk_level.value,
            "strategy": config.strategy.value,
            "description": config.description,
            "time_risk_factor": self.calculate_time_risk_factor(current_time),
            "strategy_rules": self.strategy_rules[config.strategy]
        }
    
    def generate_report(self, hours: int = 24) -> Dict:
        """生成风控报告"""
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=hours)
        
        # 筛选时间范围内的事件
        period_events = [
            event for event in self.event_history
            if start_time <= event.timestamp <= end_time
        ]
        
        if not period_events:
            return {"message": "指定时间范围内无风控事件"}
        
        # 统计分析
        total_events = len(period_events)
        decisions = {}
        risk_levels = {}
        
        for event in period_events:
            # 统计决策分布
            decisions[event.final_decision] = decisions.get(event.final_decision, 0) + 1
            
            # 统计风险等级分布
            config = self.get_current_time_config(event.timestamp)
            risk_levels[config.risk_level.value] = risk_levels.get(config.risk_level.value, 0) + 1
        
        # 计算平均风险分数
        avg_risk_score = sum(event.risk_score for event in period_events) / total_events
        avg_time_factor = sum(event.time_risk_factor for event in period_events) / total_events
        
        return {
            "report_period": f"{start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}",
            "total_events": total_events,
            "average_risk_score": round(avg_risk_score, 3),
            "average_time_factor": round(avg_time_factor, 2),
            "decision_distribution": decisions,
            "risk_level_distribution": risk_levels,
            "recent_events": [
                {
                    "time": event.timestamp.strftime("%H:%M:%S"),
                    "user": event.user_id,
                    "action": event.action,
                    "risk": round(event.risk_score, 3),
                    "decision": event.final_decision
                }
                for event in period_events[-10:]  # 最近10个事件
            ]
        }


def demo_risk_engine():
    """演示时间感知风控引擎"""
    print("=== 时间感知风控算法演示 ===\n")
    
    # 创建风控引擎
    engine = TimePerceptionRiskEngine()
    
    # 显示当前策略信息
    current_info = engine.get_current_strategy_info()
    print("当前风控策略信息:")
    print(f"  当前时间: {current_info['current_time']}")
    print(f"  时间段: {current_info['time_period']}")
    print(f"  风险等级: {current_info['risk_level']}")
    print(f"  策略: {current_info['strategy']}")
    print(f"  策略描述: {current_info['description']}")
    print(f"  时间风险因子: {current_info['time_risk_factor']}")
    print()
    
    # 模拟不同时间段的风控决策
    test_scenarios = [
        ("user001", "login", 0.15, datetime.datetime(2024, 1, 1, 2, 30)),   # 深夜登录
        ("user002", "transfer", 0.25, datetime.datetime(2024, 1, 1, 8, 15)), # 早晨转账
        ("user003", "purchase", 0.35, datetime.datetime(2024, 1, 1, 14, 45)), # 下午购买
        ("user004", "withdraw", 0.18, datetime.datetime(2024, 1, 1, 23, 20)), # 深夜提现
    ]
    
    print("模拟风控决策:")
    for user_id, action, base_risk, test_time in test_scenarios:
        event = engine.evaluate_risk(user_id, action, base_risk, test_time)
        config = engine.get_current_time_config(test_time)
        
        print(f"  时间: {test_time.strftime('%H:%M')} | "
              f"用户: {user_id} | 动作: {action} | "
              f"基础风险: {base_risk:.2f} | 时间因子: {event.time_risk_factor:.2f} | "
              f"最终风险: {event.risk_score:.3f} | 决策: {event.final_decision} | "
              f"策略: {config.strategy.value}")
    
    print("\n" + "="*60)
    
    # 生成报告
    report = engine.generate_report(24)
    print("\n风控报告:")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo_risk_engine()