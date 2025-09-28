
"""
场景感知器（Detector）算法 v1.0
用于检测和防范爬虫等恶意行为，实现智能策略联动

核心功能：
1. 多维度KPI监控（下单、支付、爬虫、风控）
2. 基于历史基线的异常检测
3. 状态机驱动的场景切换
4. 策略联动与风险控制
"""
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import sqlite3
import threading
from abc import ABC, abstractmethod


class SceneState(Enum):
    """场景状态枚举"""
    NORMAL = "NORMAL"
    CAMPAIGN_STRICT = "CAMPAIGN_STRICT"
    NIGHT_STRICT = "NIGHT_STRICT"
    EMERGENCY = "EMERGENCY"


@dataclass
class KPIData:
    """KPI数据结构"""
    timestamp: float
    order_requests: int  # x_A: 每分钟下单请求数
    payment_success: int  # x_B: 每分钟支付成功数
    product_pv: int  # x_C: 商品详情PV
    risk_hits: int  # x_D: 风控命中次数
    time_offset: float = 0.0  # 时间偏差
    source_entropy: float = 0.0  # 来源熵（反作弊）
    ip_entropy: float = 0.0  # IP熵
    # PV驱动策略切换新增字段
    detail_pv: int = 0  # 详情页PV（用于漏斗分析）
    is_bot_like: bool = False  # 是否机器人流量
    quality_flag: bool = True  # 质量标识
    prerender_flag: bool = False  # 预渲染标识


@dataclass
class BaselineStats:
    """基线统计数据"""
    baseline: float
    std_dev: float
    mad: float  # 中位绝对偏差
    sample_count: int


@dataclass
class AnomalyScore:
    """异常分数"""
    z_score: float
    ratio: float
    normalized_score: float
    derivative: float = 0.0
    # PV驱动策略新增字段
    slope_c: float = 0.0  # PV斜率
    f_z_score: float = 0.0  # 变换后的z分数
    

@dataclass
class PVDrivenSignals:
    """PV驱动信号"""
    timestamp: float
    z_c: float  # PV异常强度
    slope_c: float  # PV近5分钟线性斜率
    r_ca: float  # 漏斗一致性
    up_tick: bool = False  # 上行信号
    down_tick: bool = False  # 下行信号
    theta_up: float = 0.0  # 上行一致率
    theta_down: float = 0.0  # 下行一致率


@dataclass
class DetectorResult:
    """检测器输出结果"""
    timestamp: float
    scene_score: float  # S(t)
    state: SceneState
    confidence: float
    reason_snapshot: Dict[str, Any]
    policy_recommendation: str
    anomaly_scores: Dict[str, AnomalyScore]


class DetectorConfig:
    """检测器配置"""
    def __init__(self):
        # 权重配置
        self.weights = {
            'order_requests': 0.5,
            'payment_success': 0.2,
            'product_pv': 0.1,
            'risk_hits': 0.2
        }
        
        # PV驱动策略切换配置
        self.pv_driven_config = {
            'tau_z': 2.0,  # z分上行阈值
            'tau_z_low': 0.8,  # z分下行阈值
            'tau_s': 0.1,  # 斜率阈值
            'gamma': 2.0,  # 漏斗失配上限
            'theta_up_threshold': 0.8,  # 进入策略的一致率阈值
            'theta_down_threshold': 0.9,  # 退出策略的一致率阈值
            'tick_seconds': 25,  # 路由tick周期
            'window_seconds': 150,  # 滑动窗口时长
            'slope_window_minutes': 5,  # 斜率计算窗口
            'emergency_z_threshold': 0.9,  # 紧急策略z分阈值
            'funnel_check_minutes': 2,  # 失配检查时长
            'pv_weight_decay': 0.5  # PV权重衰减系数
        }
        
        # 夜间权重调整
        self.night_weight_adjust = {
            'payment_success': 0.05,
            'risk_hits': 0.05,
            'order_requests': -0.10
        }
        
        # 状态切换阈值
        self.thresholds = {
            'campaign_up': 0.70,
            'campaign_down': 0.30,
            'night_strict': 0.60,
            'emergency_z': 6.0,
            'emergency_risk': 0.9,
            'emergency_ratio': 3.0
        }
        
        # 时间参数（分钟）
        self.time_params = {
            'duration_up': 3,
            'duration_down': 10,
            'cooldown': 15,
            'emergency_duration': 1
        }
        
        # 夜间时段
        self.night_hours = (0, 6)
        
        # 基线建模参数
        self.baseline_weeks = 6
        self.min_samples = 10
        self.ema_alpha = 0.1
        
        # 时间偏差阈值（秒）
        self.time_offset_threshold = 300
        
        # PV基线统计
        self.pv_baseline_mean = 1000.0
        self.pv_baseline_std = 200.0


class BaselineManager:
    """基线管理器"""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.historical_data = defaultdict(list)  # 按槽位存储历史数据
        self.baselines = {}  # 缓存的基线
        
    def _get_slot_key(self, timestamp: float) -> str:
        """获取时间槽位键（周几+小时）"""
        dt = datetime.fromtimestamp(timestamp)
        return f"{dt.weekday()}_{dt.hour}"
    
    def update_historical_data(self, kpi_data: KPIData):
        """更新历史数据"""
        slot_key = self._get_slot_key(kpi_data.timestamp)
        data_point = {
            'timestamp': kpi_data.timestamp,
            'order_requests': kpi_data.order_requests,
            'payment_success': kpi_data.payment_success,
            'product_pv': kpi_data.product_pv,
            'risk_hits': kpi_data.risk_hits
        }
        
        self.historical_data[slot_key].append(data_point)
        
        # 保持历史数据在指定周数内
        cutoff_time = time.time() - (self.config.baseline_weeks * 7 * 24 * 3600)
        self.historical_data[slot_key] = [
            dp for dp in self.historical_data[slot_key] 
            if dp['timestamp'] > cutoff_time
        ]
    
    def get_baseline_stats(self, timestamp: float, metric: str) -> BaselineStats:
        """获取基线统计"""
        slot_key = self._get_slot_key(timestamp)
        
        if slot_key not in self.historical_data:
            return self._get_fallback_baseline(timestamp, metric)
        
        values = [dp[metric] for dp in self.historical_data[slot_key]]
        
        if len(values) < self.config.min_samples:
            return self._get_fallback_baseline(timestamp, metric)
        
        baseline = np.median(values)
        mad = np.median(np.abs(np.array(values) - baseline))
        std_dev = 1.4826 * mad  # 从MAD估算标准差
        
        return BaselineStats(
            baseline=float(baseline),
            std_dev=float(max(std_dev, 1e-6)),  # 避免除零
            mad=float(mad),
            sample_count=len(values)
        )
    
    def _get_fallback_baseline(self, timestamp: float, metric: str) -> BaselineStats:
        """获取兜底基线"""
        # 尝试相邻小时
        dt = datetime.fromtimestamp(timestamp)
        for hour_offset in [-1, 1, -2, 2]:
            adj_hour = (dt.hour + hour_offset) % 24
            adj_key = f"{dt.weekday()}_{adj_hour}"
            if adj_key in self.historical_data:
                values = [dp[metric] for dp in self.historical_data[adj_key]]
                if len(values) >= self.config.min_samples:
                    baseline = np.median(values)
                    mad = np.median(np.abs(np.array(values) - baseline))
                    std_dev = 1.4826 * mad
                    return BaselineStats(
                        baseline=float(baseline),
                        std_dev=float(max(std_dev, 1e-6)),
                        mad=float(mad),
                        sample_count=len(values)
                    )
        
        # 全局兜底
        return BaselineStats(
            baseline=10.0 if metric == 'order_requests' else 5.0,
            std_dev=5.0,
            mad=3.0,
            sample_count=0
        )


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.recent_scores = defaultdict(lambda: deque(maxlen=5))  # 用于计算导数
        # PV驱动策略新增
        self.pv_history = deque(maxlen=100)  # PV历史数据
        self.signal_history = deque(maxlen=20)  # 信号历史
        self.pv_baseline_window = deque(maxlen=60)  # PV基线窗口
    
    def update_pv_baseline(self, kpi_data: KPIData):
        """更新PV基线统计"""
        if not kpi_data.is_bot_like and kpi_data.quality_flag:
            self.pv_baseline_window.append(kpi_data.product_pv)
            
            if len(self.pv_baseline_window) >= 30:
                baseline_data = list(self.pv_baseline_window)
                self.config.pv_baseline_mean = float(np.mean(baseline_data))
                self.config.pv_baseline_std = float(max(np.std(baseline_data), 1.0))
    
    def calculate_pv_slope(self, current_time: float) -> float:
        """计算PV线性斜率"""
        # 获取近5分钟的数据
        cutoff_time = current_time - (self.config.pv_driven_config['slope_window_minutes'] * 60)
        recent_data = [
            (data.timestamp, data.product_pv) 
            for data in self.pv_history 
            if data.timestamp >= cutoff_time and not data.is_bot_like
        ]
        
        if len(recent_data) < 3:
            return 0.0
        
        # 转换为时间序列数据
        times = [(t - recent_data[0][0]) / 60.0 for t, _ in recent_data]  # 转换为分钟
        values = [v for _, v in recent_data]
        
        # 计算线性回归斜率
        if len(times) > 1:
            slope = np.polyfit(times, values, 1)[0]
            # 归一化为相对基线的变化率
            return slope / max(self.config.pv_baseline_mean, 1.0)
        
        return 0.0
    
    def calculate_funnel_consistency(self, kpi_data: KPIData) -> float:
        """计算漏斗一致性"""
        if kpi_data.detail_pv == 0:
            return 0.0
        # 简化的漏斗一致性：订单数/详情页PV
        consistency = kpi_data.order_requests / max(kpi_data.detail_pv, 1)
        return consistency
    
    def transform_z_score(self, z_score: float) -> float:
        """变换z分数到[0,1]区间"""
        # 使用sigmoid变换：f(z) = 1 / (1 + exp(-z/2))
        import math
        return 1.0 / (1.0 + math.exp(-z_score / 2.0))
    
    def calculate_pv_driven_signals(self, kpi_data: KPIData) -> PVDrivenSignals:
        """计算PV驱动信号"""
        # 更新PV历史和基线
        self.pv_history.append(kpi_data)
        self.update_pv_baseline(kpi_data)
        
        # 1. 计算PV异常强度 z_C(t)
        z_c = (kpi_data.product_pv - self.config.pv_baseline_mean) / self.config.pv_baseline_std
        
        # 2. 计算PV斜率 slope_C(t)
        slope_c = self.calculate_pv_slope(kpi_data.timestamp)
        
        # 3. 计算漏斗一致性 r_CA(t)
        r_ca = self.calculate_funnel_consistency(kpi_data)
        
        # 4. 计算tick条件
        f_zc = self.transform_z_score(z_c)
        config = self.config.pv_driven_config
        
        # 上行条件
        up_tick = (
            f_zc >= config['tau_z'] and 
            slope_c >= config['tau_s'] and 
            r_ca <= config['gamma'] and
            not kpi_data.is_bot_like and
            kpi_data.quality_flag
        )
        
        # 下行条件
        down_tick = (
            f_zc <= config['tau_z_low'] or 
            slope_c <= -config['tau_s']
        )
        
        signals = PVDrivenSignals(
            timestamp=kpi_data.timestamp,
            z_c=z_c,
            slope_c=slope_c,
            r_ca=r_ca,
            up_tick=up_tick,
            down_tick=down_tick
        )
        
        # 存储信号历史
        self.signal_history.append(signals)
        
        # 计算窗口稳定性
        theta_up, theta_down = self.calculate_window_stability(kpi_data.timestamp)
        signals.theta_up = theta_up
        signals.theta_down = theta_down
        
        return signals
    
    def calculate_window_stability(self, current_time: float) -> Tuple[float, float]:
        """计算窗口稳定性"""
        config = self.config.pv_driven_config
        window_start = current_time - config['window_seconds']
        
        # 获取窗口内的信号
        window_signals = [
            s for s in self.signal_history 
            if window_start <= s.timestamp <= current_time
        ]
        
        if not window_signals:
            return 0.0, 0.0
        
        up_ticks = sum(1 for s in window_signals if s.up_tick)
        down_ticks = sum(1 for s in window_signals if s.down_tick)
        total_samples = len(window_signals)
        
        theta_up = up_ticks / total_samples if total_samples > 0 else 0.0
        theta_down = down_ticks / total_samples if total_samples > 0 else 0.0
        
        return theta_up, theta_down
    
    def calculate_anomaly_scores(self, kpi_data: KPIData, 
                                baseline_stats: Dict[str, BaselineStats]) -> Dict[str, AnomalyScore]:
        """计算异常分数"""
        scores = {}
        
        metrics = ['order_requests', 'payment_success', 'product_pv', 'risk_hits']
        
        for metric in metrics:
            value = getattr(kpi_data, metric)
            stats = baseline_stats[metric]
            
            # Z分数
            z_score = (value - stats.baseline) / stats.std_dev
            
            # 倍率
            ratio = value / (stats.baseline + 1e-6)
            
            # 归一化分数
            normalized_score = min(max(z_score, 0), 3) / 3
            
            # 计算导数特征
            self.recent_scores[metric].append(normalized_score)
            derivative = 0.0
            if len(self.recent_scores[metric]) >= 2:
                scores_array = list(self.recent_scores[metric])
                derivative = np.polyfit(range(len(scores_array)), scores_array, 1)[0]
            
            scores[metric] = AnomalyScore(
                z_score=z_score,
                ratio=ratio,
                normalized_score=normalized_score,
                derivative=derivative
            )
        
        return scores


class StateManager:
    """状态管理器"""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.current_state = SceneState.NORMAL
        self.state_start_time = time.time()
        self.state_duration_counter = 0
        self.last_state_change = time.time()
        self.emergency_start_time = None
    
    def is_night_time(self, timestamp: float) -> bool:
        """判断是否为夜间"""
        dt = datetime.fromtimestamp(timestamp)
        start_hour, end_hour = self.config.night_hours
        return start_hour <= dt.hour < end_hour
    
    def check_emergency_conditions(self, anomaly_scores: Dict[str, AnomalyScore]) -> bool:
        """检查紧急状态条件"""
        # 条件1: z_A >= 6
        if anomaly_scores['order_requests'].z_score >= self.config.thresholds['emergency_z']:
            return True
        
        # 条件2: f(z_D) >= 0.9
        if anomaly_scores['risk_hits'].normalized_score >= self.config.thresholds['emergency_risk']:
            return True
        
        # 条件3: r_A >= 3
        if anomaly_scores['order_requests'].ratio >= self.config.thresholds['emergency_ratio']:
            return True
        
        return False
    
    def update_state(self, scene_score: float, anomaly_scores: Dict[str, AnomalyScore], 
                    timestamp: float, pv_signals: Optional[PVDrivenSignals] = None) -> Tuple[SceneState, str]:
        """更新状态机（集成PV驱动策略）"""
        current_time = time.time()
        is_night = self.is_night_time(timestamp)
        
        # 检查紧急状态
        if self.check_emergency_conditions(anomaly_scores):
            if self.emergency_start_time is None:
                self.emergency_start_time = current_time
            elif current_time - self.emergency_start_time >= self.config.time_params['emergency_duration'] * 60:
                if self.current_state != SceneState.EMERGENCY:
                    self._change_state(SceneState.EMERGENCY, current_time)
                    return self.current_state, "紧急状态触发"
        else:
            self.emergency_start_time = None
        
        # PV驱动策略检查（优先级高于普通策略）
        if pv_signals:
            has_pv_decision, pv_decision = self._evaluate_pv_driven_strategy(pv_signals, current_time)
            if has_pv_decision:  # 有PV驱动决策
                return pv_decision
        
        # 冷却期检查
        if (current_time - self.last_state_change < self.config.time_params['cooldown'] * 60 and 
            self.current_state == SceneState.NORMAL):
            return self.current_state, "冷却期内"
        
        # 原有的状态转换逻辑
        if self.current_state == SceneState.NORMAL:
            if scene_score >= self.config.thresholds['campaign_up']:
                self.state_duration_counter += 1
                if self.state_duration_counter >= self.config.time_params['duration_up']:
                    if is_night:
                        self._change_state(SceneState.NIGHT_STRICT, current_time)
                        return self.current_state, "夜间严格模式"
                    else:
                        self._change_state(SceneState.CAMPAIGN_STRICT, current_time)
                        return self.current_state, "活动严格模式"
            else:
                self.state_duration_counter = 0
        
        elif self.current_state in [SceneState.CAMPAIGN_STRICT, SceneState.NIGHT_STRICT]:
            if scene_score <= self.config.thresholds['campaign_down']:
                self.state_duration_counter += 1
                if self.state_duration_counter >= self.config.time_params['duration_down']:
                    self._change_state(SceneState.NORMAL, current_time)
                    return self.current_state, "回归正常模式"
            else:
                self.state_duration_counter = 0
        
        elif self.current_state == SceneState.EMERGENCY:
            if not self.check_emergency_conditions(anomaly_scores):
                self._change_state(SceneState.NORMAL, current_time)
                return self.current_state, "紧急状态解除"
        
        return self.current_state, "状态保持"
    
    def _evaluate_pv_driven_strategy(self, pv_signals: PVDrivenSignals, current_time: float) -> Tuple[bool, Tuple[SceneState, str]]:
        """评估PV驱动策略切换"""
        config = self.config.pv_driven_config
        
        # 检查紧急PV条件
        f_zc = 1.0 / (1.0 + np.exp(-pv_signals.z_c / 2.0))  # sigmoid变换
        if f_zc >= config['emergency_z_threshold']:
            if self.current_state != SceneState.EMERGENCY:
                self._change_state(SceneState.EMERGENCY, current_time)
                return True, (self.current_state, "PV紧急条件触发")
        
        # 检查漏斗护栏
        if self._check_funnel_guard(pv_signals, current_time):
            return True, (self.current_state, "PV漏斗失配，权重降权")
        
        # PV上行策略切换
        if (pv_signals.theta_up >= config['theta_up_threshold'] and 
            self.current_state != SceneState.EMERGENCY):
            
            if self.current_state == SceneState.NORMAL:
                self._change_state(SceneState.CAMPAIGN_STRICT, current_time)
                return True, (self.current_state, f"PV上行一致率{pv_signals.theta_up:.3f}达到阈值，升级策略")
            elif self.current_state == SceneState.CAMPAIGN_STRICT:
                self._change_state(SceneState.NIGHT_STRICT, current_time)
                return True, (self.current_state, f"PV上行一致率{pv_signals.theta_up:.3f}达到阈值，进一步升级")
        
        # PV下行策略切换
        elif (pv_signals.theta_down >= config['theta_down_threshold'] and 
              self.current_state != SceneState.NORMAL):
            
            if self.current_state in [SceneState.CAMPAIGN_STRICT, SceneState.NIGHT_STRICT]:
                self._change_state(SceneState.NORMAL, current_time)
                return True, (self.current_state, f"PV下行一致率{pv_signals.theta_down:.3f}达到阈值，降级策略")
            elif self.current_state == SceneState.EMERGENCY:
                self._change_state(SceneState.CAMPAIGN_STRICT, current_time)
                return True, (self.current_state, f"PV下行一致率{pv_signals.theta_down:.3f}达到阈值，从紧急状态降级")
        
        return False, (self.current_state, "")
    
    def _check_funnel_guard(self, pv_signals: PVDrivenSignals, current_time: float) -> bool:
        """检查漏斗护栏"""
        config = self.config.pv_driven_config
        
        # 检查r_CA是否持续异常
        if pv_signals.r_ca > config['gamma']:
            # 这里可以加入更复杂的持续性检查逻辑
            return True
        
        return False
    
    def _change_state(self, new_state: SceneState, timestamp: float):
        """改变状态"""
        self.current_state = new_state
        self.last_state_change = timestamp
        self.state_duration_counter = 0
        self.emergency_start_time = None
        self.current_state = new_state


class PolicyEngine:
    """策略引擎"""
    
    def __init__(self):
        self.policies = {
            SceneState.NORMAL: {
                'max_orders_10min': 20,
                'max_single_amount': 50000,
                'max_product_categories': 10,
                'ban_after_violations': 3,
                'ban_duration_hours': 24,
                'require_verification': False,
                'malicious_link_threshold': 'normal'
            },
            SceneState.CAMPAIGN_STRICT: {
                'max_orders_10min': 15,
                'max_single_amount': 50000,
                'max_product_categories': 7,
                'ban_after_violations': 2,
                'ban_duration_hours': 24,
                'require_verification': True,
                'malicious_link_threshold': 'strict',
                'verification_frequency': 'high'
            },
            SceneState.NIGHT_STRICT: {
                'max_orders_10min': 12,
                'max_single_amount': 30000,
                'max_product_categories': 5,
                'ban_after_violations': 2,
                'ban_duration_hours': 48,
                'require_verification': True,
                'malicious_link_threshold': 'strict',
                'login_failure_sensitivity': 'high'
            },
            SceneState.EMERGENCY: {
                'max_orders_10min': 5,
                'max_single_amount': 10000,
                'max_product_categories': 3,
                'ban_after_violations': 1,
                'ban_duration_hours': 72,
                'require_verification': True,
                'suspicious_group_block': True,
                'rate_limit_enabled': True,
                'malicious_link_threshold': 'emergency'
            }
        }
    
    def get_policy_recommendation(self, state: SceneState, 
                                confidence: float) -> str:
        """获取策略建议"""
        policy = self.policies[state]
        
        recommendations = []
        
        if state == SceneState.NORMAL:
            recommendations.append("维持标准风控策略")
        elif state == SceneState.CAMPAIGN_STRICT:
            recommendations.append(f"收紧订单限制至{policy['max_orders_10min']}笔/10分钟")
            recommendations.append(f"商品类目限制{policy['max_product_categories']}个")
            if policy['require_verification']:
                recommendations.append("启用频繁二次验证")
        elif state == SceneState.NIGHT_STRICT:
            recommendations.append(f"夜间模式：订单限制{policy['max_orders_10min']}笔/10分钟")
            recommendations.append(f"单笔金额上限{policy['max_single_amount']}元")
            recommendations.append("提高登录失败敏感度")
        elif state == SceneState.EMERGENCY:
            recommendations.append("启用紧急模式：严格限流")
            recommendations.append("可疑群体灰度拦截")
            recommendations.append("全量二次验证")
            recommendations.append("人工审核面板准备")
        
        return "; ".join(recommendations)


class SceneDetector:
    """场景感知器主类"""
    
    def __init__(self, config: Optional[DetectorConfig] = None, db_path: Optional[str] = None):
        self.config = config or DetectorConfig()
        self.baseline_manager = BaselineManager(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.state_manager = StateManager(self.config)
        self.policy_engine = PolicyEngine()
        
        # 数据库连接
        self.db_path = db_path or '/tmp/scene_detector.db'
        self.init_database()
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # KPI分钟数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kpi_minute (
                ts REAL PRIMARY KEY,
                order_requests INTEGER,
                payment_success INTEGER, 
                product_pv INTEGER,
                risk_hits INTEGER,
                baseline_order REAL,
                baseline_payment REAL,
                baseline_pv REAL,
                baseline_risk REAL,
                z_order REAL,
                z_payment REAL,
                z_pv REAL,
                z_risk REAL,
                scene_score REAL,
                state TEXT,
                confidence REAL
            )
        ''')
        
        # 状态变更日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scene_state_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_state TEXT,
                to_state TEXT,
                ts REAL,
                scene_score REAL,
                z_snapshot TEXT,
                r_snapshot TEXT,
                confidence REAL,
                operator TEXT DEFAULT 'auto'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_scene_score(self, anomaly_scores: Dict[str, AnomalyScore], 
                            kpi_data: KPIData, timestamp: float) -> float:
        """计算场景分数"""
        weights = self.config.weights.copy()
        
        # 夜间权重调整
        if self.state_manager.is_night_time(timestamp):
            for metric, adjust in self.config.night_weight_adjust.items():
                weights[metric] += adjust
        
        # 时间偏差权重调整
        if abs(kpi_data.time_offset) > self.config.time_offset_threshold:
            weights['order_requests'] -= 0.1
            weights['payment_success'] += 0.05
            weights['risk_hits'] += 0.05
        
        # 计算加权分数
        score = (
            weights['order_requests'] * anomaly_scores['order_requests'].normalized_score +
            weights['payment_success'] * anomaly_scores['payment_success'].normalized_score +
            weights['product_pv'] * anomaly_scores['product_pv'].normalized_score +
            weights['risk_hits'] * anomaly_scores['risk_hits'].normalized_score
        )
        
        return min(max(score, 0.0), 1.0)
    
    def calculate_confidence(self, anomaly_scores: Dict[str, AnomalyScore], 
                           duration: float) -> float:
        """计算置信度"""
        # 持续时长映射 (0-3分钟 -> 0-1)
        duration_score = min(duration / (3 * 60), 1.0)
        
        # 信号一致性：多指标同向异常的占比
        anomaly_count = sum(1 for score in anomaly_scores.values() 
                          if score.normalized_score > 0.3)
        consistency_score = anomaly_count / len(anomaly_scores)
        
        # 加权置信度
        confidence = 0.6 * duration_score + 0.4 * consistency_score
        
        return min(max(confidence, 0.0), 1.0)
    
    def process_kpi_data(self, kpi_data: KPIData) -> DetectorResult:
        """处理KPI数据并返回检测结果（集成PV驱动策略）"""
        # 更新历史数据
        self.baseline_manager.update_historical_data(kpi_data)
        
        # 获取基线统计
        baseline_stats = {}
        for metric in ['order_requests', 'payment_success', 'product_pv', 'risk_hits']:
            baseline_stats[metric] = self.baseline_manager.get_baseline_stats(
                kpi_data.timestamp, metric)
        
        # 计算异常分数
        anomaly_scores = self.anomaly_detector.calculate_anomaly_scores(
            kpi_data, baseline_stats)
        
        # 计算PV驱动信号
        pv_signals = self.anomaly_detector.calculate_pv_driven_signals(kpi_data)
        
        # 计算场景分数
        scene_score = self.calculate_scene_score(
            anomaly_scores, kpi_data, kpi_data.timestamp)
        
        # 更新状态（集成PV驱动策略）
        duration = time.time() - self.state_manager.state_start_time
        new_state, reason = self.state_manager.update_state(
            scene_score, anomaly_scores, kpi_data.timestamp, pv_signals)
        
        # 计算置信度
        confidence = self.calculate_confidence(anomaly_scores, duration)
        
        # 生成策略建议
        policy_recommendation = self.policy_engine.get_policy_recommendation(
            new_state, confidence)
        
        # 构建原因快照（加入PV驱动信息）
        reason_snapshot = {
            'trigger_reason': reason,
            'baseline_stats': {k: asdict(v) for k, v in baseline_stats.items()},
            'kpi_values': asdict(kpi_data),
            'scene_score': scene_score,
            'is_night': self.state_manager.is_night_time(kpi_data.timestamp),
            'pv_driven_signals': asdict(pv_signals) if pv_signals else None
        }
        
        result = DetectorResult(
            timestamp=kpi_data.timestamp,
            scene_score=scene_score,
            state=new_state,
            confidence=confidence,
            reason_snapshot=reason_snapshot,
            policy_recommendation=policy_recommendation,
            anomaly_scores=anomaly_scores
        )
        
        # 记录决策日志
        self.logger.info(f"PV驱动策略决策 - 状态: {new_state.value}, 原因: {reason}")
        
        return result
    
    def get_current_policy(self) -> Dict[str, Any]:
        """获取当前策略配置"""
        return self.policy_engine.policies[self.state_manager.current_state]
    
    def manual_state_override(self, target_state: SceneState, operator: str = "manual"):
        """人工状态覆盖"""
        old_state = self.state_manager.current_state
        self.state_manager._change_state(target_state, time.time())
        
        # 记录人工操作日志
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO scene_state_log (
                from_state, to_state, ts, scene_score, z_snapshot, r_snapshot, confidence, operator
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            old_state.value,
            target_state.value,
            time.time(),
            0.0,
            "{}",
            "{}",
            1.0,
            operator
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"人工状态覆盖: {old_state.value} -> {target_state.value}, 操作员: {operator}")
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (hours * 3600)
        
        # 状态分布统计
        cursor.execute('''
            SELECT state, COUNT(*) as count
            FROM kpi_minute 
            WHERE ts > ?
            GROUP BY state
        ''', (cutoff_time,))
        
        state_distribution = dict(cursor.fetchall())
        
        # 平均场景分数
        cursor.execute('''
            SELECT AVG(scene_score) as avg_score, MAX(scene_score) as max_score
            FROM kpi_minute 
            WHERE ts > ?
        ''', (cutoff_time,))
        
        score_stats = cursor.fetchone()
        
        # 状态切换次数
        cursor.execute('''
            SELECT COUNT(*) as changes
            FROM scene_state_log 
            WHERE ts > ?
        ''', (cutoff_time,))
        
        state_changes = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'time_range_hours': hours,
            'state_distribution': state_distribution,
            'avg_scene_score': score_stats[0] if score_stats[0] else 0.0,
            'max_scene_score': score_stats[1] if score_stats[1] else 0.0,
            'state_changes': state_changes,
            'current_state': self.state_manager.current_state.value
        }


def create_sample_kpi_data(timestamp: Optional[float] = None) -> KPIData:
    """创建示例KPI数据（用于测试）"""
    if timestamp is None:
        timestamp = time.time()
    
    import random
    
    # 模拟正常业务数据
    base_orders = 15
    base_payments = 12
    base_pv = 500
    base_risk = 2
    
    # 添加一些随机波动
    return KPIData(
        timestamp=timestamp,
        order_requests=max(0, int(base_orders + random.normalvariate(0, 3))),
        payment_success=max(0, int(base_payments + random.normalvariate(0, 2))),
        product_pv=max(0, int(base_pv + random.normalvariate(0, 50))),
        risk_hits=max(0, int(base_risk + random.normalvariate(0, 1))),
        time_offset=random.normalvariate(0, 100),
        source_entropy=random.uniform(0.5, 0.9),
        ip_entropy=random.uniform(0.6, 0.95),
        # PV驱动策略新增字段
        detail_pv=max(0, int(base_pv * 0.8 + random.normalvariate(0, 40))),
        is_bot_like=random.choice([True, False]) if random.random() < 0.1 else False,
        quality_flag=random.choice([False]) if random.random() < 0.05 else True,
        prerender_flag=random.choice([True, False]) if random.random() < 0.03 else False
    )


def create_anomaly_kpi_data(timestamp: Optional[float] = None) -> KPIData:
    """创建异常KPI数据（用于测试）"""
    if timestamp is None:
        timestamp = time.time()
    
    # 模拟异常情况：订单暴增、支付异常、风控频繁触发
    return KPIData(
        timestamp=timestamp,
        order_requests=50,  # 异常高
        payment_success=5,   # 异常低
        product_pv=2000,    # 异常高（可能是爬虫）
        risk_hits=15,       # 异常高
        time_offset=500,    # 时间偏差大
        source_entropy=0.2, # 来源集中（可疑）
        ip_entropy=0.3,     # IP集中（可疑）
        # PV驱动策略新增字段
        detail_pv=1500,     # 详情PV异常高
        is_bot_like=True,   # 标记为机器人流量
        quality_flag=False, # 质量标识异常
        prerender_flag=True # 预渲染标识
    )


def main():
    """主函数 - 演示感知器使用（集成PV驱动策略切换）"""
    print("场景感知器（Detector）算法 v2.0 启动...（集成PV驱动策略切换）")
    
    # 创建感知器实例
    detector = SceneDetector()
    
    print(f"初始状态: {detector.state_manager.current_state.value}")
    print(f"PV驱动策略配置: {detector.config.pv_driven_config}")
    print("="*60)
    
    # 模拟一段时间的正常数据
    print("模拟正常业务数据...")
    for i in range(3):
        kpi_data = create_sample_kpi_data()
        result = detector.process_kpi_data(kpi_data)
        
        print(f"时间: {datetime.fromtimestamp(kpi_data.timestamp).strftime('%H:%M:%S')}")
        print(f"订单: {kpi_data.order_requests}, 支付: {kpi_data.payment_success}, PV: {kpi_data.product_pv}, 详情PV: {kpi_data.detail_pv}, 风控: {kpi_data.risk_hits}")
        print(f"机器人标识: {kpi_data.is_bot_like}, 质量标识: {kpi_data.quality_flag}")
        print(f"场景分数: {result.scene_score:.3f}, 状态: {result.state.value}, 置信度: {result.confidence:.3f}")
        
        # 显示PV驱动信号
        if result.reason_snapshot.get('pv_driven_signals'):
            pv_signals = result.reason_snapshot['pv_driven_signals']
            print(f"PV驱动信号: z_c={pv_signals['z_c']:.3f}, slope_c={pv_signals['slope_c']:.3f}, r_ca={pv_signals['r_ca']:.3f}")
            print(f"Tick信号: 上行={pv_signals['up_tick']}, 下行={pv_signals['down_tick']}, θ↑={pv_signals['theta_up']:.3f}, θ↓={pv_signals['theta_down']:.3f}")
        
        print(f"策略建议: {result.policy_recommendation}")
        print("-"*40)
        
        time.sleep(1)
    
    # 模拟PV异常数据触发策略切换
    print("\n模拟PV异常数据（触发PV驱动策略切换）...")
    for i in range(4):
        # 创建异常PV数据
        kpi_data = create_anomaly_kpi_data()
        # 调整为非机器人流量以触发PV策略
        kpi_data.is_bot_like = False
        kpi_data.quality_flag = True
        kpi_data.product_pv = 1500 + i * 300  # PV逐步增长
        kpi_data.detail_pv = int(kpi_data.product_pv * 0.7)
        
        result = detector.process_kpi_data(kpi_data)
        
        print(f"时间: {datetime.fromtimestamp(kpi_data.timestamp).strftime('%H:%M:%S')}")
        print(f"订单: {kpi_data.order_requests}, 支付: {kpi_data.payment_success}, PV: {kpi_data.product_pv}, 详情PV: {kpi_data.detail_pv}, 风控: {kpi_data.risk_hits}")
        print(f"场景分数: {result.scene_score:.3f}, 状态: {result.state.value}, 置信度: {result.confidence:.3f}")
        
        # 显示PV驱动信号详情
        if result.reason_snapshot.get('pv_driven_signals'):
            pv_signals = result.reason_snapshot['pv_driven_signals']
            print(f"PV驱动信号: z_c={pv_signals['z_c']:.3f}, slope_c={pv_signals['slope_c']:.3f}, r_ca={pv_signals['r_ca']:.3f}")
            print(f"Tick信号: 上行={pv_signals['up_tick']}, 下行={pv_signals['down_tick']}, θ↑={pv_signals['theta_up']:.3f}, θ↓={pv_signals['theta_down']:.3f}")
        
        print(f"决策原因: {result.reason_snapshot['trigger_reason']}")
        print(f"策略建议: {result.policy_recommendation}")
        
        # 显示异常分数详情
        print(f"异常分数详情:")
        for metric, score in result.anomaly_scores.items():
            print(f"  {metric}: z={score.z_score:.2f}, ratio={score.ratio:.2f}, norm={score.normalized_score:.3f}")
        print("-"*40)
        
        time.sleep(1)
    
    # 显示统计信息
    print("\n统计信息:")
    stats = detector.get_statistics(hours=1)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✨ PV驱动策略切换功能特点:")
    print("✅ PV异常强度检测 (z分数)")
    print("✅ PV趋势斜率分析 (5分钟线性回归)")
    print("✅ 漏斗一致性验证 (PV→订单转化)")
    print("✅ 窗口抖动测试 (150秒滑动窗口)")
    print("✅ 机器人流量过滤")
    print("✅ 紧急越权机制")
    print("✅ 漏斗护栏机制")
    
    print("\n🎉 集成PV驱动策略切换的感知器演示完成！")


if __name__ == "__main__":
    main()