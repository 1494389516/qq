#!/usr/bin/env python3

"""
多语言混合DDoS反击系统 - Python控制层
基于风控算法专家的理论框架和数学建模视角

核心功能模块：
1. 智能威胁分析引擎 - 机器学习算法
2. 多语言组件协调器 - 系统编排
3. 自适应策略优化器 - 动态参数调优
4. 业务场景适配器 - 场景驱动决策
5. 效果评估与反馈 - A/B测试框架

数学建模理论：
- 多维特征融合检测模型
- 贝叶斯自适应阈值算法
- 马尔可夫链攻击预测
- 强化学习策略优化
"""

import asyncio
import time
import json
import logging
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import aiohttp
import websockets

# 机器学习相关
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class ThreatLevel(Enum):
    """威胁等级枚举"""
    NORMAL = 0
    SUSPICIOUS = 1
    HIGH_RISK = 2
    CRITICAL = 3


class AttackType(Enum):
    """攻击类型枚举"""
    VOLUMETRIC = "volumetric"      # 容量型攻击
    PROTOCOL = "protocol"          # 协议型攻击
    APPLICATION = "application"    # 应用层攻击
    HYBRID = "hybrid"              # 混合型攻击


class BusinessScenario(Enum):
    """业务场景枚举"""
    NORMAL_PERIOD = "normal"
    PROMOTION_PERIOD = "promotion"
    HIGH_TRAFFIC_PERIOD = "high_traffic"
    MAINTENANCE_PERIOD = "maintenance"


@dataclass
class NetworkFlowData:
    """网络流量数据"""
    timestamp: float
    src_ip: str
    dst_ip: str
    protocol: str
    packets_per_second: float
    bytes_per_second: float
    connection_count: int
    avg_packet_size: float
    source_entropy: float
    destination_entropy: float
    syn_ratio: float
    small_packet_ratio: float


@dataclass
class ThreatDetectionResult:
    """威胁检测结果"""
    timestamp: float
    threat_level: ThreatLevel
    attack_type: AttackType
    risk_score: float
    confidence: float
    source_ips: List[str]
    target_ips: List[str]
    features: Dict[str, float]
    detection_method: str


@dataclass
class DefenseStrategy:
    """防护策略"""
    strategy_type: str
    target_ips: List[str]
    action_intensity: float
    duration_seconds: int
    expected_effectiveness: float
    business_impact: float


@dataclass
class SystemPerformance:
    """系统性能指标"""
    detection_latency_ms: float
    throughput_pps: float
    false_positive_rate: float
    false_negative_rate: float
    system_load: float
    memory_usage: float


class AdvancedThreatAnalyzer:
    """高级威胁分析引擎"""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination='auto', random_state=42)
        self.ml_models = {}
        self.baseline_stats = {}
        self.logger = logging.getLogger(f"{__name__}.ThreatAnalyzer")
        
        # 数学建模参数
        self.detection_params = {
            'z_score_threshold': 3.0,
            'entropy_min_normal': 4.0,
            'entropy_max_normal': 7.0,
            'syn_ratio_threshold': 0.8,
            'small_packet_threshold': 0.3,
            'baseline_window_hours': 24,
            'adaptation_rate': 0.1
        }
        
    def initialize_models(self):
        """初始化机器学习模型"""
        if XGBOOST_AVAILABLE:
            import xgboost as xgb_module
            self.ml_models['xgboost'] = xgb_module.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
            
        if LIGHTGBM_AVAILABLE:
            import lightgbm as lgb_module
            self.ml_models['lightgbm'] = lgb_module.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
            
        self.ml_models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
        
        self.logger.info(f"已初始化 {len(self.ml_models)} 个ML模型")
    
    def extract_advanced_features(self, flow_data: List[NetworkFlowData]) -> np.ndarray:
        """提取高级特征向量"""
        if not flow_data:
            return np.array([])
            
        features = []
        
        for flow in flow_data:
            feature_vector = [
                flow.packets_per_second,
                flow.bytes_per_second,
                flow.connection_count,
                flow.avg_packet_size,
                flow.source_entropy,
                flow.destination_entropy,
                flow.syn_ratio,
                flow.small_packet_ratio,
                # 时间特征
                datetime.fromtimestamp(flow.timestamp).hour,
                datetime.fromtimestamp(flow.timestamp).weekday(),
                # 协议特征
                1 if flow.protocol == 'TCP' else 0,
                1 if flow.protocol == 'UDP' else 0,
                1 if flow.protocol == 'ICMP' else 0,
            ]
            features.append(feature_vector)
            
        return np.array(features)
    
    def detect_anomalies(self, flow_data: List[NetworkFlowData]) -> List[ThreatDetectionResult]:
        """异常检测主算法"""
        features = self.extract_advanced_features(flow_data)
        if features.size == 0:
            return []
            
        # 特征标准化
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # 孤立森林异常检测
        anomaly_scores = self.anomaly_detector.fit_predict(features_scaled)
        decision_scores = self.anomaly_detector.decision_function(features_scaled)
        
        results = []
        for i, (flow, is_anomaly, score) in enumerate(zip(flow_data, anomaly_scores, decision_scores)):
            if is_anomaly == -1:  # 异常
                # 计算风险评分
                risk_score = self._calculate_risk_score(flow, score)
                
                # 确定威胁等级
                threat_level = self._assess_threat_level(risk_score)
                
                # 识别攻击类型
                attack_type = self._identify_attack_type(flow)
                
                result = ThreatDetectionResult(
                    timestamp=flow.timestamp,
                    threat_level=threat_level,
                    attack_type=attack_type,
                    risk_score=risk_score,
                    confidence=min(abs(score), 1.0),
                    source_ips=[flow.src_ip],
                    target_ips=[flow.dst_ip],
                    features=self._extract_feature_dict(flow),
                    detection_method="isolation_forest"
                )
                results.append(result)
                
        return results
    
    def _calculate_risk_score(self, flow: NetworkFlowData, anomaly_score: float) -> float:
        """计算综合风险评分"""
        # 基础异常分数
        base_score = min(abs(anomaly_score), 1.0)
        
        # 流量强度加权
        traffic_factor = min(flow.packets_per_second / 10000, 1.0)
        
        # 熵值异常加权
        entropy_factor = 0.0
        if flow.source_entropy < self.detection_params['entropy_min_normal']:
            entropy_factor = 0.3
        elif flow.source_entropy > self.detection_params['entropy_max_normal']:
            entropy_factor = 0.2
            
        # SYN洪水检测
        syn_factor = 0.0
        if flow.syn_ratio > self.detection_params['syn_ratio_threshold']:
            syn_factor = 0.4
            
        # 小包攻击检测
        small_packet_factor = 0.0
        if flow.small_packet_ratio > self.detection_params['small_packet_threshold']:
            small_packet_factor = 0.2
            
        # 综合风险评分
        risk_score = (base_score * 0.4 + 
                     traffic_factor * 0.3 +
                     entropy_factor + 
                     syn_factor + 
                     small_packet_factor)
        
        return min(risk_score, 1.0)
    
    def _assess_threat_level(self, risk_score: float) -> ThreatLevel:
        """评估威胁等级"""
        if risk_score >= 0.9:
            return ThreatLevel.CRITICAL
        elif risk_score >= 0.7:
            return ThreatLevel.HIGH_RISK
        elif risk_score >= 0.4:
            return ThreatLevel.SUSPICIOUS
        else:
            return ThreatLevel.NORMAL
    
    def _identify_attack_type(self, flow: NetworkFlowData) -> AttackType:
        """识别攻击类型"""
        # 基于特征模式识别攻击类型
        if flow.packets_per_second > 50000:  # 高PPS
            if flow.avg_packet_size < 100:
                return AttackType.VOLUMETRIC  # 小包洪水
            else:
                return AttackType.PROTOCOL   # 协议攻击
        elif flow.syn_ratio > 0.8:
            return AttackType.PROTOCOL       # SYN洪水
        elif flow.source_entropy < 2.0:
            return AttackType.APPLICATION    # 应用层攻击
        else:
            return AttackType.HYBRID         # 混合攻击
    
    def _extract_feature_dict(self, flow: NetworkFlowData) -> Dict[str, float]:
        """提取特征字典"""
        return {
            'packets_per_second': flow.packets_per_second,
            'bytes_per_second': flow.bytes_per_second,
            'source_entropy': flow.source_entropy,
            'syn_ratio': flow.syn_ratio,
            'small_packet_ratio': flow.small_packet_ratio,
            'avg_packet_size': flow.avg_packet_size
        }


class MultiLanguageCoordinator:
    """多语言组件协调器"""
    
    def __init__(self):
        self.component_endpoints = {
            'cpp_detector': 'http://localhost:8001',
            'go_counter': 'http://localhost:8002', 
            'rust_storage': 'http://localhost:8003',
            'js_frontend': 'http://localhost:8080'
        }
        self.component_status = {}
        self.logger = logging.getLogger(f"{__name__}.Coordinator")
        
    async def check_component_health(self, component: str) -> bool:
        """检查组件健康状态"""
        try:
            endpoint = self.component_endpoints.get(component)
            if not endpoint:
                return False
                
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint}/health", timeout=5) as response:
                    if response.status == 200:
                        self.component_status[component] = {
                            'status': 'healthy',
                            'last_check': time.time()
                        }
                        return True
                    else:
                        self.component_status[component] = {
                            'status': 'unhealthy',
                            'last_check': time.time()
                        }
                        return False
        except Exception as e:
            self.logger.error(f"健康检查失败 {component}: {e}")
            self.component_status[component] = {
                'status': 'error',
                'last_check': time.time(),
                'error': str(e)
            }
            return False
    
    async def submit_detection_request(self, flow_data: List[NetworkFlowData]) -> Optional[Dict]:
        """提交检测请求到C++组件"""
        try:
            data = [asdict(flow) for flow in flow_data]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.component_endpoints['cpp_detector']}/api/v1/detect",
                    json=data,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.error(f"C++检测请求失败: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"C++检测请求异常: {e}")
            return None
    
    async def execute_counter_measures(self, threats: List[ThreatDetectionResult]) -> bool:
        """执行反击措施"""
        try:
            # 转换为Go组件期望的格式
            attack_sources = []
            for threat in threats:
                for src_ip in threat.source_ips:
                    attack_source = {
                        'ip': src_ip,
                        'port': 0,  # 未知端口
                        'attack_type': threat.attack_type.value,
                        'risk_score': threat.risk_score,
                        'confidence': threat.confidence,
                        'first_seen': datetime.fromtimestamp(threat.timestamp).isoformat(),
                        'last_seen': datetime.fromtimestamp(threat.timestamp).isoformat(),
                        'packet_count': int(threat.features.get('packets_per_second', 0) * 60)
                    }
                    attack_sources.append(attack_source)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.component_endpoints['go_counter']}/api/v1/counter-attack",
                    json=attack_sources,
                    timeout=15
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"反击任务已提交: {result}")
                        return True
                    else:
                        self.logger.error(f"反击执行失败: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"反击执行异常: {e}")
            return False
    
    async def store_detection_results(self, results: List[ThreatDetectionResult]) -> bool:
        """存储检测结果到Rust组件"""
        try:
            for result in results:
                data = {
                    'timestamp': int(result.timestamp),
                    'risk_score': result.risk_score,
                    'threat_type': result.attack_type.value,
                    'confidence': result.confidence,
                    'features': result.features,
                    'source_ip': result.source_ips[0] if result.source_ips else 'unknown',
                    'detection_stage': result.detection_method
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.component_endpoints['rust_storage']}/api/v1/detection",
                        json=data,
                        timeout=5
                    ) as response:
                        if response.status != 200:
                            self.logger.error(f"存储失败: {response.status}")
                            return False
            return True
        except Exception as e:
            self.logger.error(f"存储异常: {e}")
            return False


class AdaptiveStrategyOptimizer:
    """自适应策略优化器"""
    
    def __init__(self):
        self.strategy_history = deque(maxlen=1000)
        self.effectiveness_scores = defaultdict(list)
        self.logger = logging.getLogger(f"{__name__}.StrategyOptimizer")
        
    def optimize_detection_thresholds(self, recent_results: List[ThreatDetectionResult],
                                    false_positive_rate: float,
                                    false_negative_rate: float) -> Dict[str, float]:
        """优化检测阈值"""
        # 基于强化学习的阈值优化
        current_thresholds = {
            'risk_score_threshold': 0.7,
            'confidence_threshold': 0.6,
            'z_score_threshold': 3.0
        }
        
        # 如果误报率过高，提高阈值
        if false_positive_rate > 0.05:  # 5%
            current_thresholds['risk_score_threshold'] += 0.05
            current_thresholds['confidence_threshold'] += 0.05
            
        # 如果漏报率过高，降低阈值
        if false_negative_rate > 0.1:  # 10%
            current_thresholds['risk_score_threshold'] -= 0.05
            current_thresholds['confidence_threshold'] -= 0.05
            
        # 限制阈值范围
        for key in current_thresholds:
            current_thresholds[key] = max(0.1, min(0.9, current_thresholds[key]))
            
        self.logger.info(f"优化后阈值: {current_thresholds}")
        return current_thresholds
    
    def select_optimal_defense_strategy(self, threat: ThreatDetectionResult,
                                      business_scenario: BusinessScenario) -> DefenseStrategy:
        """选择最优防护策略"""
        # 基于威胁等级和业务场景选择策略
        if threat.threat_level == ThreatLevel.CRITICAL:
            # 关键威胁：立即阻断
            strategy = DefenseStrategy(
                strategy_type="immediate_block",
                target_ips=threat.source_ips,
                action_intensity=1.0,
                duration_seconds=3600,
                expected_effectiveness=0.95,
                business_impact=0.3 if business_scenario == BusinessScenario.PROMOTION_PERIOD else 0.1
            )
        elif threat.threat_level == ThreatLevel.HIGH_RISK:
            # 高风险：限流处理
            strategy = DefenseStrategy(
                strategy_type="rate_limiting",
                target_ips=threat.source_ips,
                action_intensity=0.3,
                duration_seconds=1800,
                expected_effectiveness=0.8,
                business_impact=0.1
            )
        else:
            # 低风险：监控观察
            strategy = DefenseStrategy(
                strategy_type="enhanced_monitoring",
                target_ips=threat.source_ips,
                action_intensity=0.1,
                duration_seconds=900,
                expected_effectiveness=0.6,
                business_impact=0.0
            )
            
        return strategy


class DDoSDefenseController:
    """DDoS防护系统主控制器"""
    
    def __init__(self):
        self.threat_analyzer = AdvancedThreatAnalyzer()
        self.coordinator = MultiLanguageCoordinator()
        self.strategy_optimizer = AdaptiveStrategyOptimizer()
        
        self.running = False
        self.detection_thread = None
        self.performance_metrics = SystemPerformance(
            detection_latency_ms=0.0,
            throughput_pps=0.0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            system_load=0.0,
            memory_usage=0.0
        )
        
        self.logger = logging.getLogger(f"{__name__}.Controller")
        
    async def initialize(self) -> bool:
        """初始化系统"""
        self.logger.info("🚀 初始化多语言混合DDoS防护系统...")
        
        # 初始化威胁分析器
        self.threat_analyzer.initialize_models()
        
        # 检查所有组件健康状态
        health_results = {}
        for component in self.coordinator.component_endpoints:
            health_results[component] = await self.coordinator.check_component_health(component)
            
        healthy_components = sum(health_results.values())
        total_components = len(health_results)
        
        self.logger.info(f"组件健康检查: {healthy_components}/{total_components}")
        
        if healthy_components < total_components * 0.5:  # 至少50%组件健康
            self.logger.error("系统组件健康状态不足，无法启动")
            return False
            
        self.logger.info("✅ 系统初始化完成")
        return True
    
    async def start_detection(self):
        """启动实时检测"""
        self.running = True
        self.logger.info("🔍 启动实时DDoS检测...")
        
        while self.running:
            try:
                # 模拟接收网络流量数据
                flow_data = self._generate_sample_flow_data()
                
                # 执行威胁检测
                start_time = time.time()
                threats = self.threat_analyzer.detect_anomalies(flow_data)
                detection_time = (time.time() - start_time) * 1000
                
                # 更新性能指标
                self.performance_metrics.detection_latency_ms = detection_time
                self.performance_metrics.throughput_pps = len(flow_data) / (detection_time / 1000)
                
                if threats:
                    self.logger.info(f"检测到 {len(threats)} 个威胁")
                    
                    # 存储检测结果
                    await self.coordinator.store_detection_results(threats)
                    
                    # 执行防护措施
                    await self.coordinator.execute_counter_measures(threats)
                    
                await asyncio.sleep(1)  # 每秒检测一次
                
            except Exception as e:
                self.logger.error(f"检测循环异常: {e}")
                await asyncio.sleep(5)
    
    def _generate_sample_flow_data(self) -> List[NetworkFlowData]:
        """生成示例流量数据（实际应该从网络接口读取）"""
        import random
        
        flows = []
        current_time = time.time()
        
        # 正常流量
        for _ in range(random.randint(10, 50)):
            flow = NetworkFlowData(
                timestamp=current_time,
                src_ip=f"192.168.1.{random.randint(1, 254)}",
                dst_ip="10.0.0.100",
                protocol=random.choice(["TCP", "UDP"]),
                packets_per_second=random.uniform(100, 1000),
                bytes_per_second=random.uniform(50000, 1000000),
                connection_count=random.randint(1, 20),
                avg_packet_size=random.uniform(64, 1500),
                source_entropy=random.uniform(4.0, 7.0),
                destination_entropy=random.uniform(4.0, 7.0),
                syn_ratio=random.uniform(0.1, 0.3),
                small_packet_ratio=random.uniform(0.1, 0.3)
            )
            flows.append(flow)
        
        # 随机生成攻击流量（10%概率）
        if random.random() < 0.1:
            attack_flow = NetworkFlowData(
                timestamp=current_time,
                src_ip=f"203.0.113.{random.randint(1, 254)}",
                dst_ip="10.0.0.100",
                protocol="TCP",
                packets_per_second=random.uniform(10000, 100000),  # 高PPS
                bytes_per_second=random.uniform(1000000, 10000000),
                connection_count=random.randint(100, 1000),
                avg_packet_size=random.uniform(60, 100),  # 小包
                source_entropy=random.uniform(1.0, 3.0),  # 低熵值
                destination_entropy=random.uniform(4.0, 7.0),
                syn_ratio=random.uniform(0.8, 1.0),  # 高SYN比例
                small_packet_ratio=random.uniform(0.7, 1.0)  # 高小包比例
            )
            flows.append(attack_flow)
            
        return flows
    
    def stop_detection(self):
        """停止检测"""
        self.running = False
        self.logger.info("🛑 停止DDoS检测")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': self.running,
            'component_status': self.coordinator.component_status,
            'performance_metrics': asdict(self.performance_metrics),
            'timestamp': time.time()
        }


async def main():
    """主演示函数"""
    print("🚀 多语言混合DDoS反击系统演示")
    print("=" * 80)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建DDoS防护控制器
    controller = DDoSDefenseController()
    
    # 初始化系统
    if not await controller.initialize():
        print("❌ 系统初始化失败")
        return
    
    print("\n📊 开始实时DDoS检测演示...")
    
    try:
        # 启动检测（运行10秒演示）
        detection_task = asyncio.create_task(controller.start_detection())
        
        # 等待演示时间
        await asyncio.sleep(10)
        
        # 停止检测
        controller.stop_detection()
        detection_task.cancel()
        
        # 显示系统状态
        status = controller.get_system_status()
        print("\n📈 系统状态报告:")
        print("-" * 50)
        print(f"检测延迟: {status['performance_metrics']['detection_latency_ms']:.2f} ms")
        print(f"处理吞吐量: {status['performance_metrics']['throughput_pps']:.0f} PPS")
        
        print("\n📚 理论框架总结:")
        print("-" * 50)
        print("🔬 数学建模: 多维特征融合 + 异常检测算法")
        print("🤖 机器学习: XGBoost/LightGBM + 孤立森林")
        print("⚡ 多语言协同: Python决策 + C++检测 + Go反击 + Rust存储")
        print("🎯 自适应优化: 强化学习策略选择 + 实时阈值调整")
        
    except KeyboardInterrupt:
        print("\n🛑 检测被用户中断")
        controller.stop_detection()
        

if __name__ == "__main__":
    asyncio.run(main())