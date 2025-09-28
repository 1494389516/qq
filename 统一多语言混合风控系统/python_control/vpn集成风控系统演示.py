
"""
集成VPN检测的多层次风控系统演示
基于风控算法专家的理论框架和数学建模视角

核心集成算法理论：
1. 四阶段级联VPN检测算法
2. 业务场景驱动的智能模型选择
3. 多层次风险融合理论
4. 自适应威胁响应机制
"""

import time
import numpy as np
import pandas as pd
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class AttackType(Enum):
    """攻击类型枚举"""
    NORMAL = 0
    CRAWLER = 1
    BRUTE_FORCE = 2
    ORDER_FRAUD = 3
    PAYMENT_FRAUD = 4
    DDOS = 5
    VPN_TUNNEL = 6
    NETWORK_ANOMALY = 7

class VPNType(Enum):
    """VPN类型枚举"""
    OPENVPN = "OpenVPN"
    IPSEC = "IPSec"
    WIREGUARD = "WireGuard"
    PPTP = "PPTP"
    L2TP = "L2TP"
    SSTP = "SSTP"
    UNKNOWN = "Unknown"

class BusinessScenarioType(Enum):
    """业务场景类型枚举"""
    NORMAL_PERIOD = "normal_period"      # 平时期
    CAMPAIGN_PERIOD = "campaign_period"  # 活动期
    HIGH_RISK_PERIOD = "high_risk_period" # 高风险期
    MAINTENANCE_PERIOD = "maintenance_period" # 维护期

@dataclass
class NetworkPacket:
    """网络数据包结构"""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    size: int
    direction: str  # 'up' or 'down'
    payload_size: int
    tls_info: Optional[Dict] = None

@dataclass
class NetworkDetectionResult:
    """网络检测结果"""
    flow_id: str
    is_threat: bool
    threat_type: str
    confidence: float
    detection_stage: str
    features: Dict[str, Any]
    timestamp: float

@dataclass
class BusinessContext:
    """业务上下文信息"""
    scenario_type: BusinessScenarioType
    user_experience_priority: float  # 用户体验优先级 [0,1]
    accuracy_requirement: float      # 精度要求 [0,1] 
    latency_requirement: float       # 延迟要求 ms
    resource_constraint: float       # 资源约束 [0,1]
    traffic_volume_multiplier: float # 流量倍数
    complex_feature_enabled: bool    # 是否启用复杂特征工程

@dataclass
class ModelPerformanceMetrics:
    """模型性能指标"""
    model_name: str
    accuracy: float
    f1_score: float
    training_time: float
    scenario_suitability: float


class NetworkFlowAnalyzer:
    """网络流量分析器 - 集成VPN检测算法"""
    
    def __init__(self):
        self.ike_esp_ports = [500, 4500]  # IPsec IKE/ESP
        self.openvpn_ports = [1194]       # OpenVPN
        self.wireguard_ports = [51820]    # WireGuard
        self.ddos_threshold = 1000        # DDoS检测阈值
        self.logger = logging.getLogger(f"{__name__}.NetworkFlow")
        
    def generate_network_packets(self, attack_type: AttackType, count: int = 100) -> List[NetworkPacket]:
        """生成网络数据包 - 模拟不同的攻击类型"""
        packets = []
        base_time = time.time()
        
        np.random.seed(42 + attack_type.value)
        
        for i in range(count):
            if attack_type == AttackType.VPN_TUNNEL:
                # VPN隧道流量特征
                src_port = np.random.choice(self.openvpn_ports + self.ike_esp_ports)
                dst_port = np.random.choice(self.openvpn_ports + self.ike_esp_ports)
                size = int(np.random.normal(800, 100))  # 更规律的包大小
                direction = np.random.choice(['up', 'down'], p=[0.45, 0.55])
                time_interval = np.random.exponential(0.1)  # 更规律的时间间隔
                protocol = "UDP"
                
            elif attack_type == AttackType.DDOS:
                # DDoS攻击特征
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.choice([80, 443, 53])
                size = int(np.random.choice([64, 128, 256]))  # 小包攻击
                direction = 'up'  # 主要是上行流量
                time_interval = np.random.exponential(0.01)  # 非常快的请求
                protocol = np.random.choice(["TCP", "UDP"])
                
            elif attack_type == AttackType.NETWORK_ANOMALY:
                # 网络异常流量
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.randint(1024, 65535)  # 随机端口
                size = int(np.random.uniform(100, 1500))  # 不规则包大小
                direction = np.random.choice(['up', 'down'])
                time_interval = np.random.uniform(0.001, 1.0)  # 不规则时间间隔
                protocol = np.random.choice(["TCP", "UDP", "ICMP"])
                
            else:  # NORMAL
                # 正常流量特征
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.choice([80, 443, 53, 8080])
                size = int(np.random.choice([64, 128, 256, 512, 1024, 1500]))
                direction = np.random.choice(['up', 'down'])
                time_interval = np.random.exponential(np.random.uniform(0.05, 0.5))
                protocol = np.random.choice(["TCP", "UDP"])
                
            packet = NetworkPacket(
                timestamp=base_time + i * time_interval,
                src_ip=f"192.168.1.{np.random.randint(1, 254)}",
                dst_ip=f"8.8.8.{np.random.randint(1, 254)}",
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                size=max(64, min(1500, size)),
                direction=direction,
                payload_size=max(20, size - 40)
            )
            packets.append(packet)
            
        return packets
    
    def detect_vpn_tunnel(self, packets: List[NetworkPacket]) -> NetworkDetectionResult:
        """四阶段级联VPN检测算法"""
        if not packets:
            return NetworkDetectionResult(
                flow_id="empty_flow",
                is_threat=False,
                threat_type="NORMAL",
                confidence=0.9,
                detection_stage="EmptyFlow",
                features={},
                timestamp=time.time()
            )
            
        flow_id = f"{packets[0].src_ip}:{packets[0].dst_ip}"
        
        # 阶段A: 规则预筛
        protocol_indicators = self._check_protocol_indicators(packets)
        if not any(protocol_indicators.values()):
            return NetworkDetectionResult(
                flow_id=flow_id,
                is_threat=False,
                threat_type="NORMAL",
                confidence=0.95,
                detection_stage="RulePreFilter",
                features=protocol_indicators,
                timestamp=time.time()
            )
        
        # 提取网络特征
        features = self._extract_network_features(packets)
        
        # 阶段B: 相对熵过滤
        if not self._relative_entropy_filter(features):
            return NetworkDetectionResult(
                flow_id=flow_id,
                is_threat=False,
                threat_type="NORMAL",
                confidence=0.8,
                detection_stage="RelativeEntropyFilter",
                features=features,
                timestamp=time.time()
            )
        
        # 阶段C: 序列模型精判
        sequence_score = self._cnn_lstm_predict(packets)
        
        # 阶段D: 多窗融合（简化版）
        is_vpn = sequence_score > 0.6
        confidence = sequence_score if is_vpn else 1.0 - sequence_score
        threat_type = "VPN_TUNNEL" if is_vpn else "NORMAL"
        
        return NetworkDetectionResult(
            flow_id=flow_id,
            is_threat=is_vpn,
            threat_type=threat_type,
            confidence=confidence,
            detection_stage="SequenceModel",
            features=features,
            timestamp=time.time()
        )
    
    def detect_ddos_attack(self, packets: List[NetworkPacket]) -> NetworkDetectionResult:
        """DDoS攻击检测"""
        if not packets:
            return NetworkDetectionResult(
                flow_id="empty_flow",
                is_threat=False,
                threat_type="NORMAL",
                confidence=0.9,
                detection_stage="EmptyFlow",
                features={},
                timestamp=time.time()
            )
            
        flow_id = f"{packets[0].src_ip}:{packets[0].dst_ip}"
        features = self._extract_network_features(packets)
        
        # DDoS特征检测
        pps = features.get('packets_per_second', 0)
        small_packet_ratio = sum(1 for p in packets if p.size < 200) / len(packets)
        up_ratio = features.get('direction_ratio', 0.5)
        
        # DDoS评分算法
        ddos_score = 0.0
        
        # 高PPS的可疑性
        if pps > 100:
            ddos_score += 0.4
        elif pps > 50:
            ddos_score += 0.2
            
        # 小包攻击模式
        if small_packet_ratio > 0.8:
            ddos_score += 0.3
        elif small_packet_ratio > 0.6:
            ddos_score += 0.2
            
        # 上行流量主导
        if up_ratio > 0.8:
            ddos_score += 0.3
        elif up_ratio > 0.6:
            ddos_score += 0.15
            
        is_ddos = ddos_score > 0.5
        threat_type = "DDOS" if is_ddos else "NORMAL"
        
        return NetworkDetectionResult(
            flow_id=flow_id,
            is_threat=is_ddos,
            threat_type=threat_type,
            confidence=ddos_score,
            detection_stage="DDoSDetector",
            features=features,
            timestamp=time.time()
        )
    
    # 辅助方法
    def _check_protocol_indicators(self, packets: List[NetworkPacket]) -> Dict[str, bool]:
        """检查协议指示器"""
        indicators = {
            'ike_esp_detected': False,
            'dtls_tls_tunnel': False,
            'vpn_port_detected': False
        }
        
        for packet in packets:
            # 检查IPsec IKE/ESP
            if packet.dst_port in self.ike_esp_ports or packet.src_port in self.ike_esp_ports:
                indicators['ike_esp_detected'] = True
                
            # 检查VPN常用端口
            vpn_ports = self.ike_esp_ports + self.openvpn_ports + self.wireguard_ports
            if packet.dst_port in vpn_ports or packet.src_port in vpn_ports:
                indicators['vpn_port_detected'] = True
                
            # 检查DTLS/TLS隧道
            if packet.dst_port == 443 and packet.protocol == 'UDP':
                indicators['dtls_tls_tunnel'] = True
                
        return indicators
    
    def _extract_network_features(self, packets: List[NetworkPacket]) -> Dict[str, Any]:
        """提取网络流量特征"""
        if not packets:
            return {}
            
        # 分离上行和下行流量
        up_packets = [p for p in packets if p.direction == 'up']
        down_packets = [p for p in packets if p.direction == 'down']
        
        # 基本统计特征
        packet_sizes = [p.size for p in packets]
        
        # 时间间隔特征
        iats = []
        if len(packets) > 1:
            iats = [packets[i].timestamp - packets[i-1].timestamp for i in range(1, len(packets))]
        
        # 端口分析
        dst_ports = [p.dst_port for p in packets]
        vpn_ports = self.ike_esp_ports + self.openvpn_ports + self.wireguard_ports
        vpn_port_count = sum(1 for port in dst_ports if port in vpn_ports)
        
        features = {
            'packet_count': len(packets),
            'total_bytes': sum(p.size for p in packets),
            'direction_ratio': len(up_packets) / len(packets) if packets else 0,
            'avg_packet_size': np.mean(packet_sizes) if packet_sizes else 0,
            'packet_size_std': np.std(packet_sizes) if packet_sizes else 0,
            'avg_iat': np.mean(iats) if iats else 0,
            'iat_std': np.std(iats) if iats else 0,
            'vpn_port_ratio': vpn_port_count / len(dst_ports) if dst_ports else 0,
            'flow_duration': packets[-1].timestamp - packets[0].timestamp if len(packets) > 1 else 0,
            'packets_per_second': len(packets) / (packets[-1].timestamp - packets[0].timestamp + 1e-6) if len(packets) > 1 else 0
        }
        
        return features
    
    def _relative_entropy_filter(self, features: Dict[str, Any], threshold: float = 0.1) -> bool:
        """相对熵过滤 - 简化版"""
        # 基于统计特征的简单判断
        if features.get('vpn_port_ratio', 0) > 0.5:  # VPN端口比例高
            return True
        if features.get('avg_iat', 0) < 0.2 and features.get('iat_std', 0) < 0.1:  # 高规律性
            return True
            
        return False
    
    def _cnn_lstm_predict(self, packets: List[NetworkPacket]) -> float:
        """模拟1D-CNN + LSTM预测"""
        if not packets:
            return 0.5
            
        packet_sizes = np.array([p.size for p in packets])
        directions = np.array([1 if p.direction == 'up' else 0 for p in packets])
        
        # 模拟1D-CNN特征提取
        cnn_kernel = np.array([0.1, 0.3, 0.3, 0.3, 0.1])
        if len(packet_sizes) >= len(cnn_kernel):
            cnn_features = np.convolve(packet_sizes, cnn_kernel, mode='valid')
        else:
            cnn_features = packet_sizes
            
        # 模拟LSTM时序建模
        lstm_output = np.mean(cnn_features) if len(cnn_features) > 0 else np.mean(packet_sizes)
        
        # 结合方向信息
        direction_score = np.mean(directions) if len(directions) > 0 else 0.5
        
        # 综合得分
        final_score = (lstm_output / 1500 + direction_score) / 2
        return float(np.clip(final_score, 0, 1))


class IntelligentModelSelector:
    """业务场景驱动的智能模型选择器"""
    
    def __init__(self):
        self.model_profiles = {
            'LightGBM': {
                'speed_score': 0.95,
                'accuracy_score': 0.85,
                'resource_efficiency': 0.90,
                'user_experience_impact': 0.95,
                'scalability': 0.88,
                'complexity_tolerance': 0.75
            },
            'XGBoost': {
                'speed_score': 0.75,
                'accuracy_score': 0.92,
                'resource_efficiency': 0.70,
                'user_experience_impact': 0.80,
                'scalability': 0.85,
                'complexity_tolerance': 0.95
            },
            'RandomForest': {
                'speed_score': 0.70,
                'accuracy_score': 0.88,
                'resource_efficiency': 0.75,
                'user_experience_impact': 0.85,
                'scalability': 0.80,
                'complexity_tolerance': 0.85
            },
            'Ensemble': {
                'speed_score': 0.60,
                'accuracy_score': 0.95,
                'resource_efficiency': 0.50,
                'user_experience_impact': 0.70,
                'scalability': 0.65,
                'complexity_tolerance': 0.90
            }
        }
        self.logger = logging.getLogger(f"{__name__}.IntelligentSelector")
        
    def select_optimal_model(self, business_context: BusinessContext) -> Tuple[str, Dict[str, float]]:
        """根据业务上下文选择最优模型"""
        
        # 根据业务场景计算权重分配
        weights = self._calculate_scenario_weights(business_context)
        
        # 计算每个模型的综合得分
        model_scores = {}
        
        for model_name, profile in self.model_profiles.items():
            # 多目标加权得分公式：U(m) = Σ w_i * s_i(m)
            total_score = (
                weights['speed'] * profile['speed_score'] +
                weights['accuracy'] * profile['accuracy_score'] +
                weights['resource'] * profile['resource_efficiency'] +
                weights['user_experience'] * profile['user_experience_impact'] +
                weights['scalability'] * profile['scalability'] +
                weights['complexity'] * profile['complexity_tolerance']
            )
            
            # 业务场景特定调整函数 A(m, c)
            total_score = self._apply_scenario_adjustments(
                total_score, model_name, business_context
            )
            
            model_scores[model_name] = total_score
        
        # 选择最优模型
        optimal_model = max(model_scores.items(), key=lambda x: x[1])
        
        self.logger.info(
            f"业务场景: {business_context.scenario_type.value}, "
            f"选择模型: {optimal_model[0]}, "
            f"置信度: {optimal_model[1]:.4f}"
        )
        
        return optimal_model[0], model_scores
    
    def _calculate_scenario_weights(self, context: BusinessContext) -> Dict[str, float]:
        """根据业务场景计算权重分配"""
        
        if context.scenario_type == BusinessScenarioType.CAMPAIGN_PERIOD:
            # 活动期策略：优先考虑用户体验和响应速度
            return {
                'speed': 0.35,
                'user_experience': 0.30,
                'accuracy': 0.15,
                'resource': 0.10,
                'scalability': 0.08,
                'complexity': 0.02
            }
        
        elif context.scenario_type == BusinessScenarioType.NORMAL_PERIOD:
            # 平时期策略：平衡精度和效率，支持复杂特征工程
            return {
                'accuracy': 0.30,
                'complexity': 0.25,
                'speed': 0.20,
                'resource': 0.15,
                'user_experience': 0.08,
                'scalability': 0.02
            }
            
        elif context.scenario_type == BusinessScenarioType.HIGH_RISK_PERIOD:
            # 高风险期策略：精度至上，复杂特征工程全力开启
            return {
                'accuracy': 0.45,
                'complexity': 0.30,
                'scalability': 0.10,
                'speed': 0.08,
                'resource': 0.05,
                'user_experience': 0.02
            }
            
        else:  # MAINTENANCE_PERIOD
            # 维护期策略：资源效率和稳定性优先
            return {
                'resource': 0.35,
                'speed': 0.25,
                'user_experience': 0.20,
                'accuracy': 0.15,
                'scalability': 0.03,
                'complexity': 0.02
            }
    
    def _apply_scenario_adjustments(self, base_score: float, model_name: str, 
                                  context: BusinessContext) -> float:
        """应用业务场景特定调整函数"""
        
        adjusted_score = base_score
        
        # 活动期特化优化：LightGBM 在高并发情况下的性能优势
        if (context.scenario_type == BusinessScenarioType.CAMPAIGN_PERIOD and 
            model_name == 'LightGBM' and context.traffic_volume_multiplier > 2.0):
            adjusted_score *= 1.15  # 15% 加成
            
        # 平时期复杂特征工程：XGBoost 的二阶优化优势
        if (context.scenario_type == BusinessScenarioType.NORMAL_PERIOD and 
            model_name == 'XGBoost' and context.complex_feature_enabled):
            adjusted_score *= 1.20  # 20% 加成
            
        # 高风险期精度优先：集成模型的精度优势
        if (context.scenario_type == BusinessScenarioType.HIGH_RISK_PERIOD and 
            model_name == 'Ensemble' and context.accuracy_requirement > 0.8):
            adjusted_score *= 1.25  # 25% 加成
            
        return adjusted_score


class EnhancedRiskDetector:
    """增强型风控检测器 - 融合业务风控和网络威胁检测"""
    
    def __init__(self):
        # 业务风控组件
        self.intelligent_selector = IntelligentModelSelector()
        
        # 网络威胁检测组件
        self.network_analyzer = NetworkFlowAnalyzer()
        
        # 融合参数
        self.business_weight = 0.7  # 业务层权重
        self.network_weight = 0.3   # 网络层权重
        
        self.logger = logging.getLogger(f"{__name__}.EnhancedRisk")
        
    def comprehensive_risk_assessment(self, business_features: np.ndarray, 
                                    network_packets: List[NetworkPacket],
                                    business_context: BusinessContext) -> Dict[str, Any]:
        """综合风险评估 - 融合业务和网络层威胁"""
        
        assessment_start = time.time()
        
        # 1. 业务场景驱动的模型选择
        optimal_model, model_scores = self.intelligent_selector.select_optimal_model(business_context)
        
        # 2. 业务层风险检测（简化版）
        business_risk_score = 0.5  # 默认值
        business_threat_type = "NORMAL"
        
        # 模拟业务层风险评分
        if len(business_features) > 0:
            feature_sum = np.sum(business_features)
            if feature_sum > 50:
                business_risk_score = 0.8
                business_threat_type = "HIGH_BUSINESS_RISK"
            elif feature_sum > 30:
                business_risk_score = 0.6
                business_threat_type = "MEDIUM_BUSINESS_RISK"
            else:
                business_risk_score = 0.3
                business_threat_type = "LOW_BUSINESS_RISK"
        
        # 3. 网络层威胁检测
        network_detection = self.network_analyzer.detect_vpn_tunnel(network_packets)
        network_risk_score = network_detection.confidence if network_detection.is_threat else 0.3
        
        # 4. 多层次风险融合算法
        # 使用加权融合: R_total = w1 * R_business + w2 * R_network + α * I(business, network)
        base_risk = (self.business_weight * business_risk_score + 
                    self.network_weight * network_risk_score)
        
        # 交互项: 如果同时检测到业务和网络威胁，增加风险评分
        interaction_boost = 0.0
        if business_risk_score > 0.6 and network_risk_score > 0.6:
            interaction_boost = 0.2  # 20% 的协同威胁加成
            
        total_risk_score = min(1.0, base_risk + interaction_boost)
        
        # 5. 威胁类型综合判断
        if total_risk_score > 0.8:
            final_threat_level = "HIGH"
        elif total_risk_score > 0.6:
            final_threat_level = "MEDIUM"
        else:
            final_threat_level = "LOW"
            
        # 6. 生成综合评估报告
        assessment_result = {
            'total_risk_score': float(total_risk_score),
            'threat_level': final_threat_level,
            'business_assessment': {
                'risk_score': float(business_risk_score),
                'threat_type': business_threat_type,
                'selected_model': optimal_model,
                'model_scores': model_scores
            },
            'network_assessment': {
                'risk_score': float(network_risk_score),
                'threat_type': network_detection.threat_type,
                'detection_stage': network_detection.detection_stage,
                'flow_id': network_detection.flow_id
            },
            'fusion_analysis': {
                'business_weight': self.business_weight,
                'network_weight': self.network_weight,
                'interaction_boost': interaction_boost,
                'assessment_time': time.time() - assessment_start
            },
            'recommendations': self._generate_recommendations(
                business_risk_score, network_risk_score, total_risk_score, business_context
            )
        }
        
        return assessment_result
    
    def _generate_recommendations(self, business_risk: float, network_risk: float, 
                                total_risk: float, context: BusinessContext) -> List[str]:
        """生成风控建议"""
        recommendations = []
        
        if total_risk > 0.8:
            recommendations.append("🚨 高风险警告：建议立即阻断并人工审核")
            if business_risk > 0.7:
                recommendations.append("📊 业务层威胁：加强订单/支付验证")
            if network_risk > 0.7:
                recommendations.append("🌐 网络层威胁：检查VPN/代理工具使用")
                
        elif total_risk > 0.6:
            recommendations.append("⚠️ 中等风险：增强监控和验证")
            if context.scenario_type == BusinessScenarioType.CAMPAIGN_PERIOD:
                recommendations.append("🎯 活动期建议：平衡用户体验与安全验证")
            else:
                recommendations.append("🔍 建议启用复杂特征工程进行深度分析")
                
        else:
            recommendations.append("✅ 低风险：正常处理")
            
        return recommendations


def demo_integrated_detection():
    """演示集成的多层次风控检测系统"""
    print("🚀 多层次风控检测系统演示")
    print("=" * 70)
    
    # 创建增强型风控检测器
    enhanced_detector = EnhancedRiskDetector()
    
    # 创建网络流量分析器
    network_analyzer = NetworkFlowAnalyzer()
    
    print("📊 生成测试数据...")
    
    # 1. 生成不同类型的网络数据包
    test_scenarios = [
        (AttackType.NORMAL, "正常流量"),
        (AttackType.VPN_TUNNEL, "VPN隧道"),
        (AttackType.DDOS, "DDoS攻击"),
        (AttackType.NETWORK_ANOMALY, "网络异常")
    ]
    
    # 2. 模拟业务特征数据
    business_features_normal = np.array([15, 12, 10, 2, 8])
    business_features_risky = np.array([80, 2, 25, 50, 30])
    
    # 3. 创建业务上下文
    campaign_context = BusinessContext(
        scenario_type=BusinessScenarioType.CAMPAIGN_PERIOD,
        user_experience_priority=0.9,
        accuracy_requirement=0.7,
        latency_requirement=50,
        resource_constraint=0.3,
        traffic_volume_multiplier=3.0,
        complex_feature_enabled=False
    )
    
    print("\n🔍 执行网络威胁检测测试:")
    print("-" * 50)
    
    comprehensive_result = None
    for attack_type, description in test_scenarios:
        print(f"\n测试场景: {description}")
        
        # 生成网络数据包
        packets = network_analyzer.generate_network_packets(attack_type, count=50)
        print(f"  生成数据包数量: {len(packets)}")
        
        # VPN检测
        vpn_result = network_analyzer.detect_vpn_tunnel(packets)
        print(f"  VPN检测结果: {vpn_result.threat_type} (置信度: {vpn_result.confidence:.3f})")
        print(f"  检测阶段: {vpn_result.detection_stage}")
        
        # 综合风险评估
        if attack_type in [AttackType.VPN_TUNNEL, AttackType.DDOS]:
            business_features = business_features_risky
        else:
            business_features = business_features_normal
            
        comprehensive_result = enhanced_detector.comprehensive_risk_assessment(
            business_features, packets, campaign_context
        )
        
        print(f"  综合风险评分: {comprehensive_result['total_risk_score']:.3f}")
        print(f"  威胁等级: {comprehensive_result['threat_level']}")
        print(f"  业务风险: {comprehensive_result['business_assessment']['risk_score']:.3f}")
        print(f"  网络风险: {comprehensive_result['network_assessment']['risk_score']:.3f}")
        print(f"  推荐措施: {comprehensive_result['recommendations'][0]}")
    
    print("\n🎯 理论框架总结:")
    print("-" * 50)
    print("1. 四阶段级联VPN检测: 规则预筛 → 相对熵过滤 → 序列模型精判 → 多窗融合")
    print("2. 业务场景驱动选择: 根据活动期/平时期/高风险期智能选择最优模型")
    print("3. 多层次风险融合: R_total = w1*R_business + w2*R_network + α*I(business,network)")
    print("4. 自适应威胁响应: 基于威胁等级和业务场景生成差异化处理建议")
    
    return comprehensive_result


def main():
    """主函数 - 演示集成的多层次风控检测系统"""
    print("🚀 集成VPN检测的梯度提升模型对比系统")
    print("=" * 70)
    
    # 1. 业务场景驱动的模型选择演示
    print("\n🤖 业务场景驱动的智能模型选择...")
    intelligent_selector = IntelligentModelSelector()
    
    # 活动期场景
    campaign_context = BusinessContext(
        scenario_type=BusinessScenarioType.CAMPAIGN_PERIOD,
        user_experience_priority=0.9,
        accuracy_requirement=0.7,
        latency_requirement=50,
        resource_constraint=0.3,
        traffic_volume_multiplier=3.0,
        complex_feature_enabled=False
    )
    
    optimal_model, model_scores = intelligent_selector.select_optimal_model(campaign_context)
    
    print(f"\n🎯 活动期智能模型选择结果:")
    print(f"推荐模型: {optimal_model}")
    print(f"模型评分: {model_scores}")
    
    print(f"\n💡 选择理由:")
    if optimal_model == 'LightGBM':
        print("  • 活动期优先用户体验和响应速度")
        print("  • LightGBM具有极致的推理速度和内存效率")
        print("  • 适合高并发场景下的实时决策")
    
    # 2. 网络威胁检测演示
    print("\n🌐 网络威胁检测演示...")
    network_demo_result = demo_integrated_detection()
    
    # 3. 理论框架总结
    print("\n📚 集成系统理论框架总结:")
    print("=" * 70)
    print("🔄 一、业务层梯度提升模型对比理论:")
    print("  • XGBoost: 二阶梯度优化，适合复杂特征工程")
    print("  • LightGBM: 直方图优化，极致速度，适合活动期")
    print("  • RandomForest: 集成学习，方差减少，可解释性强")
    
    print("\n📊 二、业务场景驱动选择理论:")
    print("  • 数学模型: U(m) = Σ w_i * s_i(m) + α * A(m, c)")
    print("  • 活动期: 优先用户体验(35%) + 速度(30%)")
    print("  • 平时期: 优先精度(30%) + 复杂特征(25%)")
    print("  • 高风险期: 精度至上(45%) + 不计成本(30%)")
    
    print("\n🔒 三、VPN检测四阶段级联算法:")
    print("  • 阶段A - 规则预筛: IKE/ESP协议检测、VPN端口识别")
    print("  • 阶段B - 相对熵过滤: KL散度计算、基线分布比较")
    print("  • 阶段C - 序列模型精判: 1D-CNN + LSTM时序建模")
    print("  • 阶段D - 多窗融合: 多数投票、置信度聚合")
    
    print("\n🔄 四、多层次风控融合理论:")
    print("  • 融合公式: R_total = w1*R_business + w2*R_network + α*I(business,network)")
    print("  • 交互项: 协同威胁检测，双层高风险时增加风险评分")
    print("  • 自适应响应: 基于威胁等级和业务场景生成差异化处理建议")
    
    print("\n🏆 五、企业级风控系统优势:")
    print("  • 理论驱动: 基于数学模型和统计学理论")
    print("  • 场景适应: 根据业务场景动态调整策略")
    print("  • 多层检测: 业务层 + 网络层双重保障")
    print("  • 智能融合: 自动化威胁识别和风险评估")
    
    return network_demo_result


if __name__ == "__main__":
    main()