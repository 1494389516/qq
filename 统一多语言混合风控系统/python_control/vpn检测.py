import numpy as np
import json
import hashlib
import threading
import queue
import time
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import copy

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据结构定义
@dataclass
class Packet:
    """数据包结构"""
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
class Flow:
    """流结构"""
    flow_id: str
    packets: List[Packet]
    start_time: float
    end_time: float
    src_ip: str
    dst_ip: str
    
@dataclass
class DetectionResult:
    """检测结果"""
    flow_id: str
    is_vpn: bool
    confidence: float
    detection_stage: str
    features: Dict[str, Any]
    timestamp: float
    
class VPNType(Enum):
    """VPN类型枚举"""
    OPENVPN = "OpenVPN"
    IPSEC = "IPSec"
    WIREGUARD = "WireGuard"
    PPTP = "PPTP"
    L2TP = "L2TP"
    SSTP = "SSTP"
    UNKNOWN = "Unknown"
    
class DataSource(ABC):
    """数据源抽象基类"""
    
    @abstractmethod
    def collect_data(self) -> List[Packet]:
        """采集数据"""
        pass
        
class PacketCapture(DataSource):
    """数据包捕获器 - 模拟pcap/NetFlow/IPFIX"""
    
    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.is_capturing = False
        
    def collect_data(self) -> List[Packet]:
        """模拟数据包采集"""
        packets = []
        # 模拟从网络接口捕获数据包
        for i in range(10):  # 模拟捕获10个包
            packet = Packet(
                timestamp=time.time() + i * 0.1,
                src_ip=f"192.168.1.{np.random.randint(1, 254)}",
                dst_ip=f"8.8.8.{np.random.randint(1, 254)}",
                src_port=np.random.randint(1024, 65535),
                dst_port=np.random.choice([80, 443, 53, 1194, 500, 4500]),
                protocol=np.random.choice(["TCP", "UDP"]),
                size=np.random.randint(64, 1500),
                direction=np.random.choice(["up", "down"]),
                payload_size=np.random.randint(20, 1400)
            )
            packets.append(packet)
        return packets
        
class TLSMetadataExtractor:
    """TLS元数据提取器"""
    
    def extract_tls_metadata(self, packet: Packet) -> Dict:
        """提取TLS握手和记录头信息"""
        if packet.dst_port not in [443, 993, 995]:
            return {}
            
        # 模拟TLS元数据提取
        tls_metadata = {
            "handshake_type": np.random.choice(["client_hello", "server_hello", "certificate", "finished"]),
            "cipher_suite": np.random.choice(["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"]),
            "tls_version": np.random.choice(["1.2", "1.3"]),
            "sni": f"example{np.random.randint(1, 100)}.com" if np.random.random() > 0.3 else None,
            "certificate_length": np.random.randint(800, 4096) if np.random.random() > 0.7 else None
        }
        return tls_metadata

class MessageBus:
    """消息总线 - 模拟Kafka/Redis Stream"""
    
    def __init__(self, max_size: int = 10000):
        self.queues = defaultdict(lambda: queue.Queue(maxsize=max_size))
        self.subscribers = defaultdict(list)
        
    def publish(self, topic: str, message: Any):
        """发布消息"""
        try:
            self.queues[topic].put_nowait(message)
            logger.debug(f"Published message to topic: {topic}")
        except queue.Full:
            logger.warning(f"Topic {topic} queue is full, dropping message")
            
    def subscribe(self, topic: str, callback):
        """订阅消息"""
        self.subscribers[topic].append(callback)
        
    def get_message(self, topic: str, timeout: float = 1.0) -> Optional[Any]:
        """获取消息"""
        try:
            return self.queues[topic].get(timeout=timeout)
        except queue.Empty:
            return None
            
class SlidingWindow:
    """滑动窗口实现"""
    
    def __init__(self, window_size: float = 5.0, step_size: float = 2.0):
        self.window_size = window_size  # W=5s
        self.step_size = step_size      # S=2s
        self.data = deque()
        self.last_step_time = time.time()
        
    def add_packet(self, packet: Packet):
        """添加数据包到窗口"""
        current_time = packet.timestamp
        self.data.append(packet)
        
        # 清理过期数据
        while self.data and (current_time - self.data[0].timestamp) > self.window_size:
            self.data.popleft()
            
    def should_process(self) -> bool:
        """判断是否应该处理当前窗口"""
        current_time = time.time()
        if (current_time - self.last_step_time) >= self.step_size:
            self.last_step_time = current_time
            return True
        return False
        
    def get_window_data(self) -> List[Packet]:
        """获取当前窗口数据"""
        return list(self.data)
        
class BiDirectionalFeatureExtractor:
    """双向结构化特征提取器"""
    
    def extract_features(self, packets: List[Packet]) -> Dict[str, Any]:
        """提取双向结构化特征"""
        if not packets:
            return self._empty_features()
            
        # 分离上行和下行流量
        up_packets = [p for p in packets if p.direction == 'up']
        down_packets = [p for p in packets if p.direction == 'down']
        
        features = {
            # 包长直方图
            'packet_length_histogram': self._calculate_histogram([p.size for p in packets]),
            'up_packet_length_histogram': self._calculate_histogram([p.size for p in up_packets]),
            'down_packet_length_histogram': self._calculate_histogram([p.size for p in down_packets]),
            
            # IAT(到达间隔)直方图
            'iat_histogram': self._calculate_iat_histogram(packets),
            'up_iat_histogram': self._calculate_iat_histogram(up_packets),
            'down_iat_histogram': self._calculate_iat_histogram(down_packets),
            
            # 方向比/切换率
            'direction_ratio': len(up_packets) / len(packets) if packets else 0,
            'direction_switches': self._calculate_direction_switches(packets),
            
            # 突发度/熵
            'burstiness': self._calculate_burstiness(packets),
            'entropy': self._calculate_entropy([p.size for p in packets]),
            
            # 会话与上下文
            'flow_duration': packets[-1].timestamp - packets[0].timestamp if len(packets) > 1 else 0,
            'packet_count': len(packets),
            'total_bytes': sum(p.size for p in packets)
        }
        
        return features
        
    def _empty_features(self) -> Dict[str, Any]:
        """Empty features for empty packet list"""
        return {
            'packet_length_histogram': [0] * 10,
            'up_packet_length_histogram': [0] * 10,
            'down_packet_length_histogram': [0] * 10,
            'iat_histogram': [0] * 10,
            'up_iat_histogram': [0] * 10,
            'down_iat_histogram': [0] * 10,
            'direction_ratio': 0,
            'direction_switches': 0,
            'burstiness': 0,
            'entropy': 0,
            'flow_duration': 0,
            'packet_count': 0,
            'total_bytes': 0
        }
        
    def _calculate_histogram(self, values: List[float], bins: int = 10) -> List[int]:
        """计算直方图"""
        if not values:
            return [0] * bins
        hist, _ = np.histogram(values, bins=bins)
        return hist.tolist()
        
    def _calculate_iat_histogram(self, packets: List[Packet], bins: int = 10) -> List[int]:
        """计算到达间隔直方图"""
        if len(packets) < 2:
            return [0] * bins
        iats = [packets[i].timestamp - packets[i-1].timestamp for i in range(1, len(packets))]
        return self._calculate_histogram(iats, bins)
        
    def _calculate_direction_switches(self, packets: List[Packet]) -> int:
        """计算方向切换次数"""
        if len(packets) < 2:
            return 0
        switches = sum(1 for i in range(1, len(packets)) 
                      if packets[i].direction != packets[i-1].direction)
        return switches
        
    def _calculate_burstiness(self, packets: List[Packet]) -> float:
        """计算突发度"""
        if len(packets) < 2:
            return 0
        iats = [packets[i].timestamp - packets[i-1].timestamp for i in range(1, len(packets))]
        if not iats:
            return 0
        mean_iat = np.mean(iats)
        var_iat = np.var(iats)
        return float(var_iat / (mean_iat ** 2)) if mean_iat > 0 else 0.0
        
    def _calculate_entropy(self, values: List[float]) -> float:
        """计算熵"""
        if not values:
            return 0
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / len(values)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

class RulePreFilter:
    """阶段A: 规则预筛"""
    
    def __init__(self):
        self.ike_esp_ports = [500, 4500]  # IPsec IKE/ESP
        self.openvpn_ports = [1194]       # OpenVPN
        self.wireguard_ports = [51820]    # WireGuard
        
    def check_protocol_indicators(self, packets: List[Packet]) -> Dict[str, bool]:
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
        
class RelativeEntropyFilter:
    """阶段B: 相对熵过滤"""
    
    def __init__(self, threshold_l: float = 0.1):
        self.threshold_l = threshold_l
        self.p_benign = None  # 非VPN基线
        self.p_vpn = None     # VPN分布/家族均值
        
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算KL散度"""
        # 添加小的常数避免除零
        p = p + 1e-10
        q = q + 1e-10
        # 归一化
        p = p / np.sum(p)
        q = q / np.sum(q)
        return float(np.sum(p * np.log(p / q)))
        
    def multi_dimensional_kl_filter(self, features: Dict[str, Any]) -> bool:
        """多维KL过滤"""
        if self.p_benign is None or self.p_vpn is None:
            return True  # 如果没有基线，默认通过
            
        # 提取直方图特征进行KL计算
        packet_hist = np.array(features['packet_length_histogram'])
        iat_hist = np.array(features['iat_histogram'])
        
        # 计算与基线的KL散度
        kl_packet_benign = self.calculate_kl_divergence(packet_hist, self.p_benign['packet_hist'])
        kl_iat_benign = self.calculate_kl_divergence(iat_hist, self.p_benign['iat_hist'])
        
        # 多维KL值
        multi_kl = (kl_packet_benign + kl_iat_benign) / 2
        
        return multi_kl > self.threshold_l
        
class SequenceModel:
    """阶段C: 序列模型精判"""
    
    def __init__(self):
        self.is_trained = False
        self.model_weights = None
        
    def prepare_sequence_input(self, packets: List[Packet]) -> Dict[str, np.ndarray]:
        """准备序列输入"""
        if not packets:
            return {
                'packet_sizes': np.array([]),
                'directions': np.array([]),
                'iats': np.array([]),
                'context_tokens': np.array([])
            }
            
        packet_sizes = np.array([p.size for p in packets])
        directions = np.array([1 if p.direction == 'up' else 0 for p in packets])
        
        # 计算IAT
        iats = np.array([0] + [packets[i].timestamp - packets[i-1].timestamp 
                              for i in range(1, len(packets))])
        
        # 上下文token(可选)
        context_tokens = np.array([hash(f"{p.src_ip}:{p.dst_ip}") % 1000 for p in packets])
        
        return {
            'packet_sizes': packet_sizes,
            'directions': directions,
            'iats': iats,
            'context_tokens': context_tokens
        }
        
    def cnn_lstm_predict(self, sequence_input: Dict[str, np.ndarray]) -> float:
        """模拟1D-CNN + LSTM预测"""
        packet_sizes = sequence_input['packet_sizes']
        directions = sequence_input['directions']
        iats = sequence_input['iats']
        
        if len(packet_sizes) == 0:
            return 0.5
            
        # 模拟1D-CNN特征提取
        cnn_kernel = np.array([0.1, 0.3, 0.3, 0.3, 0.1])
        if len(packet_sizes) >= len(cnn_kernel):
            cnn_features = np.convolve(packet_sizes, cnn_kernel, mode='valid')
        else:
            cnn_features = packet_sizes
            
        # 模拟LSTM时序建模
        lstm_output = np.mean(cnn_features) if len(cnn_features) > 0 else np.mean(packet_sizes)
        
        # 结合方向和IAT信息
        direction_score = np.mean(directions) if len(directions) > 0 else 0.5
        iat_score = 1.0 / (1.0 + np.mean(iats)) if len(iats) > 0 and np.mean(iats) > 0 else 0.5
        
        # 综合得分
        final_score = (lstm_output / 1500 + direction_score + iat_score) / 3
        return float(np.clip(final_score, 0, 1))
        
class MultiWindowFusion:
    """阶段D: 多窗融合"""
    
    def __init__(self, window_count: int = 3):
        self.window_count = window_count
        self.window_results = deque(maxlen=window_count)
        
    def add_window_result(self, result: DetectionResult):
        """添加窗口结果"""
        self.window_results.append(result)
        
    def majority_voting(self) -> Tuple[bool, float]:
        """多数投票"""
        if not self.window_results:
            return False, 0.0
            
        vpn_votes = sum(1 for r in self.window_results if r.is_vpn)
        total_votes = len(self.window_results)
        
        is_vpn = vpn_votes > total_votes / 2
        confidence = vpn_votes / total_votes if is_vpn else (total_votes - vpn_votes) / total_votes
        
        return is_vpn, float(confidence)
        
    def confidence_aggregation(self) -> Tuple[bool, float]:
        """置信度聚合"""
        if not self.window_results:
            return False, 0.0
            
        vpn_confidences = [r.confidence for r in self.window_results if r.is_vpn]
        normal_confidences = [r.confidence for r in self.window_results if not r.is_vpn]
        
        avg_vpn_conf = np.mean(vpn_confidences) if vpn_confidences else 0.0
        avg_normal_conf = np.mean(normal_confidences) if normal_confidences else 0.0
        
        is_vpn = avg_vpn_conf > avg_normal_conf
        confidence = max(avg_vpn_conf, avg_normal_conf)
        
        return bool(is_vpn), float(confidence)
        
class VPNDetectionSystem:
    """主要的VPN检测系统"""
    
    def __init__(self):
        # 数据采集层
        self.packet_capture = PacketCapture()
        self.tls_extractor = TLSMetadataExtractor()
        
        # 消息总线
        self.message_bus = MessageBus()
        
        # 实时处理
        self.sliding_window = SlidingWindow()
        self.feature_extractor = BiDirectionalFeatureExtractor()
        
        # 检测级联
        self.rule_filter = RulePreFilter()
        self.entropy_filter = RelativeEntropyFilter()
        self.sequence_model = SequenceModel()
        self.multi_window_fusion = MultiWindowFusion()
        
        # 状态
        self.is_running = False
        self.flows = {}
        
    def start_detection(self):
        """启动棂测系统"""
        self.is_running = True
        logger.info("VPN检测系统已启动")
        
        # 启动数据采集线程
        data_thread = threading.Thread(target=self._data_collection_loop)
        data_thread.daemon = True
        data_thread.start()
        
        # 启动检测线程
        detection_thread = threading.Thread(target=self._detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
    def stop_detection(self):
        """停止检测系统"""
        self.is_running = False
        logger.info("VPN检测系统已停止")
        
    def _data_collection_loop(self):
        """数据采集循环"""
        while self.is_running:
            try:
                packets = self.packet_capture.collect_data()
                for packet in packets:
                    # 提取TLS元数据
                    packet.tls_info = self.tls_extractor.extract_tls_metadata(packet)
                    
                    # 发布到消息总线
                    self.message_bus.publish("raw_packets", packet)
                    
                time.sleep(0.1)  # 采集间隔
            except Exception as e:
                logger.error(f"数据采集错误: {e}")
                
    def _detection_loop(self):
        """检测循环"""
        while self.is_running:
            try:
                # 从消息总线获取数据包
                packet = self.message_bus.get_message("raw_packets", timeout=0.1)
                if packet:
                    self.sliding_window.add_packet(packet)
                    
                    # 检查是否应该处理当前窗口
                    if self.sliding_window.should_process():
                        window_data = self.sliding_window.get_window_data()
                        if window_data:
                            result = self._process_window(window_data)
                            if result:
                                self.message_bus.publish("detection_results", result)
                                
            except Exception as e:
                logger.error(f"检测循环错误: {e}")
                
    def _process_window(self, packets: List[Packet]) -> Optional[DetectionResult]:
        """处理窗口数据"""
        if not packets:
            return None
            
        flow_id = f"{packets[0].src_ip}:{packets[0].dst_ip}"
        
        # 阶段A: 规则预筛
        protocol_indicators = self.rule_filter.check_protocol_indicators(packets)
        if not any(protocol_indicators.values()):
            # 没有VPN指示器，直接返回非VPN
            return DetectionResult(
                flow_id=flow_id,
                is_vpn=False,
                confidence=0.9,
                detection_stage="RulePreFilter",
                features=protocol_indicators,
                timestamp=time.time()
            )
            
        # 提取特征
        features = self.feature_extractor.extract_features(packets)
        
        # 阶段B: 相对熵过滤
        if not self.entropy_filter.multi_dimensional_kl_filter(features):
            return DetectionResult(
                flow_id=flow_id,
                is_vpn=False,
                confidence=0.7,
                detection_stage="RelativeEntropyFilter",
                features=features,
                timestamp=time.time()
            )
            
        # 阶段C: 序列模型精判
        sequence_input = self.sequence_model.prepare_sequence_input(packets)
        sequence_score = self.sequence_model.cnn_lstm_predict(sequence_input)
        
        is_vpn = sequence_score > 0.5
        confidence = sequence_score if is_vpn else 1.0 - sequence_score
        
        result = DetectionResult(
            flow_id=flow_id,
            is_vpn=is_vpn,
            confidence=confidence,
            detection_stage="SequenceModel",
            features=features,
            timestamp=time.time()
        )
        
        # 阶段D: 多窗融合
        self.multi_window_fusion.add_window_result(result)
        final_is_vpn, final_confidence = self.multi_window_fusion.confidence_aggregation()
        
        result.is_vpn = final_is_vpn
        result.confidence = final_confidence
        result.detection_stage = "MultiWindowFusion"
        
        return result
        
    def get_detection_results(self) -> List[DetectionResult]:
        """获取检测结果"""
        results = []
        while True:
            result = self.message_bus.get_message("detection_results", timeout=0.1)
            if result is None:
                break
            results.append(result)
        return results

# 模拟流量数据生成器
def generate_sample_packets(n_packets: int = 100, is_vpn: bool = False) -> List[Packet]:
    """
    生成示例数据包用于训练和测试
    """
    packets = []
    base_time = time.time()
    
    for i in range(n_packets):
        if is_vpn:
            # VPN流量特征
            src_port = np.random.choice([1194, 500, 4500])  # VPN常用端口
            dst_port = np.random.choice([1194, 500, 4500])
            size = int(np.random.normal(800, 100))  # 更规律的包大小
            direction = np.random.choice(['up', 'down'], p=[0.45, 0.55])
            time_interval = np.random.exponential(0.1)  # 更规律的时间间隔
        else:
            # 正常流量特征
            src_port = np.random.randint(1024, 65535)
            dst_port = np.random.choice([80, 443, 53, 8080])
            size = int(np.random.choice([64, 128, 256, 512, 1024, 1500]))
            direction = np.random.choice(['up', 'down'])
            time_interval = np.random.exponential(np.random.uniform(0.05, 0.5))
            
        packet = Packet(
            timestamp=base_time + i * time_interval,
            src_ip=f"192.168.1.{np.random.randint(1, 254)}",
            dst_ip=f"8.8.8.{np.random.randint(1, 254)}",
            src_port=src_port,
            dst_port=dst_port,
            protocol=np.random.choice(["TCP", "UDP"]),
            size=max(64, min(1500, size)),
            direction=direction,
            payload_size=max(20, size - 40)
        )
        packets.append(packet)
        
    return packets

def test_vpn_detection_system():
    """
    测试VPN检测系统
    """
    print("=== VPN检测系统测试 ===")
    
    # 创建检测系统
    detection_system = VPNDetectionSystem()
    
    # 生成测试数据
    print("生成测试数据...")
    vpn_packets = generate_sample_packets(50, is_vpn=True)
    normal_packets = generate_sample_packets(50, is_vpn=False)
    
    print(f"生成了 {len(vpn_packets)} 个VPN数据包")
    print(f"生成了 {len(normal_packets)} 个正常数据包")
    
    # 测试VPN流量检测
    print("\n测试VPN流量...")
    vpn_result = detection_system._process_window(vpn_packets)
    if vpn_result:
        print(f"VPN检测结果: {'VPN' if vpn_result.is_vpn else 'Normal'} "
              f"(置信度: {vpn_result.confidence:.2f}, 阶段: {vpn_result.detection_stage})")
    
    # 测试正常流量检测
    print("\n测试正常流量...")
    normal_result = detection_system._process_window(normal_packets)
    if normal_result:
        print(f"正常流量检测结果: {'VPN' if normal_result.is_vpn else 'Normal'} "
              f"(置信度: {normal_result.confidence:.2f}, 阶段: {normal_result.detection_stage})")
    
    # 展示特征信息
    if vpn_result and 'packet_count' in vpn_result.features:
        print(f"\nVPN流量特征:")
        print(f"  数据包数量: {vpn_result.features['packet_count']}")
        print(f"  总字节数: {vpn_result.features['total_bytes']}")
        print(f"  流持续时间: {vpn_result.features['flow_duration']:.2f}s")
        print(f"  方向比: {vpn_result.features['direction_ratio']:.2f}")
        print(f"  突发度: {vpn_result.features['burstiness']:.2f}")
        
    return vpn_result, normal_result

def demo_real_time_detection():
    """
    演示实时检测功能
    """
    print("\n=== 实时检测演示 ===")
    
    detection_system = VPNDetectionSystem()
    
    # 启动检测系统
    detection_system.start_detection()
    print("检测系统已启动，运行5秒...")
    
    # 运行5秒
    time.sleep(5)
    
    # 获取检测结果
    results = detection_system.get_detection_results()
    print(f"\n检测到 {len(results)} 个结果:")
    
    for i, result in enumerate(results[:5]):  # 只显示前5个结果
        print(f"  结果 {i+1}: {'VPN' if result.is_vpn else 'Normal'} "
              f"(置信度: {result.confidence:.2f}, 流ID: {result.flow_id})")
    
    # 停止检测系统
    detection_system.stop_detection()
    print("检测系统已停止")
    
    return results

# 主程序
if __name__ == "__main__":
    try:
        # 运行基本测试
        vpn_result, normal_result = test_vpn_detection_system()
        
        # 运行实时检测演示
        demo_results = demo_real_time_detection()
        
        print("\n=== 测试完成 ===")
        print("VPN检测系统所有功能已验证完成")
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        logger.error(f"程序执行错误: {e}")
        import traceback
        traceback.print_exc()
