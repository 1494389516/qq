"""
VPN检测系统配置文件
根据思维导图的离线侧和监控要求配置各项参数
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class DataCollectionConfig:
    """数据采集配置"""
    # pcap/NetFlow/IPFIX配置
    packet_capture_interface: str = "eth0"
    capture_buffer_size: int = 10000
    packet_timeout: float = 1.0
    
    # TLS元数据提取配置
    tls_metadata_enabled: bool = True
    extract_handshake: bool = True
    extract_record_headers: bool = True
    
    # 数据来源配置
    enable_client_data: bool = True     # 客户端(移动端/浏览器)
    enable_internet_data: bool = True   # Internet(运营商/企业网)
    enable_probe_data: bool = True      # 探针(Mirror/TAP / SPAN / 旁路)

@dataclass
class MessageBusConfig:
    """消息总线配置"""
    # Kafka/Redis Stream模拟配置
    max_queue_size: int = 10000
    timeout_seconds: float = 1.0
    
    # 主题配置
    topics: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = ["raw_packets", "detection_results", "alerts", "metrics"]

@dataclass
class SlidingWindowConfig:
    """滑动窗口配置"""
    window_size: float = 5.0    # W=5s
    step_size: float = 2.0      # S=2s
    max_packets_per_window: int = 1000

@dataclass
class FeatureExtractionConfig:
    """特征提取配置"""
    # 双向结构化特征配置
    histogram_bins: int = 10
    
    # 包长直方图配置
    packet_length_bins: int = 10
    max_packet_size: int = 1500
    
    # IAT(到达间隔)直方图配置
    iat_bins: int = 10
    max_iat: float = 10.0
    
    # 会话与上下文配置
    context_window_seconds: float = 60.0  # 近T秒伴随流统计

@dataclass
class DetectionCascadeConfig:
    """检测级联配置"""
    # Stage A 规则预筛配置
    ike_esp_ports: Optional[List[int]] = None
    openvpn_ports: Optional[List[int]] = None
    wireguard_ports: Optional[List[int]] = None
    
    # Stage B 相对熵过滤配置
    kl_threshold_l: float = 0.1  # 阈值L
    enable_multi_dimensional_kl: bool = True
    
    # Stage C 序列模型配置
    sequence_model_type: str = "CNN_LSTM"  # "CNN_LSTM" or "Transformer" or "PacketBERT"
    max_sequence_length: int = 100
    model_confidence_threshold: float = 0.5
    
    # Stage D 多窗融合配置
    fusion_window_count: int = 3
    fusion_method: str = "confidence_aggregation"  # "majority_voting" or "confidence_aggregation"
    enable_hmm_smoothing: bool = False
    
    def __post_init__(self):
        if self.ike_esp_ports is None:
            self.ike_esp_ports = [500, 4500]
        if self.openvpn_ports is None:
            self.openvpn_ports = [1194]
        if self.wireguard_ports is None:
            self.wireguard_ports = [51820]

@dataclass
class OutputConfig:
    """输出与联动配置"""
    # 结论输出配置
    output_format: str = "json"  # "json" or "xml" or "csv"
    include_confidence: bool = True
    include_features: bool = True
    
    # 对接配置
    enable_siem_integration: bool = False
    enable_soar_integration: bool = False
    enable_policy_engine: bool = False
    enable_dashboard: bool = True
    
    # 策略引擎配置
    auto_block_high_confidence: bool = False
    auto_rate_limit: bool = False
    high_confidence_threshold: float = 0.8

@dataclass
class OfflineConfig:
    """离线侧配置"""
    # 存储配置
    storage_type: str = "file"  # "hdfs" or "object_storage" or "file"
    storage_path: str = "/tmp/vpn_detection_data"
    
    # 数据保留配置
    store_raw_packets: bool = True
    store_features: bool = True
    store_logs: bool = True
    data_retention_days: int = 30
    
    # 训练与标定配置
    enable_offline_training: bool = True
    training_data_ratio: float = 0.8  # 训练数据比例
    validation_data_ratio: float = 0.2  # 验证数据比例
    
    # 模型管理配置
    model_registry_enabled: bool = True
    model_versioning: bool = True
    auto_model_update: bool = False
    
    # 基线配置
    baseline_domains: Optional[List[str]] = None  # P_benign: 办公/家庭/蜂窝
    
    def __post_init__(self):
        if self.baseline_domains is None:
            self.baseline_domains = ["office", "home", "cellular"]

@dataclass
class MonitoringConfig:
    """监控与漂移配置"""
    # 性能指标配置
    enable_fpr_monitoring: bool = True      # False Positive Rate
    enable_latency_monitoring: bool = True   # 延迟监控
    enable_throughput_monitoring: bool = True # 吞吐量监控
    
    # 目标性能指标
    target_fpr: float = 0.05               # 目标假阳性率 5%
    target_latency_ms: float = 100.0       # 目标延迟 100ms
    target_throughput_gbps: float = 10.0   # 目标吞吐量 10Gbps
    
    # 分布漂移检测配置
    enable_drift_detection: bool = True
    drift_detection_window: int = 1000      # 漂移检测窗口大小
    drift_threshold: float = 0.1            # 漂移阈值
    
    # 反馈回路配置
    enable_auto_threshold_adjustment: bool = True
    enable_auto_retraining_trigger: bool = True
    retraining_drift_threshold: float = 0.2

@dataclass
class PerformanceConfig:
    """性能与部署配置"""
    # 扩展性配置
    target_throughput_gbps: float = 10.0   # 10Gbps可扩展
    
    # 流处理配置
    stream_processing_framework: str = "simulation"  # "flink" or "spark" or "simulation"
    parallelism_level: int = 4
    checkpoint_interval_ms: int = 1000
    
    # 部署配置
    deployment_mode: str = "single_node"    # "single_node" or "cluster"
    enable_gray_deployment: bool = True     # 灰度上线
    gray_traffic_percentage: float = 0.1    # 灰度流量比例
    
    # 策略配置
    alert_before_block: bool = True         # 先告警后阻断

@dataclass
class VPNDetectionSystemConfig:
    """VPN检测系统完整配置"""
    data_collection: Optional[DataCollectionConfig] = None
    message_bus: Optional[MessageBusConfig] = None
    sliding_window: Optional[SlidingWindowConfig] = None
    feature_extraction: Optional[FeatureExtractionConfig] = None
    detection_cascade: Optional[DetectionCascadeConfig] = None
    output: Optional[OutputConfig] = None
    offline: Optional[OfflineConfig] = None
    monitoring: Optional[MonitoringConfig] = None
    performance: Optional[PerformanceConfig] = None
    
    # 系统配置
    log_level: str = "INFO"
    enable_debug_mode: bool = False
    config_version: str = "1.0.0"
    
    def __post_init__(self):
        if self.data_collection is None:
            self.data_collection = DataCollectionConfig()
        if self.message_bus is None:
            self.message_bus = MessageBusConfig()
        if self.sliding_window is None:
            self.sliding_window = SlidingWindowConfig()
        if self.feature_extraction is None:
            self.feature_extraction = FeatureExtractionConfig()
        if self.detection_cascade is None:
            self.detection_cascade = DetectionCascadeConfig()
        if self.output is None:
            self.output = OutputConfig()
        if self.offline is None:
            self.offline = OfflineConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'VPNDetectionSystemConfig':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 递归创建配置对象
        def dict_to_dataclass(cls_type, data):
            if hasattr(cls_type, '__annotations__'):
                kwargs = {}
                for field_name, field_type in cls_type.__annotations__.items():
                    if field_name in data:
                        if hasattr(field_type, '__annotations__'):
                            kwargs[field_name] = dict_to_dataclass(field_type, data[field_name])
                        else:
                            kwargs[field_name] = data[field_name]
                return cls_type(**kwargs)
            return data
        
        return dict_to_dataclass(cls, config_dict)

# 默认配置实例
DEFAULT_CONFIG = VPNDetectionSystemConfig()

def get_default_config() -> VPNDetectionSystemConfig:
    """获取默认配置"""
    return DEFAULT_CONFIG

def create_production_config() -> VPNDetectionSystemConfig:
    """创建生产环境配置"""
    config = VPNDetectionSystemConfig()
    
    # 确保所有子配置对象已初始化
    if config.performance is None:
        config.performance = PerformanceConfig()
    if config.monitoring is None:
        config.monitoring = MonitoringConfig()
    if config.output is None:
        config.output = OutputConfig()
    if config.offline is None:
        config.offline = OfflineConfig()
    
    # 生产环境优化配置
    config.performance.target_throughput_gbps = 10.0
    config.performance.deployment_mode = "cluster"
    config.performance.parallelism_level = 8
    
    # 监控配置加强
    config.monitoring.enable_fpr_monitoring = True
    config.monitoring.enable_latency_monitoring = True
    config.monitoring.enable_throughput_monitoring = True
    config.monitoring.enable_drift_detection = True
    
    # 安全策略
    config.output.auto_block_high_confidence = True
    config.output.high_confidence_threshold = 0.9
    config.performance.alert_before_block = True
    
    # 离线训练启用
    config.offline.enable_offline_training = True
    config.offline.model_registry_enabled = True
    config.offline.auto_model_update = True
    
    return config

def create_development_config() -> VPNDetectionSystemConfig:
    """创建开发环境配置"""
    config = VPNDetectionSystemConfig()
    
    # 确保所有子配置对象已初始化
    if config.performance is None:
        config.performance = PerformanceConfig()
    if config.output is None:
        config.output = OutputConfig()
    
    # 开发环境配置
    config.enable_debug_mode = True
    config.log_level = "DEBUG"
    
    # 性能要求降低
    config.performance.target_throughput_gbps = 1.0
    config.performance.deployment_mode = "single_node"
    config.performance.parallelism_level = 2
    
    # 禁用自动阻断
    config.output.auto_block_high_confidence = False
    config.output.enable_policy_engine = False
    
    return config

if __name__ == "__main__":
    # 生成默认配置文件
    config = get_default_config()
    config.save_to_file("default_config.json")
    print("默认配置已生成: default_config.json")
    
    # 生成生产配置文件
    prod_config = create_production_config()
    prod_config.save_to_file("production_config.json")
    print("生产配置已生成: production_config.json")
    
    # 生成开发配置文件
    dev_config = create_development_config()
    dev_config.save_to_file("development_config.json")
    print("开发配置已生成: development_config.json")