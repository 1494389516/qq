//! VPN检测系统 - Rust模块
//! 提供高性能、内存安全的消息队列和TLS解析功能

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use tracing::{debug, error, info, warn};

pub mod message_bus;
pub mod tls_parser;
pub mod data_storage;
pub mod ffi;

// 重新导出主要类型
pub use message_bus::{MessageBus, MessageBusConfig, TopicSubscription};
pub use tls_parser::{TlsParser, TlsHandshakeInfo, TlsMetadata};
pub use data_storage::{DataStorage, StorageConfig, StorageError};

/// 错误类型定义
#[derive(thiserror::Error, Debug)]
pub enum VpnDetectionError {
    #[error("消息总线错误: {0}")]
    MessageBus(String),
    
    #[error("TLS解析错误: {0}")]
    TlsParser(String),
    
    #[error("存储错误: {0}")]
    Storage(String),
    
    #[error("配置错误: {0}")]
    Configuration(String),
    
    #[error("网络错误: {0}")]
    Network(String),
}

/// 结果类型别名
pub type Result<T> = std::result::Result<T, VpnDetectionError>;

/// 数据包结构 (与C++兼容)
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Packet {
    pub timestamp_ns: u64,          // 纳秒时间戳
    pub src_ip: u32,                // 源IP地址
    pub dst_ip: u32,                // 目标IP地址
    pub src_port: u16,              // 源端口
    pub dst_port: u16,              // 目标端口
    pub protocol: u8,               // 协议类型
    pub packet_size: u16,           // 数据包大小
    pub payload_size: u16,          // 载荷大小
    pub direction: u8,              // 方向 (0=up, 1=down)
    pub has_tls: bool,              // 是否包含TLS
    pub tls_version: u8,            // TLS版本
    pub cipher_suite: u16,          // 密码套件
}

impl Packet {
    pub fn new() -> Self {
        Self {
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            src_ip: 0,
            dst_ip: 0,
            src_port: 0,
            dst_port: 0,
            protocol: 0,
            packet_size: 0,
            payload_size: 0,
            direction: 0,
            has_tls: false,
            tls_version: 0,
            cipher_suite: 0,
        }
    }
    
    /// 获取流标识符
    pub fn flow_key(&self) -> String {
        format!("{}:{}:{}:{}:{}", 
                self.src_ip, self.dst_ip, 
                self.src_port, self.dst_port, 
                self.protocol)
    }
    
    /// 检查是否可能是VPN流量
    pub fn is_potential_vpn(&self) -> bool {
        // VPN常用端口检查
        let vpn_ports = [500, 1194, 4500, 51820];
        vpn_ports.contains(&self.src_port) || vpn_ports.contains(&self.dst_port)
    }
}

/// 检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub flow_id: String,
    pub is_vpn: bool,
    pub confidence: f64,
    pub detection_stage: String,
    pub timestamp: DateTime<Utc>,
    pub features: HashMap<String, serde_json::Value>,
}

impl DetectionResult {
    pub fn new(flow_id: String, is_vpn: bool, confidence: f64, stage: String) -> Self {
        Self {
            flow_id,
            is_vpn,
            confidence,
            detection_stage: stage,
            timestamp: Utc::now(),
            features: HashMap::new(),
        }
    }
}

/// 系统配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub message_bus: MessageBusConfig,
    pub storage: StorageConfig,
    pub tls_parser: TlsParserConfig,
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsParserConfig {
    pub enabled: bool,
    pub extract_handshake: bool,
    pub extract_certificates: bool,
    pub max_certificate_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub max_concurrent_flows: usize,
    pub packet_buffer_size: usize,
    pub processing_timeout_ms: u64,
    pub gc_interval_ms: u64,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            message_bus: MessageBusConfig::default(),
            storage: StorageConfig::default(),
            tls_parser: TlsParserConfig {
                enabled: true,
                extract_handshake: true,
                extract_certificates: false,
                max_certificate_size: 4096,
            },
            performance: PerformanceConfig {
                max_concurrent_flows: 100000,
                packet_buffer_size: 10000,
                processing_timeout_ms: 1000,
                gc_interval_ms: 30000,
            },
        }
    }
}

/// 主要的Rust组件集成器
pub struct VpnDetectionSystem {
    config: SystemConfig,
    message_bus: Arc<MessageBus>,
    tls_parser: Arc<Mutex<TlsParser>>,
    data_storage: Arc<Mutex<DataStorage>>,
    running: Arc<Mutex<bool>>,
}

impl VpnDetectionSystem {
    /// 创建新的检测系统实例
    pub fn new(config: SystemConfig) -> Result<Self> {
        let message_bus = Arc::new(MessageBus::new(config.message_bus.clone())?);
        let tls_parser = Arc::new(Mutex::new(TlsParser::new(config.tls_parser.clone())?));
        let data_storage = Arc::new(Mutex::new(DataStorage::new(config.storage.clone())?));
        
        Ok(Self {
            config,
            message_bus,
            tls_parser,
            data_storage,
            running: Arc::new(Mutex::new(false)),
        })
    }
    
    /// 启动系统
    pub async fn start(&self) -> Result<()> {
        {
            let mut running = self.running.lock().unwrap();
            if *running {
                return Err(VpnDetectionError::Configuration("系统已经在运行".to_string()));
            }
            *running = true;
        }
        
        info!("启动VPN检测系统...");
        
        // 启动消息总线
        self.message_bus.start().await?;
        
        // 启动数据存储
        {
            let mut storage = self.data_storage.lock().unwrap();
            storage.start().await?;
        }
        
        info!("VPN检测系统启动完成");
        Ok(())
    }
    
    /// 停止系统
    pub async fn stop(&self) -> Result<()> {
        {
            let mut running = self.running.lock().unwrap();
            if !*running {
                return Ok(());
            }
            *running = false;
        }
        
        info!("停止VPN检测系统...");
        
        // 停止各个组件
        self.message_bus.stop().await?;
        
        {
            let mut storage = self.data_storage.lock().unwrap();
            storage.stop().await?;
        }
        
        info!("VPN检测系统已停止");
        Ok(())
    }
    
    /// 处理数据包
    pub async fn process_packet(&self, packet: Packet) -> Result<()> {
        // TLS解析
        let mut tls_metadata = None;
        if packet.has_tls {
            let parser = self.tls_parser.lock().unwrap();
            tls_metadata = parser.parse_packet(&packet).ok();
        }
        
        // 发布到消息总线
        let message = serde_json::json!({
            "packet": packet,
            "tls_metadata": tls_metadata,
            "timestamp": Utc::now()
        });
        
        self.message_bus.publish("raw_packets", message).await?;
        
        Ok(())
    }
    
    /// 获取检测结果
    pub async fn get_detection_results(&self) -> Result<Vec<DetectionResult>> {
        let messages = self.message_bus.get_messages("detection_results", 100).await?;
        
        let mut results = Vec::new();
        for message in messages {
            if let Ok(result) = serde_json::from_value::<DetectionResult>(message) {
                results.push(result);
            }
        }
        
        Ok(results)
    }
    
    /// 获取系统统计信息
    pub async fn get_statistics(&self) -> Result<SystemStatistics> {
        let mb_stats = self.message_bus.get_statistics().await?;
        
        let storage_stats = {
            let storage = self.data_storage.lock().unwrap();
            storage.get_statistics()?
        };
        
        Ok(SystemStatistics {
            message_bus: mb_stats,
            storage: storage_stats,
            uptime_ms: 0, // TODO: 实现运行时间统计
        })
    }
}

/// 系统统计信息
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemStatistics {
    pub message_bus: message_bus::MessageBusStatistics,
    pub storage: data_storage::StorageStatistics,
    pub uptime_ms: u64,
}

/// 初始化日志系统
pub fn init_logging() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_thread_ids(true)
        .with_thread_names(true)
        .init();
}

/// 创建默认配置
pub fn create_default_config() -> SystemConfig {
    SystemConfig::default()
}

/// 从文件加载配置
pub fn load_config_from_file(path: &str) -> Result<SystemConfig> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| VpnDetectionError::Configuration(format!("读取配置文件失败: {}", e)))?;
    
    let config: SystemConfig = serde_json::from_str(&content)
        .map_err(|e| VpnDetectionError::Configuration(format!("解析配置文件失败: {}", e)))?;
    
    Ok(config)
}

/// 保存配置到文件
pub fn save_config_to_file(config: &SystemConfig, path: &str) -> Result<()> {
    let content = serde_json::to_string_pretty(config)
        .map_err(|e| VpnDetectionError::Configuration(format!("序列化配置失败: {}", e)))?;
    
    std::fs::write(path, content)
        .map_err(|e| VpnDetectionError::Configuration(format!("写入配置文件失败: {}", e)))?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_packet_creation() {
        let packet = Packet::new();
        assert_eq!(packet.src_ip, 0);
        assert_eq!(packet.dst_ip, 0);
        assert!(!packet.has_tls);
    }
    
    #[test]
    fn test_vpn_port_detection() {
        let mut packet = Packet::new();
        packet.dst_port = 1194; // OpenVPN
        assert!(packet.is_potential_vpn());
        
        packet.dst_port = 80; // HTTP
        assert!(!packet.is_potential_vpn());
    }
    
    #[tokio::test]
    async fn test_system_startup() {
        let config = SystemConfig::default();
        let system = VpnDetectionSystem::new(config).unwrap();
        
        // 测试启动和停止
        assert!(system.start().await.is_ok());
        assert!(system.stop().await.is_ok());
    }
}

/// 便利函数：创建并启动检测系统
pub async fn create_and_start_system(config_path: Option<&str>) -> Result<VpnDetectionSystem> {
    let config = match config_path {
        Some(path) => load_config_from_file(path)?,
        None => SystemConfig::default(),
    };
    
    let system = VpnDetectionSystem::new(config)?;
    system.start().await?;
    
    Ok(system)
}