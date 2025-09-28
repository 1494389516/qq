//! 高性能消息总线模块
//! 基于Rust的内存安全保证和高并发能力实现

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock as TokioRwLock;
use tokio::time::{Duration, timeout};
use tracing::{debug, error, info, warn};
use crate::{Result, VpnDetectionError};

/// 消息总线配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageBusConfig {
    /// 每个主题的最大队列大小
    pub max_queue_size: usize,
    /// 是否使用有界队列
    pub use_bounded_queues: bool,
    /// 消息超时时间(毫秒)
    pub message_timeout_ms: u64,
    /// 默认主题列表
    pub default_topics: Vec<String>,
    /// 是否启用统计信息
    pub enable_statistics: bool,
}

impl Default for MessageBusConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            use_bounded_queues: true,
            message_timeout_ms: 5000,
            default_topics: vec![
                "raw_packets".to_string(),
                "detection_results".to_string(),
                "alerts".to_string(),
                "metrics".to_string(),
                "system_events".to_string(),
            ],
            enable_statistics: true,
        }
    }
}

/// 主题订阅信息
#[derive(Debug, Clone)]
pub struct TopicSubscription {
    pub topic: String,
    pub subscriber_id: String,
    pub created_at: std::time::SystemTime,
}

/// 消息统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageBusStatistics {
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub active_topics: usize,
    pub active_subscribers: usize,
    pub queue_sizes: HashMap<String, usize>,
    pub error_count: u64,
}

/// 消息总线实现
pub struct MessageBus {
    config: MessageBusConfig,
    topics: Arc<RwLock<HashMap<String, TopicInfo>>>,
    statistics: Arc<TokioRwLock<MessageBusStatistics>>,
    running: Arc<RwLock<bool>>,
}

#[derive(Debug)]
struct TopicInfo {
    sender: Sender<Value>,
    receiver: Receiver<Value>,
    subscribers: Vec<String>,
    message_count: u64,
}

impl MessageBus {
    /// 创建新的消息总线
    pub fn new(config: MessageBusConfig) -> Result<Self> {
        let topics = Arc::new(RwLock::new(HashMap::new()));
        let statistics = Arc::new(TokioRwLock::new(MessageBusStatistics {
            total_messages_sent: 0,
            total_messages_received: 0,
            active_topics: 0,
            active_subscribers: 0,
            queue_sizes: HashMap::new(),
            error_count: 0,
        }));
        
        let bus = Self {
            config: config.clone(),
            topics,
            statistics,
            running: Arc::new(RwLock::new(false)),
        };
        
        // 创建默认主题
        for topic in &config.default_topics {
            bus.create_topic(topic)?;
        }
        
        Ok(bus)
    }
    
    /// 启动消息总线
    pub async fn start(&self) -> Result<()> {
        {
            let mut running = self.running.write().unwrap();
            if *running {
                return Err(VpnDetectionError::MessageBus("消息总线已经在运行".to_string()));
            }
            *running = true;
        }
        
        info!("消息总线已启动");
        Ok(())
    }
    
    /// 停止消息总线
    pub async fn stop(&self) -> Result<()> {
        {
            let mut running = self.running.write().unwrap();
            if !*running {
                return Ok(());
            }
            *running = false;
        }
        
        info!("消息总线已停止");
        Ok(())
    }
    
    /// 创建主题
    pub fn create_topic(&self, topic: &str) -> Result<()> {
        let mut topics = self.topics.write().unwrap();
        
        if topics.contains_key(topic) {
            return Ok(()); // 主题已存在
        }
        
        let (sender, receiver) = if self.config.use_bounded_queues {
            bounded(self.config.max_queue_size)
        } else {
            unbounded()
        };
        
        let topic_info = TopicInfo {
            sender,
            receiver,
            subscribers: Vec::new(),
            message_count: 0,
        };
        
        topics.insert(topic.to_string(), topic_info);
        debug!("创建主题: {}", topic);
        
        Ok(())
    }
    
    /// 发布消息到主题
    pub async fn publish(&self, topic: &str, message: Value) -> Result<()> {
        // 检查是否在运行
        {
            let running = self.running.read().unwrap();
            if !*running {
                return Err(VpnDetectionError::MessageBus("消息总线未启动".to_string()));
            }
        }
        
        // 确保主题存在
        if !self.topic_exists(topic) {
            self.create_topic(topic)?;
        }
        
        // 发送消息
        {
            let mut topics = self.topics.write().unwrap();
            if let Some(topic_info) = topics.get_mut(topic) {
                match topic_info.sender.try_send(message) {
                    Ok(()) => {
                        topic_info.message_count += 1;
                        
                        // 更新统计信息
                        if self.config.enable_statistics {
                            tokio::spawn({
                                let stats = Arc::clone(&self.statistics);
                                async move {
                                    let mut stats = stats.write().await;
                                    stats.total_messages_sent += 1;
                                }
                            });
                        }
                        
                        debug!("消息已发布到主题: {}", topic);
                    }
                    Err(e) => {
                        let error_msg = format!("发布消息失败: {}", e);
                        error!("{}", error_msg);
                        
                        // 更新错误统计
                        if self.config.enable_statistics {
                            tokio::spawn({
                                let stats = Arc::clone(&self.statistics);
                                async move {
                                    let mut stats = stats.write().await;
                                    stats.error_count += 1;
                                }
                            });
                        }
                        
                        return Err(VpnDetectionError::MessageBus(error_msg));
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 订阅主题
    pub fn subscribe(&self, topic: &str, subscriber_id: &str) -> Result<TopicSubscription> {
        // 确保主题存在
        if !self.topic_exists(topic) {
            self.create_topic(topic)?;
        }
        
        // 添加订阅者
        {
            let mut topics = self.topics.write().unwrap();
            if let Some(topic_info) = topics.get_mut(topic) {
                if !topic_info.subscribers.contains(&subscriber_id.to_string()) {
                    topic_info.subscribers.push(subscriber_id.to_string());
                }
            }
        }
        
        let subscription = TopicSubscription {
            topic: topic.to_string(),
            subscriber_id: subscriber_id.to_string(),
            created_at: std::time::SystemTime::now(),
        };
        
        debug!("订阅者 {} 订阅了主题: {}", subscriber_id, topic);
        Ok(subscription)
    }
    
    /// 取消订阅
    pub fn unsubscribe(&self, topic: &str, subscriber_id: &str) -> Result<()> {
        let mut topics = self.topics.write().unwrap();
        if let Some(topic_info) = topics.get_mut(topic) {
            topic_info.subscribers.retain(|id| id != subscriber_id);
            debug!("订阅者 {} 取消订阅主题: {}", subscriber_id, topic);
        }
        
        Ok(())
    }
    
    /// 获取单个消息(非阻塞)
    pub async fn get_message(&self, topic: &str) -> Result<Option<Value>> {
        let topics = self.topics.read().unwrap();
        if let Some(topic_info) = topics.get(topic) {
            match topic_info.receiver.try_recv() {
                Ok(message) => {
                    // 更新统计信息
                    if self.config.enable_statistics {
                        tokio::spawn({
                            let stats = Arc::clone(&self.statistics);
                            async move {
                                let mut stats = stats.write().await;
                                stats.total_messages_received += 1;
                            }
                        });
                    }
                    
                    debug!("从主题 {} 获取消息", topic);
                    Ok(Some(message))
                }
                Err(TryRecvError::Empty) => Ok(None),
                Err(TryRecvError::Disconnected) => {
                    Err(VpnDetectionError::MessageBus(format!("主题 {} 已断开连接", topic)))
                }
            }
        } else {
            Err(VpnDetectionError::MessageBus(format!("主题 {} 不存在", topic)))
        }
    }
    
    /// 获取多个消息(带超时)
    pub async fn get_messages(&self, topic: &str, max_count: usize) -> Result<Vec<Value>> {
        let mut messages = Vec::new();
        let timeout_duration = Duration::from_millis(self.config.message_timeout_ms);
        
        for _ in 0..max_count {
            match timeout(timeout_duration, self.get_message_blocking(topic)).await {
                Ok(Ok(Some(message))) => messages.push(message),
                Ok(Ok(None)) => break, // 没有更多消息
                Ok(Err(e)) => return Err(e),
                Err(_) => break, // 超时
            }
        }
        
        Ok(messages)
    }
    
    /// 阻塞获取消息
    async fn get_message_blocking(&self, topic: &str) -> Result<Option<Value>> {
        // 简化实现，实际上应该使用异步通道
        tokio::time::sleep(Duration::from_millis(10)).await;
        self.get_message(topic).await
    }
    
    /// 检查主题是否存在
    pub fn topic_exists(&self, topic: &str) -> bool {
        let topics = self.topics.read().unwrap();
        topics.contains_key(topic)
    }
    
    /// 获取主题列表
    pub fn list_topics(&self) -> Vec<String> {
        let topics = self.topics.read().unwrap();
        topics.keys().cloned().collect()
    }
    
    /// 获取主题的订阅者列表
    pub fn get_subscribers(&self, topic: &str) -> Result<Vec<String>> {
        let topics = self.topics.read().unwrap();
        if let Some(topic_info) = topics.get(topic) {
            Ok(topic_info.subscribers.clone())
        } else {
            Err(VpnDetectionError::MessageBus(format!("主题 {} 不存在", topic)))
        }
    }
    
    /// 获取统计信息
    pub async fn get_statistics(&self) -> Result<MessageBusStatistics> {
        if !self.config.enable_statistics {
            return Err(VpnDetectionError::MessageBus("统计信息未启用".to_string()));
        }
        
        let mut stats = self.statistics.write().await;
        
        // 更新当前状态
        let topics = self.topics.read().unwrap();
        stats.active_topics = topics.len();
        
        let mut total_subscribers = 0;
        let mut queue_sizes = HashMap::new();
        
        for (topic_name, topic_info) in topics.iter() {
            total_subscribers += topic_info.subscribers.len();
            queue_sizes.insert(topic_name.clone(), topic_info.receiver.len());
        }
        
        stats.active_subscribers = total_subscribers;
        stats.queue_sizes = queue_sizes;
        
        Ok(stats.clone())
    }
    
    /// 清理主题(移除空主题)
    pub fn cleanup_topics(&self) -> Result<usize> {
        let mut topics = self.topics.write().unwrap();
        let initial_count = topics.len();
        
        topics.retain(|_, topic_info| {
            !topic_info.subscribers.is_empty() || topic_info.receiver.len() > 0
        });
        
        let removed_count = initial_count - topics.len();
        if removed_count > 0 {
            info!("清理了 {} 个空主题", removed_count);
        }
        
        Ok(removed_count)
    }
    
    /// 获取主题的消息数量
    pub fn get_topic_message_count(&self, topic: &str) -> Result<u64> {
        let topics = self.topics.read().unwrap();
        if let Some(topic_info) = topics.get(topic) {
            Ok(topic_info.message_count)
        } else {
            Err(VpnDetectionError::MessageBus(format!("主题 {} 不存在", topic)))
        }
    }
    
    /// 清空主题的所有消息
    pub fn clear_topic(&self, topic: &str) -> Result<usize> {
        let topics = self.topics.read().unwrap();
        if let Some(topic_info) = topics.get(topic) {
            let mut count = 0;
            while topic_info.receiver.try_recv().is_ok() {
                count += 1;
            }
            info!("清空主题 {} 的 {} 条消息", topic, count);
            Ok(count)
        } else {
            Err(VpnDetectionError::MessageBus(format!("主题 {} 不存在", topic)))
        }
    }
}

/// 批量消息发布器
pub struct BatchPublisher {
    message_bus: Arc<MessageBus>,
    topic: String,
    batch_size: usize,
    buffer: Vec<Value>,
}

impl BatchPublisher {
    pub fn new(message_bus: Arc<MessageBus>, topic: String, batch_size: usize) -> Self {
        Self {
            message_bus,
            topic,
            batch_size,
            buffer: Vec::with_capacity(batch_size),
        }
    }
    
    /// 添加消息到批处理缓冲区
    pub async fn add_message(&mut self, message: Value) -> Result<()> {
        self.buffer.push(message);
        
        if self.buffer.len() >= self.batch_size {
            self.flush().await?;
        }
        
        Ok(())
    }
    
    /// 刷新缓冲区(发送所有待处理消息)
    pub async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        
        let batch_message = serde_json::json!({
            "type": "batch",
            "messages": self.buffer.clone(),
            "count": self.buffer.len(),
            "timestamp": chrono::Utc::now()
        });
        
        self.message_bus.publish(&self.topic, batch_message).await?;
        self.buffer.clear();
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;
    
    #[tokio::test]
    async fn test_message_bus_basic() {
        let config = MessageBusConfig::default();
        let bus = MessageBus::new(config).unwrap();
        
        bus.start().await.unwrap();
        
        // 测试发布和接收消息
        let test_message = serde_json::json!({"test": "message"});
        bus.publish("test_topic", test_message.clone()).await.unwrap();
        
        let received = bus.get_message("test_topic").await.unwrap();
        assert!(received.is_some());
        assert_eq!(received.unwrap(), test_message);
        
        bus.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_subscription() {
        let config = MessageBusConfig::default();
        let bus = MessageBus::new(config).unwrap();
        
        let subscription = bus.subscribe("test_topic", "test_subscriber").unwrap();
        assert_eq!(subscription.topic, "test_topic");
        assert_eq!(subscription.subscriber_id, "test_subscriber");
        
        let subscribers = bus.get_subscribers("test_topic").unwrap();
        assert!(subscribers.contains(&"test_subscriber".to_string()));
        
        bus.unsubscribe("test_topic", "test_subscriber").unwrap();
        let subscribers = bus.get_subscribers("test_topic").unwrap();
        assert!(!subscribers.contains(&"test_subscriber".to_string()));
    }
    
    #[tokio::test]
    async fn test_batch_publisher() {
        let config = MessageBusConfig::default();
        let bus = Arc::new(MessageBus::new(config).unwrap());
        bus.start().await.unwrap();
        
        let mut batch_publisher = BatchPublisher::new(
            Arc::clone(&bus), 
            "test_batch".to_string(), 
            3
        );
        
        // 添加消息到批处理
        for i in 0..5 {
            let message = serde_json::json!({"id": i});
            batch_publisher.add_message(message).await.unwrap();
        }
        
        // 手动刷新剩余消息
        batch_publisher.flush().await.unwrap();
        
        bus.stop().await.unwrap();
    }
}