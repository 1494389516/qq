#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <thread>

// 条件编译SIMD支持
#ifdef __x86_64__
#include <immintrin.h>  // SIMD支持
#define SIMD_AVAILABLE
#endif

namespace ddos_defense {

/**
 * 高性能DDoS检测引擎 - C++实现
 * 基于风控算法专家的数学建模理论
 * 
 * 核心算法特性：
 * 1. 多维特征融合检测模型
 * 2. SIMD指令集优化的特征提取
 * 3. 实时流量基线自适应算法
 * 4. 微秒级威胁响应机制
 */

struct NetworkPacket {
    uint64_t timestamp;
    std::string src_ip;
    std::string dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    std::string protocol;
    uint32_t size;
    uint8_t flags;
    uint32_t payload_size;
};

struct FlowStatistics {
    double packets_per_second;
    double bytes_per_second;
    double avg_packet_size;
    double connection_rate;
    double small_packet_ratio;  // 小包占比
    double syn_flood_ratio;     // SYN洪水比例
    double source_entropy;      // 源IP熵值
    double destination_entropy; // 目标IP熵值
    double protocol_diversity;  // 协议多样性
    double temporal_variance;   // 时间方差
};

struct DDoSDetectionResult {
    enum ThreatLevel {
        NORMAL = 0,
        SUSPICIOUS = 1,
        HIGH_RISK = 2,
        CRITICAL = 3
    };
    
    ThreatLevel threat_level;
    double risk_score;          // 综合风险评分 [0.0, 1.0]
    double confidence;          // 检测置信度 [0.0, 1.0]
    std::string attack_type;    // 攻击类型
    std::vector<std::string> indicators;  // 威胁指标
    uint64_t detection_timestamp;
    FlowStatistics flow_stats;
};

/**
 * SIMD优化的特征提取器
 * 利用AVX2指令集并行计算多个统计特征
 */
class SIMDFeatureExtractor {
private:
    static constexpr size_t SIMD_WIDTH = 8;  // AVX2支持8个double
    alignas(32) double temp_buffer_[SIMD_WIDTH];

public:
    /**
     * SIMD优化的统计特征计算
     * 并行计算均值、方差、最大值、最小值等统计量
     */
    void extract_statistical_features(
        const std::vector<double>& data,
        double& mean, double& variance, 
        double& max_val, double& min_val);
    
    /**
     * 向量化的熵值计算
     * 高效计算IP地址分布熵
     */
    double calculate_entropy_simd(const std::vector<uint32_t>& ip_counts);
    
    /**
     * 批量包大小分析
     * 并行分析数据包大小分布特征
     */
    void analyze_packet_sizes_batch(
        const std::vector<uint32_t>& packet_sizes,
        FlowStatistics& stats);
};

/**
 * 自适应基线管理器
 * 基于滑动窗口维护流量基线，支持业务高峰期自适应
 */
class AdaptiveBaselineManager {
private:
    struct BaselineMetrics {
        double mean;
        double std_dev;
        double mad;  // 中位绝对偏差
        uint64_t sample_count;
        uint64_t last_update;
    };
    
    std::unordered_map<std::string, BaselineMetrics> baseline_cache_;
    static constexpr double UPDATE_ALPHA = 0.1;  // EMA更新系数
    static constexpr uint32_t MIN_SAMPLES = 100;  // 最小样本数
    
public:
    /**
     * 更新流量基线
     * 使用指数移动平均算法实时更新基线统计
     */
    void update_baseline(const std::string& metric_name, double value);
    
    /**
     * 获取异常分数
     * 计算当前值相对于历史基线的Z-Score
     */
    double get_anomaly_score(const std::string& metric_name, double current_value);
    
    /**
     * 业务高峰期基线调整
     * 根据时间模式调整检测阈值
     */
    void adjust_for_business_peak(int hour_of_day, int day_of_week);
};

/**
 * 多维特征融合检测器
 * 实现基于数学建模的多维度DDoS检测算法
 */
class MultiDimensionalDetector {
private:
    std::unique_ptr<SIMDFeatureExtractor> feature_extractor_;
    std::unique_ptr<AdaptiveBaselineManager> baseline_manager_;
    
    // 检测阈值配置
    struct DetectionThresholds {
        double z_score_threshold = 3.0;      // Z-Score异常阈值
        double entropy_min_normal = 4.0;     // 正常熵值下限
        double entropy_max_normal = 7.0;     // 正常熵值上限
        double small_packet_threshold = 0.3; // 小包比例阈值
        double syn_flood_threshold = 0.8;    // SYN洪水阈值
        double risk_fusion_weights[8] = {     // 多维特征融合权重
            0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.04, 0.03
        };
    } thresholds_;

public:
    MultiDimensionalDetector();
    ~MultiDimensionalDetector() = default;
    
    /**
     * 核心检测算法
     * 实现多维特征融合的DDoS检测模型
     * 
     * 数学模型：
     * Risk_Score = Σ(w_i × normalized_feature_i) × confidence_factor
     */
    DDoSDetectionResult detect_ddos_attack(
        const std::vector<NetworkPacket>& packet_batch);
    
    /**
     * 实时流量分析
     * 从网络包批次中提取统计特征
     */
    FlowStatistics extract_flow_features(
        const std::vector<NetworkPacket>& packets);
    
    /**
     * 威胁等级评估
     * 根据风险评分确定威胁等级
     */
    DDoSDetectionResult::ThreatLevel assess_threat_level(
        double risk_score, double confidence);

private:
    /**
     * 计算源IP熵值
     * 用于检测分布式攻击特征
     */
    double calculate_source_ip_entropy(
        const std::vector<NetworkPacket>& packets);
    
    /**
     * 检测SYN洪水攻击
     * 分析TCP标志位分布
     */
    double detect_syn_flood_pattern(
        const std::vector<NetworkPacket>& packets);
    
    /**
     * 协议异常检测
     * 分析协议分布异常
     */
    double detect_protocol_anomaly(
        const std::vector<NetworkPacket>& packets);
    
    /**
     * 时间序列分析
     * 检测流量的时间模式异常
     */
    double analyze_temporal_patterns(
        const std::vector<NetworkPacket>& packets);
};

/**
 * 高性能包处理器
 * 利用零拷贝和内存池技术实现高吞吐量处理
 */
class HighPerformancePacketProcessor {
private:
    // 环形缓冲区实现零拷贝
    template<typename T, size_t Size>
    class RingBuffer {
    private:
        alignas(64) T buffer_[Size];  // 缓存行对齐
        std::atomic<size_t> read_pos_{0};
        std::atomic<size_t> write_pos_{0};
        
    public:
        bool push(const T& item);
        bool pop(T& item);
        size_t size() const;
        bool empty() const;
    };
    
    RingBuffer<NetworkPacket, 1024*1024> packet_buffer_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    
    MultiDimensionalDetector detector_;

public:
    HighPerformancePacketProcessor(size_t num_threads = std::thread::hardware_concurrency());
    ~HighPerformancePacketProcessor();
    
    /**
     * 启动高性能处理
     * 多线程并行处理数据包
     */
    void start_processing();
    
    /**
     * 停止处理
     */
    void stop_processing();
    
    /**
     * 提交数据包批次
     * 零拷贝方式提交待处理数据包
     */
    bool submit_packet_batch(const std::vector<NetworkPacket>& packets);
    
    /**
     * 获取检测结果
     */
    bool get_detection_result(DDoSDetectionResult& result);

private:
    /**
     * 工作线程主循环
     */
    void worker_thread_main();
    
    /**
     * 批处理优化
     * 合并小批次以提高处理效率
     */
    void process_packet_batches();
};

/**
 * DDoS防护策略执行器
 * 根据检测结果执行相应的防护措施
 */
class DefenseStrategyExecutor {
public:
    enum DefenseAction {
        MONITOR_ONLY = 0,
        RATE_LIMIT = 1,
        TEMPORARY_BLOCK = 2,
        PERMANENT_BLOCK = 3,
        TRAFFIC_DIVERSION = 4
    };
    
    struct DefenseStrategy {
        DefenseAction action;
        uint32_t duration_seconds;
        double rate_limit_ratio;
        std::vector<std::string> target_ips;
        std::string reason;
    };

    /**
     * 生成防护策略
     * 根据威胁等级和攻击类型生成相应策略
     */
    DefenseStrategy generate_defense_strategy(
        const DDoSDetectionResult& detection_result);
    
    /**
     * 执行防护措施
     * 与网络设备或云服务API接口对接
     */
    bool execute_defense_action(const DefenseStrategy& strategy);
    
    /**
     * 策略效果评估
     * 评估防护策略的执行效果
     */
    double evaluate_defense_effectiveness(
        const DefenseStrategy& strategy,
        const std::vector<DDoSDetectionResult>& post_defense_results);
};

/**
 * 统一DDoS检测系统
 * 整合所有组件的主控制器
 */
class UnifiedDDoSDetectionSystem {
private:
    std::unique_ptr<HighPerformancePacketProcessor> packet_processor_;
    std::unique_ptr<DefenseStrategyExecutor> strategy_executor_;
    
    // 性能监控
    struct PerformanceMetrics {
        uint64_t packets_processed = 0;
        uint64_t attacks_detected = 0;
        uint64_t false_positives = 0;
        double avg_processing_time = 0.0;
        double system_throughput = 0.0;
    };
    
    // 内部原子统计变量
    mutable std::atomic<uint64_t> packets_processed_atomic_{0};
    mutable std::atomic<uint64_t> attacks_detected_atomic_{0};
    mutable std::atomic<uint64_t> false_positives_atomic_{0};
    mutable std::atomic<double> avg_processing_time_atomic_{0.0};
    mutable std::atomic<double> system_throughput_atomic_{0.0};

public:
    UnifiedDDoSDetectionSystem();
    ~UnifiedDDoSDetectionSystem() = default;
    
    /**
     * 系统初始化
     */
    bool initialize();
    
    /**
     * 启动检测系统
     */
    bool start_detection();
    
    /**
     * 停止检测系统
     */
    void stop_detection();
    
    /**
     * 处理实时流量
     * 主要的外部接口
     */
    bool process_network_traffic(const std::vector<NetworkPacket>& packets);
    
    /**
     * 获取系统状态
     */
    std::string get_system_status() const;
    
    /**
     * 获取性能指标
     * 返回性能指标的副本
     */
    PerformanceMetrics get_performance_metrics() const {
        PerformanceMetrics result;
        result.packets_processed = packets_processed_atomic_.load();
        result.attacks_detected = attacks_detected_atomic_.load();
        result.false_positives = false_positives_atomic_.load();
        result.avg_processing_time = avg_processing_time_atomic_.load();
        result.system_throughput = system_throughput_atomic_.load();
        return result;
    }
    
    /**
     * 配置更新
     * 支持运行时配置热更新
     */
    bool update_configuration(const std::string& config_json);
    
    /**
     * 健康检查
     */
    bool health_check() const;
};

} // namespace ddos_defense