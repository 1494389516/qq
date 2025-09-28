#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <thread>
#include <functional>

// 条件编译SIMD支持
#ifdef __x86_64__ || __i386__
#include <immintrin.h>  // SIMD支持
#define SIMD_AVAILABLE
#endif

namespace unified_risk_control {

/**
 * 统一风控检测引擎 - C++实现
 * 整合DDoS防护、VPN检测、恶意行为识别
 * 基于风控算法专家的理论框架
 * 
 * 核心优势：
 * 1. SIMD指令集优化的特征提取
 * 2. 零拷贝内存管理
 * 3. 微秒级延迟处理
 * 4. 统一威胁检测接口
 */

struct NetworkPacket {
    uint64_t timestamp;
    std::string src_ip;
    std::string dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    std::string protocol;
    uint32_t size;
    std::string direction;
    uint32_t payload_size;
};

struct BehaviorFeatures {
    double order_requests;
    double payment_success;
    double product_pv;
    double risk_hits;
    double payment_success_rate;
    double risk_hit_rate;
    double pv_order_ratio;
    double time_features[4];  // hour, day, weekend, night
    double entropy_features[2];  // source, ip
};

struct DetectionResult {
    bool is_threat;
    std::string threat_type;
    double confidence;
    std::string detection_stage;
    std::unordered_map<std::string, double> features;
    uint64_t timestamp;
};

class SIMDFeatureExtractor {
private:
    static constexpr size_t SIMD_WIDTH = 8;  // AVX2支持
    
public:
    /**
     * SIMD优化的特征向量计算
     * 使用AVX2指令集并行处理多个特征
     */
    void extract_features_simd(const std::vector<NetworkPacket>& packets,
                              BehaviorFeatures& features);
    
    /**
     * 统计特征快速计算
     * 零拷贝实现，直接在原始数据上操作
     */
    void calculate_statistics_zero_copy(const double* data, size_t size,
                                       double& mean, double& std_dev);
};

class HighPerformanceBehaviorClassifier {
private:
    std::unique_ptr<SIMDFeatureExtractor> feature_extractor_;
    
    // 模型权重矩阵（预训练好的随机森林权重）
    std::vector<std::vector<double>> tree_weights_;
    std::vector<double> tree_biases_;
    
    // 异常检测阈值
    double normal_threshold_ = 0.3;
    double suspicious_threshold_ = 0.7;
    double critical_threshold_ = 0.9;

public:
    HighPerformanceBehaviorClassifier();
    ~HighPerformanceBehaviorClassifier() = default;
    
    /**
     * 实时行为分类 - 微秒级响应
     * 
     * @param features 输入特征向量
     * @return 检测结果，包含威胁类型和置信度
     */
    DetectionResult classify_behavior(const BehaviorFeatures& features);
    
    /**
     * 批量检测优化
     * 利用SIMD并行处理多个样本
     */
    std::vector<DetectionResult> batch_classify(
        const std::vector<BehaviorFeatures>& batch_features);
    
    /**
     * 模型热更新
     * 支持运行时更新模型参数，无需重启
     */
    bool update_model_weights(const std::string& model_file_path);
};

class VPNTunnelDetector {
private:
    // VPN检测的特征端口
    std::vector<uint16_t> vpn_ports_ = {500, 4500, 1194, 51820};
    
    // 四阶段级联检测器
    struct CascadeStage {
        std::string name;
        double threshold;
        std::function<double(const std::vector<NetworkPacket>&)> detector;
    };
    
    std::vector<CascadeStage> cascade_stages_;

public:
    VPNTunnelDetector();
    
    /**
     * 四阶段级联VPN检测算法
     * 
     * 阶段A: 规则预筛 - 协议和端口检测
     * 阶段B: 相对熵过滤 - 统计特征异常检测  
     * 阶段C: 序列模型精判 - 深度学习模型
     * 阶段D: 多窗融合 - 时间窗口投票机制
     */
    DetectionResult detect_vpn_cascade(const std::vector<NetworkPacket>& packets);

private:
    double stage_a_rule_prefilter(const std::vector<NetworkPacket>& packets);
    double stage_b_entropy_filter(const std::vector<NetworkPacket>& packets);
    double stage_c_sequence_model(const std::vector<NetworkPacket>& packets);
    double stage_d_window_fusion(const std::vector<NetworkPacket>& packets);
};

class DDoSDefenseEngine {
private:
    struct AttackPattern {
        double packets_per_second;
        double small_packet_ratio;
        double direction_bias;
        double port_concentration;
    };
    
    // 自适应阈值 - 基于历史基线动态调整
    double adaptive_pps_threshold_ = 1000.0;
    double baseline_update_rate_ = 0.1;

public:
    /**
     * DDoS攻击检测与防护
     * 
     * 检测算法：
     * 1. 流量统计异常检测
     * 2. 包大小分布分析  
     * 3. 来源IP熵值计算
     * 4. 时间序列模式识别
     */
    DetectionResult detect_ddos_attack(const std::vector<NetworkPacket>& packets);
    
    /**
     * 自适应防护策略
     * 根据攻击强度动态调整防护等级
     */
    std::string generate_defense_strategy(const DetectionResult& detection);

private:
    AttackPattern extract_attack_pattern(const std::vector<NetworkPacket>& packets);
    double calculate_entropy(const std::vector<std::string>& ip_addresses);
    void update_baseline_statistics(const AttackPattern& pattern);
};

class UnifiedCppDetector {
private:
    std::unique_ptr<HighPerformanceBehaviorClassifier> behavior_classifier_;
    std::unique_ptr<VPNTunnelDetector> vpn_detector_;
    std::unique_ptr<DDoSDefenseEngine> ddos_engine_;
    
    // 性能监控
    struct PerformanceMetrics {
        std::chrono::microseconds avg_detection_time{0};
        uint64_t total_processed_packets = 0;
        uint64_t threats_detected = 0;
        double throughput_per_second = 0.0;
    };
    
    PerformanceMetrics metrics_;

public:
    UnifiedCppDetector();
    ~UnifiedCppDetector() = default;
    
    /**
     * 统一检测接口
     * 整合行为分析、VPN检测、DDoS防护
     */
    std::vector<DetectionResult> unified_detection(
        const std::vector<NetworkPacket>& packets,
        const BehaviorFeatures& business_features);
    
    /**
     * 获取性能指标
     */
    PerformanceMetrics get_performance_metrics() const { return metrics_; }
    
    /**
     * 健康检查接口
     */
    bool health_check() const;
    
    /**
     * 配置热更新
     */
    bool reload_config(const std::string& config_file);
};

// REST API接口类
class RestApiServer {
private:
    std::unique_ptr<UnifiedCppDetector> detector_;
    uint16_t port_;

public:
    RestApiServer(uint16_t port = 8001);
    
    /**
     * 启动HTTP服务器
     * 提供RESTful API接口供Python控制层调用
     */
    void start_server();
    
    /**
     * 停止服务器
     */
    void stop_server();

private:
    void setup_routes();
    std::string handle_detection_request(const std::string& json_data);
    std::string handle_health_check();
    std::string handle_metrics_request();
};

} // namespace risk_detection