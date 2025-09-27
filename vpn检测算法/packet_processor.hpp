/**
 * VPN检测系统 - 高性能数据包处理器
 * 基于C++实现10Gbps级别的实时数据包捕获与特征提取
 *
 * @author: <NAME>
 * @date: 2023/07/05
 */

 
#ifndef PACKET_PROCESSOR_HPP
#define PACKET_PROCESSOR_HPP

#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <unordered_map>
#include <deque>
#include <array>

namespace vpn_detection {

// 数据包结构
struct Packet {
    std::chrono::high_resolution_clock::time_point timestamp;
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t protocol;  // TCP=6, UDP=17
    uint16_t packet_size;
    uint16_t payload_size;
    uint8_t direction;  // 0=up, 1=down
    
    // TLS相关信息
    bool has_tls;
    uint8_t tls_version;
    uint16_t cipher_suite;
    
    Packet() = default;
    Packet(const Packet&) = default;
    Packet& operator=(const Packet&) = default;
};

// 流标识符
struct FlowKey {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t protocol;
    
    bool operator==(const FlowKey& other) const {
        return src_ip == other.src_ip && dst_ip == other.dst_ip &&
               src_port == other.src_port && dst_port == other.dst_port &&
               protocol == other.protocol;
    }
};

// FlowKey哈希函数
struct FlowKeyHash {
    std::size_t operator()(const FlowKey& key) const {
        return std::hash<uint64_t>()((static_cast<uint64_t>(key.src_ip) << 32) | key.dst_ip) ^
               std::hash<uint32_t>()((static_cast<uint32_t>(key.src_port) << 16) | key.dst_port) ^
               std::hash<uint8_t>()(key.protocol);
    }
};

// 双向统计特征
struct BiDirectionalFeatures {
    // 包长直方图 (10个bin)
    std::array<uint32_t, 10> packet_length_histogram{};
    std::array<uint32_t, 10> up_packet_length_histogram{};
    std::array<uint32_t, 10> down_packet_length_histogram{};
    
    // IAT直方图 (10个bin)
    std::array<uint32_t, 10> iat_histogram{};
    std::array<uint32_t, 10> up_iat_histogram{};
    std::array<uint32_t, 10> down_iat_histogram{};
    
    // 方向统计
    uint32_t total_packets = 0;
    uint32_t up_packets = 0;
    uint32_t down_packets = 0;
    uint32_t direction_switches = 0;
    
    // 时间统计
    double flow_duration_ms = 0.0;
    double avg_iat_ms = 0.0;
    double iat_variance = 0.0;
    
    // 突发度和熵
    double burstiness = 0.0;
    double packet_size_entropy = 0.0;
    
    // 总字节数
    uint64_t total_bytes = 0;
    uint64_t up_bytes = 0;
    uint64_t down_bytes = 0;
};

// 滑动窗口
template<typename T>
class SlidingWindow {
private:
    std::deque<T> data_;
    std::chrono::milliseconds window_size_;
    std::chrono::milliseconds step_size_;
    std::chrono::high_resolution_clock::time_point last_step_time_;
    mutable std::mutex mutex_;

public:
    SlidingWindow(std::chrono::milliseconds window_size = std::chrono::milliseconds(5000),
                  std::chrono::milliseconds step_size = std::chrono::milliseconds(2000))
        : window_size_(window_size), step_size_(step_size) {
        last_step_time_ = std::chrono::high_resolution_clock::now();
    }
    
    void add_packet(const T& packet) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto current_time = std::chrono::high_resolution_clock::now();
        
        data_.push_back(packet);
        
        // 清理过期数据
        auto cutoff_time = current_time - window_size_;
        while (!data_.empty() && 
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   current_time - data_.front().timestamp).count() > window_size_.count()) {
            data_.pop_front();
        }
    }
    
    bool should_process() {
        auto current_time = std::chrono::high_resolution_clock::now();
        if (current_time - last_step_time_ >= step_size_) {
            last_step_time_ = current_time;
            return true;
        }
        return false;
    }
    
    std::vector<T> get_window_data() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::vector<T>(data_.begin(), data_.end());
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }
};

// 特征提取器
class FeatureExtractor {
private:
    static constexpr size_t HISTOGRAM_BINS = 10;
    static constexpr uint16_t MAX_PACKET_SIZE = 1500;
    static constexpr double MAX_IAT_MS = 1000.0;
    
    void update_histogram(std::array<uint32_t, HISTOGRAM_BINS>& histogram,
                         double value, double max_value) const;
    
    double calculate_entropy(const std::array<uint32_t, HISTOGRAM_BINS>& histogram) const;
    
    double calculate_burstiness(const std::vector<double>& iats) const;

public:
    BiDirectionalFeatures extract_features(const std::vector<Packet>& packets) const;
};

// 规则预筛选器 (Stage A)
class RulePreFilter {
private:
    std::vector<uint16_t> vpn_ports_ = {500, 1194, 4500, 51820};  // IKE, OpenVPN, ESP, WireGuard
    
public:
    struct RuleFilterResult {
        bool has_vpn_ports = false;
        bool has_ike_esp = false;
        bool has_dtls_tunnel = false;
        bool should_continue = false;
    };
    
    RuleFilterResult check_indicators(const std::vector<Packet>& packets) const;
};

// 相对熵过滤器 (Stage B)
class RelativeEntropyFilter {
private:
    double threshold_l_;
    BiDirectionalFeatures baseline_benign_;
    BiDirectionalFeatures baseline_vpn_;
    bool has_baseline_ = false;
    
    double calculate_kl_divergence(const std::array<uint32_t, 10>& p,
                                  const std::array<uint32_t, 10>& q) const;

public:
    explicit RelativeEntropyFilter(double threshold_l = 0.1) : threshold_l_(threshold_l) {}
    
    void set_baseline(const BiDirectionalFeatures& benign, const BiDirectionalFeatures& vpn);
    
    bool should_continue(const BiDirectionalFeatures& features) const;
};

// 序列模型推理器 (Stage C)
class SequenceModelInference {
private:
    struct ModelWeights {
        std::vector<std::vector<double>> cnn_weights;
        std::vector<std::vector<double>> lstm_weights;
        std::vector<double> dense_weights;
    };
    
    ModelWeights model_weights_;
    bool model_loaded_ = false;
    
    std::vector<double> prepare_sequence_input(const std::vector<Packet>& packets) const;
    
    double cnn_forward(const std::vector<double>& input) const;
    
    double lstm_forward(const std::vector<double>& cnn_output) const;

public:
    bool load_model(const std::string& model_path);
    
    double predict(const std::vector<Packet>& packets) const;
};

// 多窗口融合器 (Stage D)
class MultiWindowFusion {
private:
    struct DetectionResult {
        bool is_vpn;
        double confidence;
        std::chrono::high_resolution_clock::time_point timestamp;
        std::string stage;
    };
    
    std::deque<DetectionResult> window_results_;
    size_t max_windows_;
    mutable std::mutex mutex_;

public:
    explicit MultiWindowFusion(size_t max_windows = 3) : max_windows_(max_windows) {}
    
    void add_result(bool is_vpn, double confidence, const std::string& stage);
    
    std::pair<bool, double> get_fused_result() const;
};

// 主要的数据包处理器
class PacketProcessor {
private:
    // 组件
    std::unique_ptr<SlidingWindow<Packet>> sliding_window_;
    std::unique_ptr<FeatureExtractor> feature_extractor_;
    std::unique_ptr<RulePreFilter> rule_filter_;
    std::unique_ptr<RelativeEntropyFilter> entropy_filter_;
    std::unique_ptr<SequenceModelInference> sequence_model_;
    std::unique_ptr<MultiWindowFusion> multi_window_fusion_;
    
    // 状态管理
    std::atomic<bool> running_{false};
    std::thread processing_thread_;
    
    // 性能统计
    std::atomic<uint64_t> packets_processed_{0};
    std::atomic<uint64_t> flows_detected_{0};
    std::chrono::high_resolution_clock::time_point start_time_;
    
    // 流管理
    std::unordered_map<FlowKey, std::vector<Packet>, FlowKeyHash> active_flows_;
    std::mutex flows_mutex_;
    
    void processing_loop();
    
    FlowKey extract_flow_key(const Packet& packet) const;
    
    struct ProcessingResult {
        bool is_vpn;
        double confidence;
        std::string detection_stage;
        BiDirectionalFeatures features;
    };
    
    ProcessingResult process_window(const std::vector<Packet>& packets);

public:
    PacketProcessor();
    ~PacketProcessor();
    
    // 禁止拷贝
    PacketProcessor(const PacketProcessor&) = delete;
    PacketProcessor& operator=(const PacketProcessor&) = delete;
    
    bool initialize(const std::string& config_path);
    
    void start();
    
    void stop();
    
    void add_packet(const Packet& packet);
    
    // 性能统计
    struct PerformanceStats {
        uint64_t packets_processed;
        uint64_t flows_detected;
        double processing_rate_pps;  // packets per second
        double throughput_gbps;
        std::chrono::milliseconds uptime;
    };
    
    PerformanceStats get_performance_stats() const;
    
    // 配置管理
    bool load_model(const std::string& model_path);
    
    void set_entropy_threshold(double threshold);
    
    void update_baseline(const BiDirectionalFeatures& benign, 
                        const BiDirectionalFeatures& vpn);
};

// 工厂函数
std::unique_ptr<PacketProcessor> create_packet_processor();

} // namespace vpn_detection

#endif // PACKET_PROCESSOR_HPP