/**
 * VPN检测系统 - 高性能数据包处理器实现
 */

#include "packet_processor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <iostream>
#include <sstream>
// #include <json/json.h>  // 需要安装jsoncpp库 - 暂时注释掉

namespace vpn_detection {

// FeatureExtractor 实现
void FeatureExtractor::update_histogram(std::array<uint32_t, HISTOGRAM_BINS>& histogram,
                                       double value, double max_value) const {
    if (value < 0) value = 0;
    if (value > max_value) value = max_value;
    
    size_t bin = static_cast<size_t>((value / max_value) * (HISTOGRAM_BINS - 1));
    if (bin >= HISTOGRAM_BINS) bin = HISTOGRAM_BINS - 1;
    
    histogram[bin]++;
}

double FeatureExtractor::calculate_entropy(const std::array<uint32_t, HISTOGRAM_BINS>& histogram) const {
    uint32_t total = std::accumulate(histogram.begin(), histogram.end(), 0u);
    if (total == 0) return 0.0;
    
    double entropy = 0.0;
    for (uint32_t count : histogram) {
        if (count > 0) {
            double prob = static_cast<double>(count) / total;
            entropy -= prob * std::log2(prob);
        }
    }
    return entropy;
}

double FeatureExtractor::calculate_burstiness(const std::vector<double>& iats) const {
    if (iats.size() < 2) return 0.0;
    
    double mean = std::accumulate(iats.begin(), iats.end(), 0.0) / iats.size();
    if (mean <= 0) return 0.0;
    
    double variance = 0.0;
    for (double iat : iats) {
        variance += (iat - mean) * (iat - mean);
    }
    variance /= iats.size();
    
    return variance / (mean * mean);
}

BiDirectionalFeatures FeatureExtractor::extract_features(const std::vector<Packet>& packets) const {
    BiDirectionalFeatures features;
    
    if (packets.empty()) return features;
    
    std::vector<Packet> up_packets, down_packets;
    std::vector<double> iats, up_iats, down_iats;
    
    // 分离上行和下行数据包
    for (const auto& packet : packets) {
        if (packet.direction == 0) {  // up
            up_packets.push_back(packet);
        } else {  // down
            down_packets.push_back(packet);
        }
        
        features.total_packets++;
        features.total_bytes += packet.packet_size;
        
        if (packet.direction == 0) {
            features.up_packets++;
            features.up_bytes += packet.packet_size;
        } else {
            features.down_packets++;
            features.down_bytes += packet.packet_size;
        }
    }
    
    // 计算包长直方图
    for (const auto& packet : packets) {
        update_histogram(features.packet_length_histogram, 
                        packet.packet_size, MAX_PACKET_SIZE);
    }
    
    for (const auto& packet : up_packets) {
        update_histogram(features.up_packet_length_histogram,
                        packet.packet_size, MAX_PACKET_SIZE);
    }
    
    for (const auto& packet : down_packets) {
        update_histogram(features.down_packet_length_histogram,
                        packet.packet_size, MAX_PACKET_SIZE);
    }
    
    // 计算IAT
    auto calculate_iats = [](const std::vector<Packet>& pkts) -> std::vector<double> {
        std::vector<double> result;
        for (size_t i = 1; i < pkts.size(); ++i) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                pkts[i].timestamp - pkts[i-1].timestamp);
            result.push_back(duration.count() / 1000.0);  // 转换为毫秒
        }
        return result;
    };
    
    iats = calculate_iats(packets);
    up_iats = calculate_iats(up_packets);
    down_iats = calculate_iats(down_packets);
    
    // 计算IAT直方图
    for (double iat : iats) {
        update_histogram(features.iat_histogram, iat, MAX_IAT_MS);
    }
    
    for (double iat : up_iats) {
        update_histogram(features.up_iat_histogram, iat, MAX_IAT_MS);
    }
    
    for (double iat : down_iats) {
        update_histogram(features.down_iat_histogram, iat, MAX_IAT_MS);
    }
    
    // 计算方向切换
    for (size_t i = 1; i < packets.size(); ++i) {
        if (packets[i].direction != packets[i-1].direction) {
            features.direction_switches++;
        }
    }
    
    // 计算时间统计
    if (packets.size() > 1) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            packets.back().timestamp - packets.front().timestamp);
        features.flow_duration_ms = duration.count() / 1000.0;
    }
    
    if (!iats.empty()) {
        features.avg_iat_ms = std::accumulate(iats.begin(), iats.end(), 0.0) / iats.size();
        
        double sum_sq_diff = 0.0;
        for (double iat : iats) {
            sum_sq_diff += (iat - features.avg_iat_ms) * (iat - features.avg_iat_ms);
        }
        features.iat_variance = sum_sq_diff / iats.size();
    }
    
    // 计算突发度和熵
    features.burstiness = calculate_burstiness(iats);
    features.packet_size_entropy = calculate_entropy(features.packet_length_histogram);
    
    return features;
}

// RulePreFilter 实现
RulePreFilter::RuleFilterResult RulePreFilter::check_indicators(
    const std::vector<Packet>& packets) const {
    
    RuleFilterResult result;
    
    for (const auto& packet : packets) {
        // 检查VPN端口
        for (uint16_t port : vpn_ports_) {
            if (packet.src_port == port || packet.dst_port == port) {
                result.has_vpn_ports = true;
                
                // 检查IKE/ESP
                if (port == 500 || port == 4500) {
                    result.has_ike_esp = true;
                }
                break;
            }
        }
        
        // 检查DTLS隧道 (UDP + 443端口)
        if (packet.protocol == 17 && (packet.src_port == 443 || packet.dst_port == 443)) {
            result.has_dtls_tunnel = true;
        }
    }
    
    // 决定是否继续检测
    result.should_continue = result.has_vpn_ports || result.has_ike_esp || result.has_dtls_tunnel;
    
    return result;
}

// RelativeEntropyFilter 实现
double RelativeEntropyFilter::calculate_kl_divergence(
    const std::array<uint32_t, 10>& p, const std::array<uint32_t, 10>& q) const {
    
    // 计算概率分布
    uint32_t sum_p = std::accumulate(p.begin(), p.end(), 0u);
    uint32_t sum_q = std::accumulate(q.begin(), q.end(), 0u);
    
    if (sum_p == 0 || sum_q == 0) return 0.0;
    
    double kl = 0.0;
    for (size_t i = 0; i < 10; ++i) {
        double prob_p = (static_cast<double>(p[i]) + 1e-10) / sum_p;
        double prob_q = (static_cast<double>(q[i]) + 1e-10) / sum_q;
        kl += prob_p * std::log(prob_p / prob_q);
    }
    
    return kl;
}

void RelativeEntropyFilter::set_baseline(const BiDirectionalFeatures& benign,
                                        const BiDirectionalFeatures& vpn) {
    baseline_benign_ = benign;
    baseline_vpn_ = vpn;
    has_baseline_ = true;
}

bool RelativeEntropyFilter::should_continue(const BiDirectionalFeatures& features) const {
    if (!has_baseline_) return true;
    
    // 计算与基线的KL散度
    double kl_packet_benign = calculate_kl_divergence(
        features.packet_length_histogram, baseline_benign_.packet_length_histogram);
    double kl_iat_benign = calculate_kl_divergence(
        features.iat_histogram, baseline_benign_.iat_histogram);
    
    double multi_kl = (kl_packet_benign + kl_iat_benign) / 2.0;
    
    return multi_kl > threshold_l_;
}

// SequenceModelInference 实现
std::vector<double> SequenceModelInference::prepare_sequence_input(
    const std::vector<Packet>& packets) const {
    
    std::vector<double> sequence;
    sequence.reserve(packets.size() * 3);  // size, direction, iat
    
    for (size_t i = 0; i < packets.size(); ++i) {
        // 包大小 (归一化到0-1)
        sequence.push_back(static_cast<double>(packets[i].packet_size) / 1500.0);
        
        // 方向
        sequence.push_back(static_cast<double>(packets[i].direction));
        
        // IAT (归一化)
        if (i > 0) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                packets[i].timestamp - packets[i-1].timestamp);
            double iat_ms = duration.count() / 1000.0;
            sequence.push_back(std::min(iat_ms / 1000.0, 1.0));  // 最大1秒
        } else {
            sequence.push_back(0.0);
        }
    }
    
    return sequence;
}

bool SequenceModelInference::load_model(const std::string& model_path) {
    // 简化的模型加载实现
    // 实际应用中应该加载真实的深度学习模型 (如ONNX, TensorFlow Lite等)
    
    // 模拟加载预训练权重
    model_weights_.cnn_weights = {{0.1, 0.3, 0.3, 0.3, 0.1}};  // 简化的CNN核
    model_weights_.lstm_weights = {{0.5, 0.3, 0.2}};           // 简化的LSTM权重
    model_weights_.dense_weights = {0.7, 0.3};                 // 简化的全连接层
    
    model_loaded_ = true;
    return true;
}

double SequenceModelInference::cnn_forward(const std::vector<double>& input) const {
    if (input.empty() || model_weights_.cnn_weights.empty()) return 0.5;
    
    // 简化的1D卷积
    const auto& kernel = model_weights_.cnn_weights[0];
    double result = 0.0;
    
    for (size_t i = 0; i < std::min(input.size(), kernel.size()); ++i) {
        result += input[i] * kernel[i];
    }
    
    // ReLU激活
    return std::max(0.0, result);
}

double SequenceModelInference::lstm_forward(const std::vector<double>& cnn_output) const {
    if (cnn_output.empty() || model_weights_.lstm_weights.empty()) return 0.5;
    
    // 简化的LSTM (实际上是加权平均)
    const auto& weights = model_weights_.lstm_weights[0];
    double result = 0.0;
    
    for (size_t i = 0; i < std::min(cnn_output.size(), weights.size()); ++i) {
        result += cnn_output[i] * weights[i];
    }
    
    // Sigmoid激活
    return 1.0 / (1.0 + std::exp(-result));
}

double SequenceModelInference::predict(const std::vector<Packet>& packets) const {
    if (!model_loaded_ || packets.empty()) return 0.5;
    
    auto sequence_input = prepare_sequence_input(packets);
    auto cnn_output = cnn_forward(sequence_input);
    
    std::vector<double> cnn_vector = {cnn_output};
    return lstm_forward(cnn_vector);
}

// MultiWindowFusion 实现
void MultiWindowFusion::add_result(bool is_vpn, double confidence, const std::string& stage) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    DetectionResult result;
    result.is_vpn = is_vpn;
    result.confidence = confidence;
    result.timestamp = std::chrono::high_resolution_clock::now();
    result.stage = stage;
    
    window_results_.push_back(result);
    
    // 保持窗口大小
    while (window_results_.size() > max_windows_) {
        window_results_.pop_front();
    }
}

std::pair<bool, double> MultiWindowFusion::get_fused_result() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (window_results_.empty()) return {false, 0.0};
    
    // 置信度聚合策略
    double vpn_confidence_sum = 0.0;
    double normal_confidence_sum = 0.0;
    int vpn_count = 0;
    int normal_count = 0;
    
    for (const auto& result : window_results_) {
        if (result.is_vpn) {
            vpn_confidence_sum += result.confidence;
            vpn_count++;
        } else {
            normal_confidence_sum += result.confidence;
            normal_count++;
        }
    }
    
    double avg_vpn_confidence = vpn_count > 0 ? vpn_confidence_sum / vpn_count : 0.0;
    double avg_normal_confidence = normal_count > 0 ? normal_confidence_sum / normal_count : 0.0;
    
    bool is_vpn = avg_vpn_confidence > avg_normal_confidence;
    double final_confidence = is_vpn ? avg_vpn_confidence : avg_normal_confidence;
    
    return {is_vpn, final_confidence};
}

// PacketProcessor 实现
PacketProcessor::PacketProcessor() {
    sliding_window_ = std::make_unique<SlidingWindow<Packet>>();
    feature_extractor_ = std::make_unique<FeatureExtractor>();
    rule_filter_ = std::make_unique<RulePreFilter>();
    entropy_filter_ = std::make_unique<RelativeEntropyFilter>();
    sequence_model_ = std::make_unique<SequenceModelInference>();
    multi_window_fusion_ = std::make_unique<MultiWindowFusion>();
}

PacketProcessor::~PacketProcessor() {
    stop();
}

bool PacketProcessor::initialize(const std::string& config_path) {
    // 简化的配置加载实现，不依赖外部JSON库
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "无法打开配置文件: " << config_path << std::endl;
        // 使用默认配置继续运行
        return sequence_model_->load_model("default_model");
    }
    
    std::string line;
    while (std::getline(config_file, line)) {
        // 简单的key=value解析
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            if (key == "entropy_threshold") {
                double threshold = std::stod(value);
                // entropy_filter_->set_threshold(threshold); // 需要实现这个方法
            } else if (key == "model_path") {
                if (!sequence_model_->load_model(value)) {
                    std::cerr << "加载模型失败: " << value << std::endl;
                    return false;
                }
            }
        }
    }
    
    return true;
}

void PacketProcessor::start() {
    running_ = true;
    start_time_ = std::chrono::high_resolution_clock::now();
    processing_thread_ = std::thread(&PacketProcessor::processing_loop, this);
}

void PacketProcessor::stop() {
    running_ = false;
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

void PacketProcessor::add_packet(const Packet& packet) {
    sliding_window_->add_packet(packet);
    packets_processed_++;
    
    // 更新流信息
    std::lock_guard<std::mutex> lock(flows_mutex_);
    FlowKey key = extract_flow_key(packet);
    active_flows_[key].push_back(packet);
    
    // 限制每个流的数据包数量
    if (active_flows_[key].size() > 1000) {
        active_flows_[key].erase(active_flows_[key].begin());
    }
}

FlowKey PacketProcessor::extract_flow_key(const Packet& packet) const {
    FlowKey key;
    key.src_ip = packet.src_ip;
    key.dst_ip = packet.dst_ip;
    key.src_port = packet.src_port;
    key.dst_port = packet.dst_port;
    key.protocol = packet.protocol;
    return key;
}

void PacketProcessor::processing_loop() {
    while (running_) {
        if (sliding_window_->should_process()) {
            auto window_data = sliding_window_->get_window_data();
            if (!window_data.empty()) {
                auto result = process_window(window_data);
                
                // 添加到多窗口融合
                multi_window_fusion_->add_result(result.is_vpn, result.confidence, 
                                                result.detection_stage);
                
                if (result.is_vpn) {
                    flows_detected_++;
                }
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

PacketProcessor::ProcessingResult PacketProcessor::process_window(
    const std::vector<Packet>& packets) {
    
    ProcessingResult result;
    result.is_vpn = false;
    result.confidence = 0.0;
    result.detection_stage = "Unknown";
    
    // Stage A: 规则预筛
    auto rule_result = rule_filter_->check_indicators(packets);
    if (!rule_result.should_continue) {
        result.detection_stage = "RulePreFilter";
        result.confidence = 0.9;
        return result;
    }
    
    // 提取特征
    result.features = feature_extractor_->extract_features(packets);
    
    // Stage B: 相对熵过滤
    if (!entropy_filter_->should_continue(result.features)) {
        result.detection_stage = "RelativeEntropyFilter";
        result.confidence = 0.7;
        return result;
    }
    
    // Stage C: 序列模型精判
    double sequence_score = sequence_model_->predict(packets);
    result.is_vpn = sequence_score > 0.5;
    result.confidence = result.is_vpn ? sequence_score : 1.0 - sequence_score;
    result.detection_stage = "SequenceModel";
    
    return result;
}

PacketProcessor::PerformanceStats PacketProcessor::get_performance_stats() const {
    PerformanceStats stats;
    stats.packets_processed = packets_processed_.load();
    stats.flows_detected = flows_detected_.load();
    
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - start_time_);
    stats.uptime = duration;
    
    if (duration.count() > 0) {
        stats.processing_rate_pps = static_cast<double>(stats.packets_processed) / 
                                   (duration.count() / 1000.0);
        
        // 假设每个包平均1KB，计算Gbps
        stats.throughput_gbps = (stats.processing_rate_pps * 1024 * 8) / 1e9;
    } else {
        stats.processing_rate_pps = 0.0;
        stats.throughput_gbps = 0.0;
    }
    
    return stats;
}

bool PacketProcessor::load_model(const std::string& model_path) {
    return sequence_model_->load_model(model_path);
}

void PacketProcessor::set_entropy_threshold(double threshold) {
    // 需要添加到RelativeEntropyFilter类中
    // entropy_filter_->set_threshold(threshold);
}

void PacketProcessor::update_baseline(const BiDirectionalFeatures& benign,
                                     const BiDirectionalFeatures& vpn) {
    entropy_filter_->set_baseline(benign, vpn);
}

// 工厂函数
std::unique_ptr<PacketProcessor> create_packet_processor() {
    return std::make_unique<PacketProcessor>();
}

} // namespace vpn_detection