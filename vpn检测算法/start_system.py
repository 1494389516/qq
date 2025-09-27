#!/usr/bin/env python3
"""
VPN检测系统启动脚本
简化版本 - 主要使用Python组件进行演示
"""

import time
import threading
import signal
import sys
from datetime import datetime
from vpn检测 import VPNDetectionSystem, generate_sample_packets
from config import get_default_config, create_production_config

class VPNDetectionServerManager:
    """VPN检测系统服务管理器"""
    
    def __init__(self):
        self.detection_system = None
        self.is_running = False
        self.config = get_default_config()
        
    def start_system(self):
        """启动VPN检测系统"""
        print("=" * 60)
        print("🚀 启动VPN检测系统")
        print("=" * 60)
        
        print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔧 系统架构: 多语言企业级解决方案")
        print("   - Python: AI检测引擎 (运行中)")
        print("   - C++: 高性能数据包处理器 (模拟)")
        print("   - Rust: 消息总线 (模拟)")  
        print("   - Go: API网关 (模拟)")
        print()
        
        # 初始化检测系统
        print("🔄 初始化检测组件...")
        self.detection_system = VPNDetectionSystem()
        
        # 启动检测系统
        print("▶️  启动实时检测服务...")
        self.detection_system.start_detection()
        self.is_running = True
        
        # 显示配置信息
        print("⚙️  系统配置:")
        window_size = self.config.sliding_window.window_size if self.config.sliding_window else 5.0
        step_size = self.config.sliding_window.step_size if self.config.sliding_window else 2.0
        confidence_threshold = self.config.detection_cascade.model_confidence_threshold if self.config.detection_cascade else 0.5
        histogram_bins = self.config.feature_extraction.histogram_bins if self.config.feature_extraction else 10
        print(f"   - 滑动窗口大小: {window_size}s")
        print(f"   - 步长: {step_size}s")
        print(f"   - 置信度阈值: {confidence_threshold}")
        print(f"   - 特征提取bins: {histogram_bins}")
        print()
        
        print("✅ VPN检测系统启动成功!")
        print("📊 系统正在实时监控网络流量...")
        print("💡 按 Ctrl+C 优雅停止系统")
        print("=" * 60)
        
        return True
        
    def stop_system(self):
        """停止VPN检测系统"""
        if self.is_running and self.detection_system:
            print("\n🛑 正在停止VPN检测系统...")
            self.detection_system.stop_detection()
            self.is_running = False
            print("✅ 系统已安全停止")
            
    def show_real_time_status(self):
        """显示实时状态"""
        if not self.is_running:
            return
            
        try:
            # 获取检测结果
            if self.detection_system:
                results = self.detection_system.get_detection_results()
            else:
                results = []
            
            if results:
                print(f"\n📈 检测结果更新 [{datetime.now().strftime('%H:%M:%S')}]:")
                for i, result in enumerate(results[-3:]):  # 显示最近3个结果
                    status_icon = "🔴" if result.is_vpn else "🟢"
                    print(f"   {status_icon} 流 {result.flow_id}: "
                          f"{'VPN' if result.is_vpn else '正常'} "
                          f"(置信度: {result.confidence:.2f}, "
                          f"阶段: {result.detection_stage})")
                          
        except Exception as e:
            print(f"⚠️  状态更新错误: {e}")
            
    def run_demo_traffic(self):
        """运行演示流量"""
        print("\n🎯 生成演示流量进行检测...")
        
        # 生成不同类型的流量
        demo_scenarios = [
            ("正常Web浏览", generate_sample_packets(20, is_vpn=False)),
            ("VPN加密流量", generate_sample_packets(15, is_vpn=True)),
            ("混合流量", generate_sample_packets(25, is_vpn=False) + generate_sample_packets(10, is_vpn=True))
        ]
        
        for scenario_name, packets in demo_scenarios:
            print(f"\n📦 测试场景: {scenario_name} ({len(packets)} 个数据包)")
            
            # 模拟实时数据包到达
            for packet in packets:
                if not self.is_running or not self.detection_system:
                    break
                self.detection_system.sliding_window.add_packet(packet)
                time.sleep(0.05)  # 模拟网络延迟
                
                # 检查是否需要处理
                if self.detection_system.sliding_window.should_process():
                    window_data = self.detection_system.sliding_window.get_window_data()
                    if window_data:
                        result = self.detection_system._process_window(window_data)
                        if result:
                            self.detection_system.message_bus.publish("detection_results", result)
            
            # 显示结果
            time.sleep(0.5)
            self.show_real_time_status()
            time.sleep(2)
            
    def run_monitoring_loop(self):
        """运行监控循环"""
        print("\n🔍 启动实时监控...")
        
        while self.is_running:
            try:
                time.sleep(5)  # 每5秒更新一次状态
                if self.is_running:
                    self.show_real_time_status()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  监控循环错误: {e}")
                
    def show_system_info(self):
        """显示系统信息"""
        print("\n📋 系统信息:")
        print("   🏛️  架构: 多语言微服务")
        print("   🎯 目标吞吐量: 10 Gbps")
        print("   🔬 检测算法: 四阶段级联")
        print("   🧠 AI模型: CNN+LSTM")
        print("   🔒 安全性: 内存安全设计")
        print("   📊 监控: 实时性能指标")
        print("   🔄 扩展性: 水平扩展支持")

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n🔔 收到信号 {signum}, 准备优雅退出...")
    sys.exit(0)

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建系统管理器
    server = VPNDetectionServerManager()
    
    try:
        # 启动系统
        if server.start_system():
            
            # 显示系统信息
            server.show_system_info()
            
            # 启动监控线程
            monitor_thread = threading.Thread(target=server.run_monitoring_loop)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # 运行演示流量
            time.sleep(2)
            server.run_demo_traffic()
            
            # 等待用户中断
            print("\n🎮 系统已进入运行状态")
            print("   - 实时检测: ✅ 活跃")
            print("   - 监控状态: ✅ 活跃") 
            print("   - API服务: 🔶 模拟 (需要Go环境)")
            print("   - 消息总线: 🔶 模拟 (需要Rust环境)")
            print("\n💬 提示: 当前运行Python核心引擎，其他组件为模拟状态")
            print("   完整部署请安装 Go、Rust、C++ 开发环境")
            
            # 保持运行状态
            while server.is_running:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n👋 收到退出信号...")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止系统
        server.stop_system()
        print("👋 感谢使用VPN检测系统!")

if __name__ == "__main__":
    main()