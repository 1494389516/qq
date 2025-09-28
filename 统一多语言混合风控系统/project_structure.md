# 统一多语言混合风控系统 - 项目结构

## 📁 完整项目目录结构

```
统一多语言混合风控系统/
├── README.md                           # 系统总体说明
├── project_structure.md               # 项目结构文档
├── unified_risk_control.py            # Python统一控制层
├── docker-compose.yml                 # 容器编排配置
├── kubernetes/                        # K8s部署配置
│   ├── python-control-deployment.yaml
│   ├── cpp-detector-deployment.yaml
│   ├── rust-storage-deployment.yaml
│   ├── go-gateway-deployment.yaml
│   └── js-frontend-deployment.yaml
│
├── python_control/                    # Python控制层
│   ├── __init__.py
│   ├── unified_risk_control.py        # 主控制器
│   ├── models/                        # ML模型模块
│   │   ├── gradient_boosting.py       # 梯度提升算法
│   │   ├── intelligent_selector.py    # 智能模型选择
│   │   └── ab_testing.py              # A/B测试框架
│   ├── risk_fusion/                   # 风险融合模块
│   │   ├── scene_detector.py          # 场景感知器
│   │   ├── pv_strategy.py             # PV驱动策略
│   │   └── multi_layer_fusion.py     # 多层次融合
│   ├── config/                        # 配置管理
│   │   ├── business_contexts.py       # 业务场景配置
│   │   └── model_configs.py           # 模型配置
│   └── requirements.txt               # Python依赖
│
├── cpp_detector/                      # C++高性能检测层
│   ├── include/
│   │   ├── risk_detector.hpp          # 主检测器头文件
│   │   ├── simd_feature_extractor.hpp # SIMD特征提取
│   │   ├── behavior_classifier.hpp    # 行为分类器
│   │   ├── vpn_detector.hpp           # VPN检测器
│   │   └── ddos_engine.hpp            # DDoS防护引擎
│   ├── src/
│   │   ├── risk_detector.cpp          # 主检测器实现
│   │   ├── simd_feature_extractor.cpp # SIMD优化实现
│   │   ├── behavior_classifier.cpp    # 行为分类实现
│   │   ├── vpn_detector.cpp           # VPN检测实现
│   │   ├── ddos_engine.cpp            # DDoS引擎实现
│   │   ├── api_server.cpp             # REST API服务器
│   │   └── main.cpp                   # 主程序入口
│   ├── CMakeLists.txt                 # CMake构建配置
│   ├── Dockerfile                     # 容器构建配置
│   └── config/
│       ├── model_weights.bin          # 预训练模型权重
│       └── detection_config.json     # 检测配置
│
├── rust_storage/                      # Rust安全存储层
│   ├── Cargo.toml                     # Rust项目配置
│   ├── src/
│   │   ├── lib.rs                     # 主库文件
│   │   ├── storage_engine.rs          # 存储引擎
│   │   ├── timeseries.rs              # 时序数据处理
│   │   ├── cache.rs                   # 内存缓存
│   │   ├── zero_copy.rs               # 零拷贝优化
│   │   ├── api_server.rs              # HTTP API服务
│   │   └── main.rs                    # 主程序入口
│   ├── tests/                         # 单元测试
│   │   ├── storage_tests.rs
│   │   └── cache_tests.rs
│   ├── Dockerfile                     # 容器配置
│   └── config/
│       └── storage_config.toml        # 存储配置
│
├── go_gateway/                        # Go微服务网关层
│   ├── main.go                        # 主程序
│   ├── cmd/
│   │   └── api_gateway/
│   │       └── main.go                # API网关入口
│   ├── internal/
│   │   ├── gateway/                   # 网关核心
│   │   │   ├── router.go              # 路由管理
│   │   │   ├── middleware.go          # 中间件
│   │   │   ├── load_balancer.go       # 负载均衡
│   │   │   └── circuit_breaker.go     # 熔断器
│   │   ├── service/                   # 服务发现
│   │   │   ├── discovery.go           # 服务发现
│   │   │   └── registry.go            # 服务注册
│   │   ├── monitoring/                # 监控模块
│   │   │   ├── metrics.go             # 指标收集
│   │   │   └── health_check.go        # 健康检查
│   │   └── config/
│   │       └── config.go              # 配置管理
│   ├── pkg/                           # 公共包
│   │   ├── response/
│   │   │   └── response.go            # 响应格式
│   │   └── utils/
│   │       └── utils.go               # 工具函数
│   ├── go.mod                         # Go模块定义
│   ├── go.sum                         # 依赖校验
│   ├── Dockerfile                     # 容器配置
│   └── config/
│       └── gateway_config.yaml        # 网关配置
│
├── js_frontend/                       # JavaScript前端监控层
│   ├── package.json                   # NPM配置
│   ├── src/
│   │   ├── index.js                   # 入口文件
│   │   ├── components/                # 组件模块
│   │   │   ├── RealTimeMonitor.js     # 实时监控组件
│   │   │   ├── ThreatVisualization.js # 威胁可视化
│   │   │   ├── ConfigPanel.js         # 配置面板
│   │   │   └── AlertManager.js        # 告警管理
│   │   ├── services/                  # 服务层
│   │   │   ├── api.js                 # API调用
│   │   │   ├── websocket.js           # WebSocket连接
│   │   │   └── data_processor.js      # 数据处理
│   │   ├── utils/                     # 工具函数
│   │   │   ├── chart_utils.js         # 图表工具
│   │   │   └── date_utils.js          # 日期工具
│   │   └── styles/                    # 样式文件
│   │       ├── main.css
│   │       └── components.css
│   ├── public/                        # 静态资源
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── Dockerfile                     # 容器配置
│   └── config/
│       └── frontend_config.json      # 前端配置
│
├── scripts/                           # 部署脚本
│   ├── build_all.sh                  # 全量构建脚本
│   ├── deploy_local.sh               # 本地部署脚本
│   ├── deploy_k8s.sh                 # K8s部署脚本
│   ├── health_check.sh               # 健康检查脚本
│   └── performance_test.sh           # 性能测试脚本
│
├── docs/                              # 文档
│   ├── architecture.md               # 架构设计文档
│   ├── api_reference.md              # API参考文档
│   ├── deployment_guide.md           # 部署指南
│   ├── performance_tuning.md         # 性能调优指南
│   └── troubleshooting.md            # 故障排除指南
│
├── tests/                             # 集成测试
│   ├── integration/                   # 集成测试
│   │   ├── test_full_pipeline.py     # 完整流程测试
│   │   ├── test_multi_language.py    # 多语言协同测试
│   │   └── test_performance.py       # 性能测试
│   ├── load_testing/                 # 压力测试
│   │   ├── locustfile.py             # Locust压测脚本
│   │   └── k6_script.js              # K6压测脚本
│   └── data/                         # 测试数据
│       ├── sample_packets.json       # 样本网络包
│       └── sample_kpi.json           # 样本KPI数据
│
├── monitoring/                        # 监控配置
│   ├── prometheus/                    # Prometheus配置
│   │   ├── prometheus.yml
│   │   └── alerts.yml
│   ├── grafana/                       # Grafana仪表板
│   │   ├── dashboards/
│   │   │   ├── system_overview.json
│   │   │   ├── threat_analysis.json
│   │   │   └── performance_metrics.json
│   │   └── provisioning/
│   └── jaeger/                        # 链路追踪配置
│       └── jaeger_config.yaml
│
└── config/                            # 全局配置
    ├── global_config.yaml             # 全局配置文件
    ├── security_policies.yaml        # 安全策略配置
    ├── model_registry.yaml           # 模型注册表
    └── deployment_configs/            # 部署环境配置
        ├── development.yaml
        ├── staging.yaml
        └── production.yaml
```

## 🔧 技术栈说明

### Python层 (控制大脑)
- **框架**: FastAPI, asyncio
- **ML库**: XGBoost, LightGBM, scikit-learn
- **数据处理**: NumPy, Pandas
- **API**: RESTful + WebSocket

### C++层 (高性能引擎)
- **编译器**: GCC 11+ / Clang 13+
- **标准**: C++17/20
- **优化**: SIMD (AVX2), OpenMP
- **HTTP**: Beast (Boost), nlohmann/json
- **构建**: CMake 3.20+

### Rust层 (安全存储)
- **版本**: Rust 1.65+
- **异步**: Tokio, async-std
- **HTTP**: Warp, Serde
- **并发**: Rayon, Crossbeam
- **存储**: RocksDB, MemoryPool

### Go层 (微服务网关)
- **版本**: Go 1.19+
- **框架**: Gin, Echo
- **网关**: Traefik, Kong
- **服务发现**: Consul, etcd
- **监控**: Prometheus, Jaeger

### JavaScript层 (前端监控)
- **运行时**: Node.js 18+
- **框架**: React 18, Vue 3
- **可视化**: D3.js, Chart.js, WebGL
- **实时**: WebSocket, Server-Sent Events
- **构建**: Webpack, Vite

## 🚀 部署架构

### 本地开发环境
```bash
# 启动完整系统
./scripts/build_all.sh
docker-compose up -d
```

### 生产环境 (Kubernetes)
```bash
# 部署到K8s集群
./scripts/deploy_k8s.sh production
```

### 性能指标
| 组件 | 延迟目标 | 吞吐量目标 | 可用性目标 |
|------|----------|------------|------------|
| Python控制层 | < 10ms | > 10K RPS | 99.9% |
| C++检测层 | < 1ms | > 100K RPS | 99.99% |
| Rust存储层 | < 5ms | > 50K RPS | 99.95% |
| Go网关层 | < 2ms | > 80K RPS | 99.9% |
| JS前端层 | < 100ms | > 1K CCU | 99.5% |

## 📊 监控体系

### 实时监控指标
- 系统性能：CPU、内存、网络、磁盘
- 业务指标：检测率、误报率、响应时间
- 威胁分析：攻击类型分布、风险等级统计
- 组件健康：各语言组件状态、服务依赖关系

### 告警策略
- **P0级**: 系统整体不可用 (< 1分钟响应)
- **P1级**: 核心功能异常 (< 5分钟响应)
- **P2级**: 性能指标异常 (< 30分钟响应)
- **P3级**: 一般性异常 (< 2小时响应)

这个统一的多语言混合风控系统充分发挥了各编程语言的技术优势，实现了高性能、高安全、高可靠的企业级风控平台。