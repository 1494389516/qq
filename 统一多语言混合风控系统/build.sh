#!/bin/bash

# 统一多语言混合风控系统构建脚本
# 基于风控算法专家的理论框架，整合所有风控模块

set -e

echo "🚀 开始构建统一多语言混合风控系统..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查依赖
check_dependencies() {
    echo -e "${BLUE}📋 检查系统依赖...${NC}"
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker未安装${NC}"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose未安装${NC}"
        exit 1
    fi
    
    # 检查C++编译器
    if ! command -v g++ &> /dev/null; then
        echo -e "${YELLOW}⚠️  警告: g++未安装，将在容器内编译${NC}"
    fi
    
    # 检查Rust
    if ! command -v cargo &> /dev/null; then
        echo -e "${YELLOW}⚠️  警告: Rust未安装，将在容器内编译${NC}"
    fi
    
    # 检查Go
    if ! command -v go &> /dev/null; then
        echo -e "${YELLOW}⚠️  警告: Go未安装，将在容器内编译${NC}"
    fi
    
    # 检查Node.js
    if ! command -v node &> /dev/null; then
        echo -e "${YELLOW}⚠️  警告: Node.js未安装，将在容器内编译${NC}"
    fi
    
    echo -e "${GREEN}✅ 依赖检查完成${NC}"
}

# 创建必要的目录
create_directories() {
    echo -e "${BLUE}📁 创建项目目录结构...${NC}"
    
    mkdir -p logs models config data
    mkdir -p monitoring/prometheus monitoring/grafana/dashboards
    mkdir -p nginx/ssl
    mkdir -p tests/integration tests/load_testing
    
    echo -e "${GREEN}✅ 目录结构创建完成${NC}"
}

# 构建C++检测层
build_cpp_detector() {
    echo -e "${BLUE}🔧 构建C++高性能检测层...${NC}"
    
    cd cpp_detector
    
    # 创建Dockerfile
    cat > Dockerfile << 'EOF'
FROM ubuntu:22.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libcurl4-openssl-dev \
    libjsoncpp-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# 编译
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

EXPOSE 8001

CMD ["./build/risk_detector"]
EOF
    
    # 创建CMakeLists.txt
    cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(UnifiedRiskDetector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

# 查找依赖
find_package(PkgConfig REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

# 包含目录
include_directories(${CMAKE_SOURCE_DIR})

# 源文件
set(SOURCES
    main.cpp
    risk_detector.cpp
    simd_feature_extractor.cpp
    behavior_classifier.cpp
    vpn_detector.cpp
    ddos_engine.cpp
    api_server.cpp
)

# 创建可执行文件
add_executable(risk_detector ${SOURCES})

# 链接库
target_link_libraries(risk_detector
    ${Boost_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
    ${CURL_LIBRARIES}
    jsoncpp
    pthread
)
EOF

    # 创建简化的main.cpp
    cat > main.cpp << 'EOF'
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "🔍 统一风控检测器启动中..." << std::endl;
    
    // 模拟检测服务
    while (true) {
        std::cout << "⚡ 高性能检测引擎运行中..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(30));
    }
    
    return 0;
}
EOF
    
    cd ..
    echo -e "${GREEN}✅ C++检测层构建配置完成${NC}"
}

# 构建Rust存储层
build_rust_storage() {
    echo -e "${BLUE}🦀 构建Rust安全存储层...${NC}"
    
    cd rust_storage
    
    # 创建Cargo.toml
    cat > Cargo.toml << 'EOF'
[package]
name = "unified-risk-storage"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
warp = "0.3"
dashmap = "5.0"
rayon = "1.7"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"

[lib]
name = "unified_risk_storage"
crate-type = ["cdylib", "rlib"]
EOF
    
    # 创建Dockerfile
    cat > Dockerfile << 'EOF'
FROM rust:1.70-slim

WORKDIR /app
COPY . .

RUN cargo build --release

EXPOSE 8003

CMD ["./target/release/unified-risk-storage"]
EOF

    # 创建main.rs
    cat > src/main.rs << 'EOF'
use unified_risk_storage::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("🦀 Rust安全存储引擎启动中...");
    
    let server = RustStorageServer::new(8003);
    server.start().await?;
    
    Ok(())
}
EOF
    
    cd ..
    echo -e "${GREEN}✅ Rust存储层构建配置完成${NC}"
}

# 构建Go微服务层
build_go_gateway() {
    echo -e "${BLUE}🐹 构建Go微服务层...${NC}"
    
    cd go_gateway
    
    # 创建go.mod
    cat > go.mod << 'EOF'
module unified-risk-gateway

go 1.19

require (
    github.com/gin-gonic/gin v1.9.1
    github.com/gorilla/websocket v1.5.0
)
EOF
    
    # 创建Dockerfile
    cat > Dockerfile << 'EOF'
FROM golang:1.19-alpine AS builder

WORKDIR /app
COPY . .
RUN go mod tidy && go build -o gateway main.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/gateway .

EXPOSE 8002

CMD ["./gateway"]
EOF
    
    cd ..
    echo -e "${GREEN}✅ Go微服务层构建配置完成${NC}"
}

# 构建Python控制层
build_python_control() {
    echo -e "${BLUE}🐍 构建Python控制层...${NC}"
    
    cd python_control
    
    # 创建requirements.txt
    cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
scipy==1.11.1
aiohttp==3.8.5
websockets==11.0.3
pydantic==2.4.2
asyncio==3.4.3
EOF
    
    # 创建Dockerfile
    cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "unified_risk_control.py"]
EOF
    
    cd ..
    echo -e "${GREEN}✅ Python控制层构建配置完成${NC}"
}

# 构建JavaScript前端层
build_js_frontend() {
    echo -e "${BLUE}📱 构建JavaScript前端层...${NC}"
    
    mkdir -p js_frontend
    cd js_frontend
    
    # 创建package.json
    cat > package.json << 'EOF'
{
  "name": "unified-risk-frontend",
  "version": "1.0.0",
  "description": "统一风控系统监控前端",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "socket.io": "^4.7.2",
    "ws": "^8.14.2"
  }
}
EOF
    
    # 创建简单的服务器
    cat > server.js << 'EOF'
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = socketIo(server);

app.use(express.static('public'));

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date() });
});

app.get('/', (req, res) => {
    res.send(`
        <h1>🛡️ 统一风控系统监控大屏</h1>
        <p>实时监控多语言混合风控系统状态</p>
        <div id="status">系统运行正常</div>
    `);
});

io.on('connection', (socket) => {
    console.log('客户端连接');
    
    // 模拟实时数据推送
    setInterval(() => {
        socket.emit('threat-update', {
            timestamp: new Date(),
            threats_detected: Math.floor(Math.random() * 10),
            system_load: Math.random() * 100
        });
    }, 5000);
});

const PORT = process.env.PORT || 8080;
server.listen(PORT, () => {
    console.log(`🚀 前端监控服务启动在端口 ${PORT}`);
});
EOF
    
    # 创建Dockerfile
    cat > Dockerfile << 'EOF'
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 8080

CMD ["npm", "start"]
EOF
    
    cd ..
    echo -e "${GREEN}✅ JavaScript前端层构建配置完成${NC}"
}

# 创建监控配置
setup_monitoring() {
    echo -e "${BLUE}📊 配置监控系统...${NC}"
    
    # Prometheus配置
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'unified-risk-control'
    static_configs:
      - targets: ['python-control:8000', 'cpp-detector:8001', 'go-gateway:8002', 'rust-storage:8003']
EOF
    
    # Grafana dashboard配置
    cat > monitoring/grafana/dashboards/system-overview.json << 'EOF'
{
  "dashboard": {
    "title": "统一风控系统监控",
    "panels": [
      {
        "title": "威胁检测数量",
        "type": "graph"
      },
      {
        "title": "系统性能指标", 
        "type": "singlestat"
      }
    ]
  }
}
EOF
    
    echo -e "${GREEN}✅ 监控配置完成${NC}"
}

# 创建测试配置
setup_testing() {
    echo -e "${BLUE}🧪 配置测试环境...${NC}"
    
    # 集成测试
    cat > tests/integration/test_system.py << 'EOF'
#!/usr/bin/env python3

import asyncio
import aiohttp
import pytest

class TestUnifiedRiskSystem:
    async def test_health_checks(self):
        """测试所有服务的健康检查"""
        services = [
            'http://localhost:8000/health',  # Python
            'http://localhost:8001/health',  # C++
            'http://localhost:8002/health',  # Go  
            'http://localhost:8003/health',  # Rust
            'http://localhost:8080/health',  # JS
        ]
        
        async with aiohttp.ClientSession() as session:
            for service in services:
                async with session.get(service) as response:
                    assert response.status == 200

if __name__ == "__main__":
    asyncio.run(TestUnifiedRiskSystem().test_health_checks())
EOF
    
    chmod +x tests/integration/test_system.py
    echo -e "${GREEN}✅ 测试配置完成${NC}"
}

# 主构建流程
main() {
    echo -e "${GREEN}🎯 统一多语言混合风控系统构建${NC}"
    echo "=========================================="
    
    check_dependencies
    create_directories
    build_cpp_detector
    build_rust_storage
    build_go_gateway
    build_python_control
    build_js_frontend
    setup_monitoring
    setup_testing
    
    echo ""
    echo -e "${GREEN}✅ 构建完成！${NC}"
    echo ""
    echo -e "${BLUE}🚀 启动系统:${NC}"
    echo "  docker-compose up -d"
    echo ""
    echo -e "${BLUE}📊 访问监控:${NC}"
    echo "  Grafana: http://localhost:3000"
    echo "  Prometheus: http://localhost:9090"
    echo "  前端监控: http://localhost:8080"
    echo ""
    echo -e "${BLUE}🧪 运行测试:${NC}"
    echo "  ./tests/integration/test_system.py"
    echo ""
    echo -e "${YELLOW}⚠️  注意: 首次启动可能需要几分钟来下载依赖${NC}"
}

# 执行主函数
main "$@"