#!/bin/bash

# 统一多语言混合风控系统启动脚本
# 整合DDoS防护、VPN检测、恶意行为识别等全方位风控能力

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}"
echo "🛡️  统一多语言混合风控系统"
echo "=================================="
echo "🔍 DDoS防护 + VPN检测 + 恶意行为识别"
echo "⚡ C++ + Rust + Go + Python + JavaScript"
echo "🎯 基于风控算法专家的理论框架"
echo -e "${NC}"

# 检查Docker环境
check_docker() {
    echo -e "${BLUE}🐳 检查Docker环境...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker未安装${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose未安装${NC}"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo -e "${RED}❌ Docker守护进程未运行${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker环境检查通过${NC}"
}

# 构建镜像
build_images() {
    echo -e "${BLUE}🔨 构建镜像...${NC}"
    
    # 运行构建脚本
    if [ -f "./build.sh" ]; then
        chmod +x ./build.sh
        ./build.sh
    else
        echo -e "${YELLOW}⚠️  build.sh不存在，跳过构建步骤${NC}"
    fi
    
    echo -e "${GREEN}✅ 镜像构建完成${NC}"
}

# 启动服务
start_services() {
    echo -e "${BLUE}🚀 启动统一风控系统...${NC}"
    
    # 停止现有服务（如果存在）
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # 启动所有服务
    docker-compose up -d
    
    echo -e "${GREEN}✅ 服务启动完成${NC}"
}

# 等待服务就绪
wait_for_services() {
    echo -e "${BLUE}⏳ 等待服务就绪...${NC}"
    
    services=(
        "Python控制层:http://localhost:8000/health"
        "C++检测层:http://localhost:8001/health"  
        "Go微服务层:http://localhost:8002/health"
        "Rust存储层:http://localhost:8003/health"
        "前端监控层:http://localhost:8080/health"
    )
    
    for service in "${services[@]}"; do
        name="${service%%:*}"
        url="${service##*:}"
        
        echo -e "${YELLOW}  等待 $name...${NC}"
        
        for i in {1..30}; do
            if curl -s "$url" &> /dev/null; then
                echo -e "${GREEN}    ✅ $name 就绪${NC}"
                break
            fi
            
            if [ $i -eq 30 ]; then
                echo -e "${RED}    ❌ $name 启动超时${NC}"
            else
                sleep 2
            fi
        done
    done
}

# 显示系统状态
show_status() {
    echo -e "${BLUE}📊 系统状态概览${NC}"
    echo "=================================="
    
    # 获取容器状态
    echo -e "${YELLOW}🐳 容器状态:${NC}"
    docker-compose ps
    
    echo ""
    echo -e "${YELLOW}🌐 服务端点:${NC}"
    echo "  📱 前端监控大屏: http://localhost:8080"
    echo "  🐍 Python控制API: http://localhost:8000" 
    echo "  ⚡ C++检测服务: http://localhost:8001"
    echo "  🐹 Go微服务网关: http://localhost:8002"
    echo "  🦀 Rust存储服务: http://localhost:8003"
    
    echo ""
    echo -e "${YELLOW}📈 监控面板:${NC}"
    echo "  📊 Grafana仪表板: http://localhost:3000 (admin/admin123)"
    echo "  📉 Prometheus监控: http://localhost:9090"
    echo "  🔍 Jaeger链路追踪: http://localhost:16686"
    
    echo ""
    echo -e "${YELLOW}💾 数据存储:${NC}"
    echo "  📦 Redis缓存: localhost:6379"
    echo "  🗄️  数据卷: $(docker volume ls --filter name=统一 --format "table {{.Name}}")"
}

# 运行健康检查
run_health_check() {
    echo -e "${BLUE}🏥 运行系统健康检查...${NC}"
    
    if [ -f "./tests/integration/test_system.py" ]; then
        python3 ./tests/integration/test_system.py
        echo -e "${GREEN}✅ 健康检查通过${NC}"
    else
        echo -e "${YELLOW}⚠️  健康检查脚本不存在${NC}"
    fi
}

# 显示日志
show_logs() {
    echo -e "${BLUE}📝 查看系统日志...${NC}"
    echo "按 Ctrl+C 退出日志查看"
    sleep 2
    docker-compose logs -f
}

# 停止系统
stop_system() {
    echo -e "${BLUE}🛑 停止统一风控系统...${NC}"
    docker-compose down
    echo -e "${GREEN}✅ 系统已停止${NC}"
}

# 清理系统
cleanup_system() {
    echo -e "${BLUE}🧹 清理系统资源...${NC}"
    
    # 停止并删除容器
    docker-compose down --volumes --remove-orphans
    
    # 删除未使用的镜像
    docker image prune -f
    
    # 删除未使用的卷
    docker volume prune -f
    
    echo -e "${GREEN}✅ 系统清理完成${NC}"
}

# 显示帮助信息
show_help() {
    echo -e "${YELLOW}使用说明:${NC}"
    echo "  $0 [选项]"
    echo ""
    echo -e "${YELLOW}选项:${NC}"
    echo "  start     启动系统 (默认)"
    echo "  stop      停止系统"
    echo "  restart   重启系统" 
    echo "  status    显示状态"
    echo "  logs      查看日志"
    echo "  health    健康检查"
    echo "  cleanup   清理资源"
    echo "  help      显示帮助"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  $0 start     # 启动完整的风控系统"
    echo "  $0 logs      # 查看所有服务日志"
    echo "  $0 cleanup   # 完全清理系统"
}

# 主函数
main() {
    case "${1:-start}" in
        "start")
            check_docker
            build_images
            start_services
            wait_for_services
            show_status
            echo ""
            echo -e "${GREEN}🎉 统一多语言混合风控系统启动成功！${NC}"
            echo -e "${BLUE}💡 访问 http://localhost:8080 查看监控大屏${NC}"
            ;;
        "stop")
            stop_system
            ;;
        "restart")
            stop_system
            sleep 2
            main start
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "health")
            run_health_check
            ;;
        "cleanup")
            cleanup_system
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            echo -e "${RED}❌ 未知选项: $1${NC}"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"