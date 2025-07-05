#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================"
echo "      京东商城爬虫 - EXE打包工具"
echo -e "========================================${NC}"
echo ""

# 检查Python环境
echo -e "${BLUE}检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到Python，请先安装Python 3.8或更高版本${NC}"
    exit 1
fi

# 检查Python版本
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if (( $(echo "$PY_VERSION < 3.8" | bc -l) )); then
    echo -e "${RED}错误: Python版本过低，需要Python 3.8或更高版本${NC}"
    echo -e "${RED}当前版本: $PY_VERSION${NC}"
    exit 1
fi

# 检查是否已安装依赖
echo -e "${BLUE}检查依赖...${NC}"
if ! python3 -c "import PyInstaller" &> /dev/null; then
    echo -e "${YELLOW}安装PyInstaller...${NC}"
    pip3 install pyinstaller
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 安装PyInstaller失败${NC}"
        exit 1
    fi
fi

# 检查requirements.txt
if [ -f "requirements.txt" ]; then
    echo -e "${BLUE}安装项目依赖...${NC}"
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}警告: 安装部分依赖失败，但将继续构建${NC}"
    fi
fi

# 生成图标
echo -e "${BLUE}生成图标...${NC}"
python3 jd_icon.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}警告: 生成图标失败，将使用默认图标${NC}"
fi

# 清理旧的构建文件
echo -e "${BLUE}清理旧的构建文件...${NC}"
rm -rf build dist
rm -f jd_spider.spec

# 构建EXE
echo ""
echo -e "${BLUE}开始构建可执行文件...${NC}"
python3 build_exe.py
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 构建失败${NC}"
    exit 1
fi

# 检查构建结果
if [ ! -f "dist/jd_spider.exe" ] && [ ! -f "dist/jd_spider" ]; then
    echo -e "${RED}错误: 构建过程完成，但未找到生成的可执行文件${NC}"
    exit 1
fi

# 创建发布包
echo ""
echo -e "${BLUE}创建发布包...${NC}"
cd dist

# 获取版本号
VERSION="1.4.0"
if [ -f "../UPDATE_LOG.md" ]; then
    VERSION=$(grep -m 1 "v[0-9]\+\.[0-9]\+\.[0-9]\+" "../UPDATE_LOG.md" | grep -o "v[0-9]\+\.[0-9]\+\.[0-9]\+" | sed 's/v//')
fi

# 创建ZIP文件
ZIPFILE="jd_spider_v${VERSION}.zip"
echo -e "${BLUE}打包文件到 ${ZIPFILE} ...${NC}"

if command -v zip &> /dev/null; then
    zip -r "${ZIPFILE}" ./*
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}警告: 创建ZIP文件失败${NC}"
    else
        echo -e "${GREEN}成功创建发布包: ${ZIPFILE}${NC}"
    fi
else
    echo -e "${YELLOW}警告: 未找到zip命令，无法创建发布包${NC}"
    echo -e "${YELLOW}请手动压缩dist目录下的文件${NC}"
fi

cd ..

echo ""
echo -e "${CYAN}========================================"
echo -e "${GREEN}构建完成!${NC}"
echo ""
echo -e "${GREEN}可执行文件位置: $(pwd)/dist/jd_spider.exe${NC}"
if [ -f "dist/${ZIPFILE}" ]; then
    echo -e "${GREEN}发布包位置: $(pwd)/dist/${ZIPFILE}${NC}"
fi
echo -e "${CYAN}========================================${NC}"
echo "" 