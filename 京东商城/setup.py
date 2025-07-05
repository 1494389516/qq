#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东商城爬虫安装脚本
用于安装打包所需的依赖
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages

# 检查Python版本
if sys.version_info < (3, 8):
    print("错误: 京东爬虫需要Python 3.8或更高版本")
    sys.exit(1)

# 安装依赖
def install_requirements():
    """安装依赖包"""
    print("正在安装依赖...")
    
    try:
        # 从requirements.txt安装依赖
        if os.path.exists('requirements.txt'):
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        
        # 安装打包工具
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
        
        print("依赖安装完成!")
    except subprocess.CalledProcessError as e:
        print(f"安装依赖时出错: {e}")
        sys.exit(1)

# 读取README文件
def read_readme():
    """读取README文件内容"""
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "京东商城爬虫 - 一个功能强大的京东商品数据爬虫工具"

# 读取版本信息
def get_version():
    """从UPDATE_LOG.md中获取版本信息"""
    try:
        with open('UPDATE_LOG.md', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('## '):
                    # 格式: ## 2025-06-30 v1.4.0
                    parts = line.strip().split()
                    for part in parts:
                        if part.startswith('v'):
                            return part[1:]  # 去掉v前缀
        return '1.4.0'  # 默认版本
    except:
        return '1.4.0'  # 默认版本

# 主函数
def main():
    """主函数"""
    # 安装依赖
    install_requirements()
    
    # 配置setup
    setup(
        name="jd_spider",
        version=get_version(),
        description="京东商城爬虫 - 一个功能强大的京东商品数据爬虫工具",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        author="JD Spider Team",
        author_email="example@example.com",
        url="https://github.com/yourusername/jd-spider",
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            "selenium",
            "undetected-chromedriver",
            "tqdm",
            "colorama",
            "fake_useragent",
            "scrapy",
            "pillow",
            "pyautogui",
        ],
        entry_points={
            'console_scripts': [
                'jd-spider=run_jd_spider:main',
            ],
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Internet :: WWW/HTTP :: Browsers",
            "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        ],
        python_requires=">=3.8",
    )
    
    print("安装完成!")
    print("现在您可以运行 'python build_exe.py' 来构建可执行文件")

if __name__ == "__main__":
    main() 