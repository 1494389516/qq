#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东商城爬虫打包脚本
用于将爬虫项目打包成独立的exe可执行文件
"""

import os
import sys
import shutil
import subprocess
from datetime import datetime

# 打印彩色文本的函数
def print_color(text, color):
    colors = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def check_requirements():
    """检查是否安装了必要的包"""
    try:
        import PyInstaller
        print_color("✓ PyInstaller已安装", "green")
    except ImportError:
        print_color("✗ 未安装PyInstaller，正在安装...", "yellow")
        subprocess.call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print_color("✓ PyInstaller安装完成", "green")

def clean_build_dirs():
    """清理旧的构建目录"""
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print_color(f"正在清理 {dir_name} 目录...", "blue")
            shutil.rmtree(dir_name)
    
    # 清理spec文件
    spec_file = 'jd_spider.spec'
    if os.path.exists(spec_file):
        os.remove(spec_file)

def build_exe():
    """构建exe文件"""
    print_color("\n开始构建可执行文件...", "blue")
    
    # 主程序入口
    main_script = 'run_jd_spider.py'
    
    # 需要包含的数据文件
    datas = [
        ('config.py', '.'),
        ('README.md', '.'),
        ('requirements.txt', '.'),
        ('example_urls.txt', '.'),
        ('UPDATE_LOG.md', '.'),
        ('LOGIN_GUIDE.md', '.'),
        ('FEATURE_SUMMARY.md', '.')
    ]
    
    # 需要包含的目录
    dirs_to_include = [
        ('cookies', 'cookies'),
        ('logs', 'logs'),
        ('results', 'results')
    ]
    
    # 确保目录存在
    for src, _ in dirs_to_include:
        if not os.path.exists(src):
            os.makedirs(src)
    
    # 构建数据文件参数
    datas_param = []
    for src, dst in datas:
        if os.path.exists(src):
            datas_param.append(f"{src};{dst}")
    
    for src, dst in dirs_to_include:
        if os.path.exists(src):
            datas_param.append(f"{src};{dst}")
    
    # 构建命令
    cmd = [
        'pyinstaller',
        '--name=jd_spider',
        '--onefile',
        '--windowed',
        '--icon=jd_icon.ico' if os.path.exists('jd_icon.ico') else '',
        f'--add-data={";" if os.name == "nt" else ":".join(datas_param)}',
        '--hidden-import=PIL._tkinter_finder',
        '--hidden-import=colorama',
        '--hidden-import=selenium',
        '--hidden-import=undetected_chromedriver',
        '--hidden-import=tqdm',
        '--hidden-import=fake_useragent',
        main_script
    ]
    
    # 过滤掉空选项
    cmd = [item for item in cmd if item]
    
    # 执行构建
    print_color(f"执行命令: {' '.join(cmd)}", "yellow")
    subprocess.call(cmd)

def copy_chromedriver():
    """复制ChromeDriver到dist目录"""
    if os.path.exists('chromedriver.exe') and os.path.exists('dist'):
        print_color("\n正在复制ChromeDriver...", "blue")
        shutil.copy('chromedriver.exe', 'dist/chromedriver.exe')
        print_color("✓ ChromeDriver复制完成", "green")

def create_readme_for_exe():
    """为exe创建使用说明文件"""
    readme_content = """# 京东商城爬虫 (EXE版本)

## 使用说明

1. 确保chromedriver.exe与jd_spider.exe在同一目录下
2. 双击jd_spider.exe运行程序
3. 按照提示输入参数或使用命令行参数

## 命令行参数

与Python版本相同，例如：

```
jd_spider.exe --url https://item.jd.com/100050401004.html --login
```

## 注意事项

1. 首次运行可能会被Windows Defender或杀毒软件拦截，这是因为exe是自打包的
2. 请确保您的Chrome浏览器已安装且版本与chromedriver匹配
3. 程序运行日志将保存在logs目录下
4. 爬取结果将保存在results目录下

## 常见问题

如遇到问题，请参考README.md和LOGIN_GUIDE.md文件
"""
    
    with open('dist/使用说明.txt', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print_color("✓ 使用说明文件创建完成", "green")

def create_batch_file():
    """创建批处理文件，方便用户运行"""
    batch_content = """@echo off
echo 京东商城爬虫启动器
echo =====================
echo 1. 爬取单个商品
echo 2. 批量爬取商品
echo 3. 查看使用说明
echo =====================
choice /c 123 /n /m "请选择操作 [1-3]: "

if errorlevel 3 goto HELP
if errorlevel 2 goto BATCH
if errorlevel 1 goto SINGLE

:SINGLE
cls
echo 爬取单个商品
echo =====================
set /p url="请输入商品URL: "
jd_spider.exe --url %url% --login
pause
goto END

:BATCH
cls
echo 批量爬取商品
echo =====================
echo 请确保example_urls.txt文件中已添加商品URL(每行一个)
choice /c yn /n /m "是否继续? [y/n]: "
if errorlevel 2 goto END
jd_spider.exe --batch --url-file example_urls.txt --login
pause
goto END

:HELP
cls
type 使用说明.txt
pause
goto END

:END
"""
    
    with open('dist/启动爬虫.bat', 'w', encoding='utf-8') as f:
        f.write(batch_content)
    print_color("✓ 批处理启动文件创建完成", "green")

def main():
    """主函数"""
    start_time = datetime.now()
    
    print_color("\n========================================", "cyan")
    print_color("      京东商城爬虫 - EXE打包工具      ", "cyan")
    print_color("========================================\n", "cyan")
    
    # 检查依赖
    check_requirements()
    
    # 清理旧的构建文件
    clean_build_dirs()
    
    # 构建exe
    build_exe()
    
    # 复制ChromeDriver
    copy_chromedriver()
    
    # 创建使用说明
    create_readme_for_exe()
    
    # 创建批处理文件
    create_batch_file()
    
    # 计算耗时
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_color("\n========================================", "cyan")
    print_color(f"构建完成! 耗时: {duration:.2f}秒", "green")
    print_color(f"可执行文件位置: {os.path.abspath('dist/jd_spider.exe')}", "green")
    print_color("========================================\n", "cyan")

if __name__ == "__main__":
    main() 