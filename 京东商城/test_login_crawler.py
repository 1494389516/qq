#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东爬虫登录测试脚本
用于测试登录后爬取数据的功能
"""

import os
import sys
import time
import logging
import argparse
import colorama
from colorama import Fore, Style
from config import SPIDER_CONFIG
from jd_main import run_spider, Colors

# 初始化colorama
colorama.init(autoreset=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='京东爬虫登录测试')
    
    parser.add_argument('-u', '--url', type=str,
                        default="https://item.jd.com/100050401004.html",  # 默认为华为Mate60 Pro
                        help='要爬取的京东商品URL')
    
    parser.add_argument('-o', '--output', type=str,
                        default="results/test",
                        help='结果保存目录')
    
    parser.add_argument('--headless', action='store_true',
                        default=False,
                        help='是否使用无头浏览器模式')
    
    parser.add_argument('--debug', action='store_true',
                        default=False,
                        help='是否启用调试模式')
    
    return parser.parse_args()

def show_banner():
    """显示测试脚本Banner"""
    banner = f"""
{Fore.CYAN}{Style.BRIGHT}=================================================
           京东爬虫登录测试 v1.0.0
=================================================
{Fore.GREEN}✓ 自动登录并爬取商品数据
{Fore.GREEN}✓ 测试登录状态下的数据提取
{Fore.GREEN}✓ 验证cookie保存和加载功能
=================================================
{Style.RESET_ALL}
"""
    print(banner)

def main():
    """主函数"""
    # 显示Banner
    show_banner()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # 更新配置
    SPIDER_CONFIG['TARGET_URL'] = args.url
    SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'] = args.output
    SPIDER_CONFIG['BROWSER']['HEADLESS'] = args.headless
    SPIDER_CONFIG['LOGIN']['ENABLE'] = True  # 强制启用登录
    
    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 显示测试信息
    print(f"{Colors.BLUE}开始测试京东爬虫登录功能...{Colors.RESET}")
    print(f"目标URL: {Colors.CYAN}{args.url}{Colors.RESET}")
    print(f"输出目录: {Colors.CYAN}{args.output}{Colors.RESET}")
    print(f"无头模式: {Colors.CYAN}{'是' if args.headless else '否'}{Colors.RESET}")
    
    # 检查cookies目录
    cookies_dir = SPIDER_CONFIG['LOGIN']['COOKIES_DIR']
    if not os.path.exists(cookies_dir):
        os.makedirs(cookies_dir)
        print(f"{Colors.YELLOW}已创建cookies目录: {cookies_dir}{Colors.RESET}")
    else:
        cookie_files = [f for f in os.listdir(cookies_dir) if f.endswith(".pkl")]
        if cookie_files:
            print(f"{Colors.GREEN}发现{len(cookie_files)}个保存的登录状态{Colors.RESET}")
            for i, cookie_file in enumerate(cookie_files):
                username = cookie_file.replace(".pkl", "")
                last_modified = os.path.getmtime(os.path.join(cookies_dir, cookie_file))
                from datetime import datetime
                last_modified_str = datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {i+1}. {Colors.YELLOW}{username}{Colors.RESET} (保存于: {last_modified_str})")
        else:
            print(f"{Colors.YELLOW}未找到保存的登录状态，将进行扫码登录{Colors.RESET}")
    
    # 运行爬虫
    try:
        print(f"\n{Colors.GREEN}开始运行爬虫...{Colors.RESET}")
        run_spider(args.url)
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}测试完成!{Colors.RESET}")
        print(f"请检查输出目录 {Colors.CYAN}{args.output}{Colors.RESET} 中的数据文件")
        print(f"如果价格等敏感信息已正确提取，则表示登录功能正常")
        
        return 0
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}用户中断测试{Colors.RESET}")
        return 1
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"\n{Colors.RED}测试失败: {e}{Colors.RESET}")
        if args.debug:
            import traceback
            print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 