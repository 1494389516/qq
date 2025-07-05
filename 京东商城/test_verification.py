#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东验证码处理测试脚本
用于测试验证码处理功能
"""

import os
import sys
import time
import logging
import argparse
import colorama
from colorama import Fore, Style
from config import SPIDER_CONFIG
from jd_main import init_browser, Colors
from 京东商城 import JDVerificationHandler

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
    parser = argparse.ArgumentParser(description='京东验证码处理测试')
    
    parser.add_argument('-u', '--url', type=str,
                        default="https://passport.jd.com/new/login.aspx",
                        help='要测试的京东URL，默认为登录页面')
    
    parser.add_argument('--debug', action='store_true',
                        default=False,
                        help='是否启用调试模式')
    
    return parser.parse_args()

def show_banner():
    """显示测试脚本Banner"""
    banner = f"""
{Fore.CYAN}{Style.BRIGHT}=================================================
           京东验证码处理测试 v1.0.0
=================================================
{Fore.GREEN}✓ 测试滑块验证码处理
{Fore.GREEN}✓ 测试图形验证码处理
{Fore.GREEN}✓ 测试快速验证按钮处理
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
    
    # 显示测试信息
    print(f"{Colors.BLUE}开始测试京东验证码处理功能...{Colors.RESET}")
    print(f"目标URL: {Colors.CYAN}{args.url}{Colors.RESET}")
    
    # 初始化浏览器
    try:
        driver = init_browser()
        
        # 访问目标URL
        print(f"\n{Colors.BLUE}正在访问目标URL...{Colors.RESET}")
        driver.get(args.url)
        time.sleep(3)
        
        # 创建验证处理器
        verification_handler = JDVerificationHandler(driver)
        
        # 检查是否需要处理验证码
        print(f"\n{Colors.BLUE}检查是否需要处理验证码...{Colors.RESET}")
        verification_handler.handle_verification()
        
        # 等待用户确认
        print(f"\n{Colors.GREEN}验证码处理测试完成{Colors.RESET}")
        print("请观察浏览器窗口，确认验证码是否已成功处理")
        print("按回车键继续...")
        input()
        
        # 保存页面截图
        screenshot_path = "verification_result.png"
        driver.save_screenshot(screenshot_path)
        print(f"{Colors.GREEN}已保存验证结果截图: {screenshot_path}{Colors.RESET}")
        
        # 关闭浏览器
        driver.quit()
        print(f"{Colors.GREEN}测试完成!{Colors.RESET}")
        
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
    finally:
        try:
            if 'driver' in locals() and driver:
                driver.quit()
        except:
            pass

if __name__ == "__main__":
    sys.exit(main()) 