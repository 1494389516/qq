#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试京东URL重定向处理功能
"""

import os
import sys
import time
import argparse
import logging
import random
from datetime import datetime

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

from config import SPIDER_CONFIG
from 京东商城 import JDVerificationHandler
from jd_main import init_browser, Colors

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='京东URL重定向处理测试')
    
    parser.add_argument('-u', '--url', type=str, 
                        default=SPIDER_CONFIG['TARGET_URL'],
                        help='要测试的京东商品URL')
    
    parser.add_argument('--headless', action='store_true',
                        default=False,
                        help='是否使用无头浏览器模式')
    
    return parser.parse_args()

def test_url_redirect(url, headless=False):
    """测试URL重定向处理"""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}【京东URL重定向测试】启动{Colors.RESET}")
    print(f"目标URL: {Colors.BLUE}{url}{Colors.RESET}")
    print("="*60 + "\n")
    
    logger.info(f"开始测试: {url}")
    
    # 临时修改配置
    SPIDER_CONFIG['BROWSER']['HEADLESS'] = headless
    
    # 初始化浏览器
    driver = init_browser()
    
    try:
        # 访问目标URL
        logger.info(f"正在访问: {url}")
        print(f"{Colors.BLUE}正在加载页面...{Colors.RESET}")
        driver.get(url)
        
        # 随机等待，模拟人类行为
        time.sleep(random.uniform(3, 5))
        
        # 保存原始URL
        original_url = url
        logger.info(f"原始URL: {original_url}")
        
        # 获取当前URL
        current_url = driver.current_url
        logger.info(f"当前URL: {current_url}")
        
        # 保存页面截图
        screenshot_dir = os.path.join(SPIDER_CONFIG['OUTPUT']['DEBUG_DIR'], 'redirect_test')
        os.makedirs(screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(screenshot_dir, f"original_page_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        driver.save_screenshot(screenshot_path)
        logger.info(f"已保存页面截图: {screenshot_path}")
        
        # 检查URL是否发生变化
        if current_url != original_url:
            print(f"\n{Colors.YELLOW}检测到URL重定向:{Colors.RESET}")
            print(f"原始URL: {Colors.BLUE}{original_url}{Colors.RESET}")
            print(f"当前URL: {Colors.GREEN}{current_url}{Colors.RESET}")
            
            # 创建验证处理器
            verification_handler = JDVerificationHandler(driver)
            
            # 处理URL重定向
            print(f"\n{Colors.BLUE}正在处理URL重定向...{Colors.RESET}")
            updated_url = verification_handler.handle_url_redirect(original_url)
            
            # 检查处理结果
            if updated_url != original_url:
                print(f"\n{Colors.GREEN}URL已更新:{Colors.RESET}")
                print(f"处理后URL: {Colors.GREEN}{updated_url}{Colors.RESET}")
                
                # 访问更新后的URL
                print(f"\n{Colors.BLUE}正在访问更新后的URL...{Colors.RESET}")
                driver.get(updated_url)
                time.sleep(3)
                
                # 保存更新后的页面截图
                screenshot_path = os.path.join(screenshot_dir, f"updated_page_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                driver.save_screenshot(screenshot_path)
                logger.info(f"已保存更新后页面截图: {screenshot_path}")
                
                # 检查是否成功访问商品页面
                if "item.jd.com" in driver.current_url:
                    print(f"\n{Colors.GREEN}成功访问商品页面{Colors.RESET}")
                    
                    # 提取商品标题
                    try:
                        title_element = driver.find_element(By.CSS_SELECTOR, ".sku-name")
                        if title_element and title_element.text:
                            print(f"商品标题: {Colors.YELLOW}{title_element.text}{Colors.RESET}")
                    except:
                        print(f"{Colors.RED}无法提取商品标题{Colors.RESET}")
                else:
                    print(f"\n{Colors.RED}未成功访问商品页面{Colors.RESET}")
                    print(f"当前URL: {Colors.RED}{driver.current_url}{Colors.RESET}")
            else:
                print(f"\n{Colors.YELLOW}URL未发生变化，可能不需要处理{Colors.RESET}")
        else:
            print(f"\n{Colors.GREEN}URL未发生重定向，无需处理{Colors.RESET}")
            
            # 检查是否在商品页面
            if "item.jd.com" in current_url:
                print(f"{Colors.GREEN}当前已在商品页面{Colors.RESET}")
                
                # 提取商品标题
                try:
                    title_element = driver.find_element(By.CSS_SELECTOR, ".sku-name")
                    if title_element and title_element.text:
                        print(f"商品标题: {Colors.YELLOW}{title_element.text}{Colors.RESET}")
                except:
                    print(f"{Colors.RED}无法提取商品标题{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}当前不在商品页面{Colors.RESET}")
                print(f"当前URL: {Colors.YELLOW}{current_url}{Colors.RESET}")
        
        # 检查页面源代码中的商品ID
        try:
            page_source = driver.page_source
            import re
            id_pattern = r'skuid\s*[:=]\s*[\'"]?(\d+)[\'"]?'
            id_match = re.search(id_pattern, page_source)
            if id_match:
                product_id = id_match.group(1)
                print(f"\n{Colors.BLUE}从页面源代码中提取到商品ID: {Colors.YELLOW}{product_id}{Colors.RESET}")
                
                # 构建标准商品URL
                standard_url = f"https://item.jd.com/{product_id}.html"
                print(f"标准商品URL: {Colors.GREEN}{standard_url}{Colors.RESET}")
            else:
                print(f"\n{Colors.RED}无法从页面源代码中提取商品ID{Colors.RESET}")
        except Exception as e:
            logger.error(f"提取商品ID失败: {e}")
            print(f"\n{Colors.RED}提取商品ID失败: {e}{Colors.RESET}")
        
        print("\n" + "="*60)
        print(f"{Colors.GREEN}{Colors.BOLD}【测试完成】{Colors.RESET}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"\n{Colors.RED}测试失败: {e}{Colors.RESET}")
        
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 关闭浏览器
        driver.quit()
        logger.info("已关闭浏览器")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 运行测试
    try:
        test_url_redirect(args.url, args.headless)
        return 0
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 