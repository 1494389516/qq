#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东商城爬虫运行脚本
"""

import argparse
import os
import sys
import logging
import colorama
import time
import json
import csv
from colorama import Fore, Style
from jd_main import run_spider, Colors
from config import SPIDER_CONFIG
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 初始化colorama
colorama.init(autoreset=True)

# 定义颜色常量
if not hasattr(Colors, 'CYAN'):
    setattr(Colors, 'CYAN', Fore.CYAN)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def show_banner():
    """显示爬虫程序开始Banner"""
    banner = f"""
{Fore.CYAN}{Style.BRIGHT}=================================================
           京东商品爬虫 v1.4.0
=================================================
{Fore.GREEN}✓ 支持自动登录
{Fore.GREEN}✓ 支持扫码/账号密码登录
{Fore.GREEN}✓ 自动处理验证码和滑块
{Fore.GREEN}✓ 支持保存和复用登录状态
{Fore.GREEN}✓ 增强防重定向能力
{Fore.GREEN}✓ 智能验证处理机制
{Fore.GREEN}✓ 支持批量爬取商品
{Fore.GREEN}✓ 实时进度显示
=================================================
{Style.RESET_ALL}
"""
    print(banner)

def show_workflow():
    """显示爬虫工作流程"""
    workflow = f"""
{Fore.CYAN}{Style.BRIGHT}【爬虫工作流程】{Style.RESET_ALL}

{Fore.WHITE}1. {Fore.YELLOW}初始化浏览器{Style.RESET_ALL}
   └─ 配置浏览器指纹，防止被检测

{Fore.WHITE}2. {Fore.YELLOW}访问目标URL{Style.RESET_ALL}
   └─ 加载商品页面

{Fore.WHITE}3. {Fore.YELLOW}处理URL重定向{Style.RESET_ALL}
   └─ 确保使用PC版页面，避免移动端重定向

{Fore.WHITE}4. {Fore.YELLOW}处理验证码/滑块{Style.RESET_ALL}
   └─ 自动或手动完成验证

{Fore.WHITE}5. {Fore.YELLOW}检查登录状态{Style.RESET_ALL}
   └─ 检测页面是否需要登录才能获取完整数据

{Fore.WHITE}6. {Fore.YELLOW}处理登录{Style.RESET_ALL}
   └─ 如需登录，自动启动登录流程（扫码或账号密码）

{Fore.WHITE}7. {Fore.YELLOW}确认登录状态{Style.RESET_ALL}
   └─ 验证登录结果，保存登录cookies

{Fore.WHITE}8. {Fore.YELLOW}重新回到商品页面{Style.RESET_ALL}
   └─ 确保在正确的页面上获取数据

{Fore.WHITE}9. {Fore.YELLOW}提取商品信息{Style.RESET_ALL}
   └─ 包括商品标题、价格、规格、评价等

{Fore.WHITE}10. {Fore.YELLOW}保存数据{Style.RESET_ALL}
   └─ 将商品信息保存为JSON和CSV格式
"""
    print(workflow)

def check_cookies_folder():
    """检查cookies目录状态"""
    cookies_dir = SPIDER_CONFIG['LOGIN']['COOKIES_DIR'] if 'COOKIES_DIR' in SPIDER_CONFIG['LOGIN'] else "cookies"
    
    # 检查目录是否存在
    if not os.path.exists(cookies_dir):
        os.makedirs(cookies_dir)
        print(f"{Colors.YELLOW}✓ 已创建cookies目录: {cookies_dir}{Colors.RESET}")
        return False
    
    # 检查是否有cookie文件
    cookie_files = [f for f in os.listdir(cookies_dir) if f.endswith(".pkl")]
    if not cookie_files:
        print(f"{Colors.YELLOW}ℹ 未找到保存的登录状态，需要重新登录{Colors.RESET}")
        return False
    
    # 显示现有的cookie文件
    print(f"{Colors.GREEN}✓ 找到{len(cookie_files)}个保存的登录状态{Colors.RESET}")
    for i, cookie_file in enumerate(cookie_files):
        username = cookie_file.replace(".pkl", "")
        last_modified = os.path.getmtime(os.path.join(cookies_dir, cookie_file))
        from datetime import datetime
        last_modified_str = datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {i+1}. {Colors.YELLOW}{username}{Colors.RESET} (保存于: {last_modified_str})")
    
    return True

def check_environment():
    """检查环境设置"""
    print(f"{Colors.BLUE}检查环境...{Colors.RESET}")
    
    # 检查chromedriver是否存在
    driver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromedriver.exe")
    if not os.path.exists(driver_path):
        print(f"{Colors.RED}✗ 未找到chromedriver.exe，请确保它位于程序目录中{Colors.RESET}")
        return False
    else:
        print(f"{Colors.GREEN}✓ chromedriver.exe 已就绪{Colors.RESET}")
    
    # 检查输出目录
    results_dir = SPIDER_CONFIG['OUTPUT']['RESULTS_DIR']
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"{Colors.GREEN}✓ 已创建输出目录: {results_dir}{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}✓ 输出目录已就绪: {results_dir}{Colors.RESET}")
    
    # 检查保存的登录状态
    has_cookies = check_cookies_folder()
    
    # 检查配置
    login_enabled = SPIDER_CONFIG['LOGIN']['ENABLE']
    login_mode = "自动登录" if login_enabled else "访客模式"
    print(f"{Colors.GREEN}✓ 当前登录设置: {login_mode}{Colors.RESET}")
    
    if login_enabled:
        username = SPIDER_CONFIG['LOGIN']['USERNAME']
        if username:
            print(f"  • 配置的用户名: {Colors.YELLOW}{username}{Colors.RESET}")
        else:
            print(f"  • {Colors.YELLOW}将使用扫码登录{Colors.RESET}")
    
    # 检查依赖库
    try:
        import tqdm
        print(f"{Colors.GREEN}✓ 进度显示库已安装{Colors.RESET}")
    except ImportError:
        print(f"{Colors.YELLOW}⚠️ 未安装tqdm库，将使用简单进度显示{Colors.RESET}")
    
    return True

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='京东商品爬虫运行脚本')
    
    parser.add_argument('-u', '--url', type=str, 
                        default=SPIDER_CONFIG['TARGET_URL'],
                        help='要爬取的京东商品URL')
    
    parser.add_argument('-o', '--output', type=str,
                        default=SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'],
                        help='结果保存目录')
    
    parser.add_argument('--headless', action='store_true',
                        default=False,
                        help='是否使用无头浏览器模式')
    
    parser.add_argument('--login', action='store_true',
                        default=True,  # 默认启用登录
                        help='是否需要登录')
    
    parser.add_argument('--no-login', action='store_true',
                        default=False,
                        help='禁用登录功能（访客模式）')
    
    parser.add_argument('--username', type=str,
                        default=None,
                        help='京东账号（用于登录）')
    
    parser.add_argument('--password', type=str,
                        default=None,
                        help='京东密码（用于登录）')
    
    parser.add_argument('--debug', action='store_true',
                        default=False,
                        help='是否启用调试模式')
    
    parser.add_argument('--force-pc', action='store_true',
                        default=True,  # 默认强制使用PC版网页
                        help='强制使用PC版网页（防重定向）')
    
    parser.add_argument('--no-force-pc', action='store_false',
                        dest='force_pc',
                        help='不强制使用PC版网页')
    
    parser.add_argument('--max-redirects', type=int,
                        default=SPIDER_CONFIG['URL_REDIRECT']['MAX_REDIRECTS'],
                        help='最大重定向次数')
    
    parser.add_argument('--show-workflow', action='store_true',
                        default=False,
                        help='显示爬虫工作流程')
    
    # 新增批量爬取相关参数
    parser.add_argument('--batch', action='store_true',
                        default=False,
                        help='启用批量爬取模式')
    
    parser.add_argument('--url-file', type=str,
                        default=None,
                        help='包含多个URL的文件路径（每行一个URL）')
    
    parser.add_argument('--max-workers', type=int,
                        default=1,
                        help='批量爬取时的最大线程数（默认为1，建议不超过3）')
    
    parser.add_argument('--delay', type=float,
                        default=5.0,
                        help='批量爬取时每个URL之间的延迟（秒）')
    
    parser.add_argument('--category', type=str,
                        default=None,
                        help='爬取的商品分类（用于结果保存）')
    
    parser.add_argument('--retry', type=int,
                        default=2,
                        help='失败时重试次数')

    return parser.parse_args()

def read_urls_from_file(file_path):
    """从文件中读取URL列表"""
    if not os.path.exists(file_path):
        print(f"{Colors.RED}错误: URL文件不存在 - {file_path}{Colors.RESET}")
        return []
    
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and url.startswith('http'):
                    urls.append(url)
    except Exception as e:
        print(f"{Colors.RED}读取URL文件时出错: {e}{Colors.RESET}")
        return []
    
    print(f"{Colors.GREEN}从文件中读取了{len(urls)}个URL{Colors.RESET}")
    return urls

def process_single_url(url, retry_count=2):
    """处理单个URL的爬取"""
    for attempt in range(retry_count + 1):
        try:
            if attempt > 0:
                print(f"{Colors.YELLOW}第{attempt}次重试爬取: {url}{Colors.RESET}")
            
            # 运行爬虫
            result = run_spider(url)
            return result
        except Exception as e:
            if attempt < retry_count:
                print(f"{Colors.YELLOW}爬取失败，将在5秒后重试: {e}{Colors.RESET}")
                time.sleep(5)
            else:
                print(f"{Colors.RED}爬取失败，已达到最大重试次数: {e}{Colors.RESET}")
                return None

def batch_process_urls(urls, max_workers=1, delay=5.0, retry_count=2):
    """批量处理多个URL"""
    if not urls:
        print(f"{Colors.RED}没有有效的URL可爬取{Colors.RESET}")
        return
    
    results = []
    failed_urls = []
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}【批量爬取】开始处理{len(urls)}个URL{Colors.RESET}")
    print(f"最大线程数: {max_workers}, 延迟: {delay}秒, 重试次数: {retry_count}")
    
    start_time = time.time()
    
    if max_workers <= 1:
        # 单线程顺序处理
        for i, url in enumerate(urls):
            print(f"\n{Colors.BLUE}[{i+1}/{len(urls)}] 开始爬取: {url}{Colors.RESET}")
            try:
                result = process_single_url(url, retry_count)
                if result:
                    results.append(result)
                    print(f"{Colors.GREEN}✓ 爬取成功: {url}{Colors.RESET}")
                else:
                    failed_urls.append(url)
                    print(f"{Colors.RED}✗ 爬取失败: {url}{Colors.RESET}")
            except Exception as e:
                failed_urls.append(url)
                print(f"{Colors.RED}✗ 爬取出错: {url} - {e}{Colors.RESET}")
            
            # 添加延迟，除非是最后一个URL
            if i < len(urls) - 1:
                print(f"{Colors.YELLOW}等待{delay}秒后继续下一个URL...{Colors.RESET}")
                time.sleep(delay)
    else:
        # 多线程并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(process_single_url, url, retry_count): url for url in urls}
            for i, future in enumerate(as_completed(future_to_url)):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"{Colors.GREEN}✓ 爬取成功: {url}{Colors.RESET}")
                    else:
                        failed_urls.append(url)
                        print(f"{Colors.RED}✗ 爬取失败: {url}{Colors.RESET}")
                except Exception as e:
                    failed_urls.append(url)
                    print(f"{Colors.RED}✗ 爬取出错: {url} - {e}{Colors.RESET}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 显示批量爬取结果摘要
    print("\n" + "="*60)
    print(f"{Colors.CYAN}{Colors.BOLD}【批量爬取结果】{Colors.RESET}")
    print(f"总URL数: {len(urls)}")
    print(f"成功数: {Colors.GREEN}{len(results)}{Colors.RESET}")
    print(f"失败数: {Colors.RED}{len(failed_urls)}{Colors.RESET}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每个URL耗时: {total_time/len(urls):.2f}秒")
    print("="*60)
    
    # 如果有失败的URL，保存到文件
    if failed_urls:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        failed_file = f"failed_urls_{timestamp}.txt"
        try:
            with open(failed_file, 'w', encoding='utf-8') as f:
                for url in failed_urls:
                    f.write(f"{url}\n")
            print(f"{Colors.YELLOW}已将失败的URL保存到: {failed_file}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}保存失败URL列表时出错: {e}{Colors.RESET}")
    
    return results

def save_batch_results_summary(results, output_dir):
    """保存批量爬取结果摘要"""
    if not results:
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(output_dir, f"batch_summary_{timestamp}.csv")
    
    try:
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['商品ID', '标题', '价格', '店铺', '评价数', '好评率', 'URL'])
            
            for result in results:
                if isinstance(result, dict):
                    writer.writerow([
                        result.get('product_id', ''),
                        result.get('title', '')[:50],
                        result.get('price', ''),
                        result.get('shop_name', ''),
                        result.get('comments_count', ''),
                        result.get('good_rate', ''),
                        result.get('url', '')
                    ])
        
        print(f"{Colors.GREEN}✓ 已保存批量爬取摘要到: {summary_file}{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}保存批量爬取摘要时出错: {e}{Colors.RESET}")

def main():
    """主函数"""
    # 显示Banner
    show_banner()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果需要显示工作流程
    if args.show_workflow:
        show_workflow()
        return 0

    # 检查环境设置
    if not check_environment():
        print(f"{Colors.RED}环境检查失败，程序退出{Colors.RESET}")
        return 1
    
    # 设置日志级别
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # 根据命令行参数更新配置
    if args.url:
        SPIDER_CONFIG['TARGET_URL'] = args.url
    if args.output:
        SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'] = args.output
    if args.headless:
        SPIDER_CONFIG['BROWSER']['HEADLESS'] = True
    
    # 处理登录配置，优先处理--no-login参数
    if args.no_login:
        SPIDER_CONFIG['LOGIN']['ENABLE'] = False
    else:
        SPIDER_CONFIG['LOGIN']['ENABLE'] = args.login
    
    if args.username:
        SPIDER_CONFIG['LOGIN']['USERNAME'] = args.username
    if args.password:
        SPIDER_CONFIG['LOGIN']['PASSWORD'] = args.password
    
    # 更新重定向处理配置
    SPIDER_CONFIG['URL_REDIRECT']['FORCE_PC'] = args.force_pc
    SPIDER_CONFIG['URL_REDIRECT']['MAX_REDIRECTS'] = args.max_redirects
    
    # 如果指定了分类，更新输出目录
    if args.category:
        category_dir = os.path.join(SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'], args.category)
        SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'] = category_dir
        # 确保目录存在
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
            print(f"{Colors.GREEN}✓ 已创建分类目录: {category_dir}{Colors.RESET}")
    
    # 处理批量爬取模式
    if args.batch or args.url_file:
        urls = []
        
        # 从文件读取URL
        if args.url_file:
            urls = read_urls_from_file(args.url_file)
        
        # 如果还有命令行指定的URL，也添加进来
        if args.url and args.url.startswith('http'):
            if args.url not in urls:
                urls.append(args.url)
        
        if not urls:
            print(f"{Colors.RED}错误: 批量模式下未提供有效的URL{Colors.RESET}")
            print(f"请使用 --url-file 参数指定包含URL的文件，或使用 --url 参数指定单个URL")
            return 1
        
        # 执行批量爬取
        results = batch_process_urls(
            urls, 
            max_workers=args.max_workers, 
            delay=args.delay,
            retry_count=args.retry
        )
        
        # 保存批量爬取结果摘要
        if results:
            save_batch_results_summary(results, SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'])
        
        return 0
    
    # 单个URL模式
    url = SPIDER_CONFIG['TARGET_URL']
    
    # 检查URL
    if not url or not url.startswith("http"):
        print(f"{Colors.RED}错误: 未指定有效的京东商品URL{Colors.RESET}")
        print(f"请使用 --url 参数指定京东商品URL，例如:")
        print(f"{Colors.GREEN}python run_jd_spider.py --url https://item.jd.com/100037976126.html{Colors.RESET}")
        return 1
    
    # 显示爬取配置信息
    print(f"{Colors.CYAN}爬取配置:{Colors.RESET}")
    print(f"  • 登录模式: {Colors.YELLOW}{'启用' if SPIDER_CONFIG['LOGIN']['ENABLE'] else '禁用'}{Colors.RESET}")
    print(f"  • 强制PC版: {Colors.YELLOW}{'启用' if SPIDER_CONFIG['URL_REDIRECT'].get('FORCE_PC', True) else '禁用'}{Colors.RESET}")
    print(f"  • 浏览器模式: {Colors.YELLOW}{'无头模式' if SPIDER_CONFIG['BROWSER']['HEADLESS'] else '可视模式'}{Colors.RESET}")
    
    # 添加浏览器配置信息
    driver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromedriver.exe")
    if os.path.exists(driver_path):
        print(f"{Colors.GREEN}✓ 使用本地chromedriver: {driver_path}{Colors.RESET}")
    
    # 提示可以查看工作流程
    print(f"\n{Colors.YELLOW}提示: 使用 --show-workflow 参数可查看爬虫工作流程{Colors.RESET}")
    
    # 显示简明工作流程
    print(f"\n{Colors.CYAN}爬虫将按以下步骤运行:{Colors.RESET}")
    print(f"  1. 访问目标URL → 2. 处理验证 → 3. 检查登录 → ")
    print(f"  4. 处理登录 → 5. 回到商品页 → 6. 提取数据 → 7. 保存结果")
    
    # 运行爬虫
    try:
        print("\n" + "="*60)
        print(f"{Colors.GREEN}{Colors.BOLD}【开始执行爬虫】{Colors.RESET}")
        print(f"目标URL: {Colors.CYAN}{url}{Colors.RESET}")
        print("="*60 + "\n")
        
        # 使用重试机制
        result = process_single_url(url, retry_count=args.retry)
        
        if result:
            print(f"\n{Colors.GREEN}✓ 爬虫运行完成{Colors.RESET}")
        else:
            print(f"\n{Colors.YELLOW}⚠️ 爬虫运行完成，但可能有数据缺失{Colors.RESET}")
        
        logger.info("爬虫运行完成")
        return 0
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}用户中断，程序退出{Colors.RESET}")
    except Exception as e:
        logger.error(f"爬虫运行失败: {e}")
        if args.debug:
            import traceback
            print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 