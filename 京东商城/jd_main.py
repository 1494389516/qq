#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东商城爬虫主入口
"""

import os
import sys
import time
import argparse
import logging
import json
import random
from datetime import datetime
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fake_useragent import UserAgent
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import re
from tqdm import tqdm

# 检查是否在Windows环境中，以支持彩色输出
try:
    import colorama
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

# 彩色输出配置
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m" if HAS_COLORAMA else ""
    GREEN = "\033[92m" if HAS_COLORAMA else ""
    YELLOW = "\033[93m" if HAS_COLORAMA else ""
    BLUE = "\033[94m" if HAS_COLORAMA else ""
    BOLD = "\033[1m" if HAS_COLORAMA else ""
    CYAN = "\033[96m" if HAS_COLORAMA else ""

from config import SPIDER_CONFIG
from 京东商城 import JDVerificationHandler
from jd_login import JDLogin

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局变量，用于在模块之间共享商品URL
original_product_url = None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='京东商品爬虫')
    
    parser.add_argument('-u', '--url', type=str, 
                        default=SPIDER_CONFIG['TARGET_URL'],
                        help='要爬取的京东商品URL')
    
    parser.add_argument('-o', '--output', type=str,
                        default=SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'],
                        help='结果保存目录')
    
    parser.add_argument('--headless', action='store_true',
                        default=SPIDER_CONFIG['BROWSER']['HEADLESS'],
                        help='是否使用无头浏览器模式')
    
    parser.add_argument('--login', action='store_true',
                        default=False,
                        help='是否需要登录')
    
    parser.add_argument('--username', type=str,
                        default=None,
                        help='京东账号（用于登录）')
    
    parser.add_argument('--password', type=str,
                        default=None,
                        help='京东密码（用于登录）')
    
    return parser.parse_args()

def update_config(args):
    """根据命令行参数更新配置"""
    SPIDER_CONFIG['TARGET_URL'] = args.url
    SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'] = args.output
    SPIDER_CONFIG['BROWSER']['HEADLESS'] = args.headless
    SPIDER_CONFIG['LOGIN'] = {
        'ENABLE': args.login,
        'USERNAME': args.username,
        'PASSWORD': args.password,
        'COOKIES_DIR': 'cookies',
        'QRCODE_TIMEOUT': 60,  # 使用默认配置
        'QRCODE_REFRESH': True
    }

def init_browser():
    """初始化浏览器"""
    try:
        print(f"{Colors.BLUE}正在初始化浏览器，请稍候...{Colors.RESET}")
        
        # 获取随机UA
        ua = UserAgent()
        # 使用高端PC设备的UA
        random_ua = ua.chrome if hasattr(ua, 'chrome') else ua.random
        
        # 使用更强的反检测方法
        driver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromedriver.exe")
        
        # 准备高级配置
        options = uc.ChromeOptions()
        
        # 禁用自动化标志
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-automation")
        
        # 禁用各种可能暴露自动化的功能
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        
        # 自定义浏览器参数
        options.add_argument("--lang=zh-CN")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-browser-side-navigation")
        options.add_argument("--disable-features=IsolateOrigins,site-per-process")
        
        # 设置UA
        options.add_argument(f"--user-agent={random_ua}")
        
        # 设置窗口大小
        window_size = SPIDER_CONFIG['BROWSER']['WINDOW_SIZE']
        options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
        
        # 增加随机性
        options.add_argument(f"--window-position={random.randint(0, 100)},{random.randint(0, 100)}")
        
        # 添加显式的强制PC网站参数
        options.add_argument("--user-agent-string=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
        
        # 增加随机指纹信息
        platform_string = f"Win32|{random.randint(5, 10)}.{random.randint(0, 9)}|x86_64"
        options.add_argument(f"--platform={platform_string}")
        
        # 禁用WebRTC，防止IP泄露
        options.add_argument("--disable-webrtc")
        
        # 设置屏幕色深，增加随机性
        options.add_argument(f"--color-depth={random.choice([16, 24, 32])}")
        
        # 设置代理（如果配置了）
        if SPIDER_CONFIG['PROXY']['ENABLE'] and SPIDER_CONFIG['PROXY']['HOST'] and SPIDER_CONFIG['PROXY']['PORT']:
            proxy_type = SPIDER_CONFIG['PROXY']['TYPE']
            proxy_host = SPIDER_CONFIG['PROXY']['HOST']
            proxy_port = SPIDER_CONFIG['PROXY']['PORT']
            proxy_url = f"{proxy_type}://{proxy_host}:{proxy_port}"
            options.add_argument(f'--proxy-server={proxy_url}')
            print(f"{Colors.YELLOW}使用代理: {proxy_url}{Colors.RESET}")
                
        # 实验性参数：将Mobile设置为false - 移除不兼容的选项
        # options.add_experimental_option("mobileEmulation", {"mobile": False})
        
        # 尝试禁用自动化检测 - 移除不兼容的选项
        # options.add_experimental_option("excludeSwitches", ["enable-automation"])
        # options.add_experimental_option("useAutomationExtension", False)
        
        # 设置WebGL参数
        options.add_argument("--use-gl=desktop")
        
        # 设置Cookie偏好
        options.add_argument("--enable-cookies")
        options.add_argument("--cookies-without-same-site-must-be-secure=false")
        
        # 设置referrer
        options.add_argument("--referrer=https://www.jd.com/")
        
        print(f"{Colors.GREEN}使用本地ChromeDriver: {driver_path}{Colors.RESET}")
        
        # 使用undetected_chromedriver的核心功能：绕过webdriver检测
        driver = uc.Chrome(
            driver_executable_path=driver_path if os.path.exists(driver_path) else None,
            options=options,
            use_subprocess=True,
            version_main=None,  # 自动检测Chrome版本
            headless=SPIDER_CONFIG['BROWSER']['HEADLESS'],
            suppress_welcome=True  # 抑制欢迎页面
        )
        
        # 设置页面加载超时和隐式等待时间
        driver.set_page_load_timeout(SPIDER_CONFIG['BROWSER']['PAGE_LOAD_TIMEOUT'])
        driver.implicitly_wait(SPIDER_CONFIG['BROWSER']['IMPLICIT_WAIT'])
        
        # 注入进阶反检测JavaScript
        stealth_js = """
        // 高级反检测脚本 - 2025版

        // 基础WebDriver检测防御
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
            configurable: true
        });
        
        // 模拟真实用户的浏览器插件
        if (window.navigator.plugins) {
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    let plugins = [];
                    // 添加常见插件
                    const pluginNames = [
                        'Chrome PDF Plugin', 'Chrome PDF Viewer', 'Native Client',
                        'Adobe Acrobat', 'QuickTime Plugin', 'Windows Media Player', 
                        'Shockwave Flash', 'Java Applet Plug-in'
                    ];
                    const pluginDescs = [
                        'Portable Document Format', 'Chromium PDF Viewer',
                        'Native Client Executable', 'Adobe PDF Reader',
                        'Video Player', 'Media Player',
                        'Adobe Flash Player', 'Java Platform Plugin'
                    ];
                    for (let i = 0; i < 8; i++) {
                        plugins.push({
                            name: pluginNames[i % pluginNames.length],
                            description: pluginDescs[i % pluginDescs.length],
                            filename: `plugin${i}.dll`,
                            version: `${Math.floor(Math.random() * 10)}.${Math.floor(Math.random() * 10)}.${Math.floor(Math.random() * 10)}`,
                            length: i + 1
                        });
                    }
                    plugins.namedItem = (name) => plugins.find(p => p.name === name) || null;
                    plugins.refresh = () => {};
                    plugins.item = (i) => plugins[i] || null;
                    return plugins;
                }
            });
        }
        
        // 修改用户首选项
        if (window.navigator.languages !== undefined) {
            Object.defineProperty(navigator, 'languages', {
                get: () => ['zh-CN', 'zh', 'en-US', 'en']
            });
        }
        
        // 拦截并模拟权限查询
        try {
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => {
                if (parameters.name === 'notifications')
                    return Promise.resolve({ state: Notification.permission, onchange: null });
                if (parameters.name === 'clipboard-read' || parameters.name === 'clipboard-write')
                    return Promise.resolve({ state: 'granted', onchange: null });
                if (parameters.name === 'geolocation')
                    return Promise.resolve({ state: 'prompt', onchange: null });
                if (parameters.name === 'camera' || parameters.name === 'microphone')
                    return Promise.resolve({ state: 'prompt', onchange: null });
                return originalQuery(parameters);
            };
        } catch(e) {}
        
        // 模拟触摸能力
        Object.defineProperty(navigator, 'maxTouchPoints', {
            get: () => Math.floor(Math.random() * 5) + 1
        });
        
        // 模拟设备内存
        Object.defineProperty(navigator, 'deviceMemory', {
            get: () => Math.floor(Math.random() * 8) + 4
        });
        
        // 模拟硬件并发
        Object.defineProperty(navigator, 'hardwareConcurrency', {
            get: () => Math.floor(Math.random() * 4) + 4
        });
        
        // 模拟连接信息
        if (navigator.connection) {
            Object.defineProperties(navigator.connection, {
                effectiveType: {
                    get: () => ['slow-2g', '2g', '3g', '4g'][Math.floor(Math.random() * 4)]
                },
                saveData: {
                    get: () => Math.random() > 0.5
                },
                rtt: {
                    get: () => Math.floor(Math.random() * 100) + 50
                }
            });
        }
        
        // 高级Chrome自动化控制器覆盖
        window.chrome = {
            app: {
                isInstalled: false,
                getDetails: () => { return null; },
                getIsInstalled: () => { return false; },
                runningState: () => { return 'cannot_run'; }
            },
            webstore: {
                onInstallStageChanged: {},
                onDownloadProgress: {},
                install: (url, onSuccess, onFailure) => { 
                    onFailure('Not available');
                }
            },
            runtime: {
                PlatformOs: {
                    MAC: 'mac',
                    WIN: 'win',
                    ANDROID: 'android',
                    CROS: 'cros',
                    LINUX: 'linux',
                    OPENBSD: 'openbsd',
                },
                PlatformArch: {
                    ARM: 'arm',
                    X86_32: 'x86-32',
                    X86_64: 'x86-64',
                },
                PlatformNaclArch: {
                    ARM: 'arm',
                    X86_32: 'x86-32',
                    X86_64: 'x86-64',
                },
                RequestUpdateCheckStatus: {
                    THROTTLED: 'throttled',
                    NO_UPDATE: 'no_update',
                    UPDATE_AVAILABLE: 'update_available',
                },
                OnInstalledReason: {
                    INSTALL: 'install',
                    UPDATE: 'update',
                    CHROME_UPDATE: 'chrome_update',
                    SHARED_MODULE_UPDATE: 'shared_module_update',
                },
                OnRestartRequiredReason: {
                    APP_UPDATE: 'app_update',
                    OS_UPDATE: 'os_update',
                    PERIODIC: 'periodic',
                },
                connect: () => {},
                sendMessage: () => {},
            },
            csi: () => {},
            loadTimes: () => {},
        };
        
        // 防御Canvas指纹
        const originalGetContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(type, attrs) {
            const context = originalGetContext.call(this, type, attrs);
            if (context && type === '2d') {
                const originalFillText = context.fillText;
                context.fillText = function() {
                    arguments[0] = arguments[0].toString().replace(/[a-zA-Z]/g, c => 
                       String.fromCharCode((c <= 'Z' ? 90 : 122) >= (c = c.charCodeAt(0) + 1) ? c : c - 26));
                    return originalFillText.apply(this, arguments);
                };
                const originalRect = context.rect;
                context.rect = function() {
                    arguments[0] = arguments[0] + 0.00000001;
                    return originalRect.apply(this, arguments);
                };
            }
            return context;
        };
        
        // 伪装Audio指纹
        const originalGetChannelData = AudioBuffer.prototype.getChannelData;
        AudioBuffer.prototype.getChannelData = function() {
            const result = originalGetChannelData.apply(this, arguments);
            if (result && result.length > 0) {
                for (let i = 0; i < 500; i++) {
                    let index = Math.floor(Math.random() * result.length);
                    result[index] = result[index] + Math.random() * 0.0000001;
                }
            }
            return result;
        };
        
        // 模拟真实窗口属性
        Object.defineProperties(window, {
            outerWidth: { get: () => window.innerWidth },
            outerHeight: { get: () => window.innerHeight },
            screenX: { get: () => 0 + Math.floor(Math.random() * 10) },
            screenY: { get: () => 0 + Math.floor(Math.random() * 10) },
        });
        
        // 伪装性能测量
        window.performance.now = () => {
            return Date.now() - Math.floor(Math.random() * 10);
        };
        
        // 伪装屏幕信息
        if (window.screen) {
            Object.defineProperties(window.screen, {
                colorDepth: { get: () => 24 },
                pixelDepth: { get: () => 24 }
            });
        }
        
        // 覆盖WebDriver特定属性
        Object.defineProperty(document, 'webdriver', {
            get: () => false,
            enumerable: true,
            configurable: true
        });
        
        // 清除automation属性
        delete window.__nightmare;
        delete window.callPhantom;
        delete window._phantom;
        delete window.phantom;
        delete window.domAutomation;
        delete window.domAutomationController;
        delete window.webdriver;
        delete window.__webdriverFunc;
        delete window.Selenium;
        delete window.__selenium_evaluate;
        delete window.__selenium_unwrapped;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
        
        // 最终检查和覆盖
        window.navigator.webdriver = false;
        Object.defineProperty(window, "navigator", {
            value: navigator,
            writable: false
        });
        
        console.log("反爬虫防御系统已启动");
        """
        
        try:
            driver.execute_script(stealth_js)
            print(f"{Colors.GREEN}已注入高级反检测脚本{Colors.RESET}")
            
            # 注入额外的上下文和常规动作，让浏览器看起来更真实
            driver.execute_script("""
                // 添加一些随机的本地存储，模拟真实用户
                let faker = {
                    lastVisit: new Date().toISOString(),
                    visitCount: Math.floor(Math.random() * 10) + 1,
                    preference: ['推荐', '手机', '电脑', '家电', '数码'][Math.floor(Math.random() * 5)]
                };
                localStorage.setItem('user_pref', JSON.stringify(faker));
                localStorage.setItem('jd_history_count', Math.floor(Math.random() * 20).toString());
                
                // 模拟正常的滚动行为
                function simulateNormalBrowsing() {
                    if(Math.random() > 0.7) {
                        window.scrollBy(0, Math.floor(Math.random() * 100) + 50);
                    }
                }
                
                // 定时执行，模拟用户正在与页面交互
                setInterval(simulateNormalBrowsing, Math.floor(Math.random() * 2000) + 1000);
            """)
        except Exception as e:
            logger.warning(f"注入反检测脚本失败: {e}")
        
        # 模拟真实浏览行为，在启动后随机移动鼠标和点击
        try:
            # 注入真实用户行为模拟
            simulate_human_behavior(driver)
        except Exception as e:
            logger.warning(f"模拟人类行为失败: {e}")
        
        # 随机延迟，模拟真实用户
        delay = random.uniform(2, 5)
        print(f"{Colors.YELLOW}模拟人类行为，等待{delay:.1f}秒...{Colors.RESET}")
        time.sleep(delay)
        
        print(f"{Colors.GREEN}浏览器初始化成功{Colors.RESET}")
        return driver
        
    except Exception as e:
        logger.error(f"初始化浏览器失败: {e}")
        print(f"{Colors.RED}浏览器初始化失败: {e}{Colors.RESET}")
        
        # 尝试使用最简单的方法初始化
        try:
            print(f"{Colors.YELLOW}尝试使用备用方法初始化浏览器...{Colors.RESET}")
            driver = uc.Chrome()
            print(f"{Colors.GREEN}备用方法初始化成功{Colors.RESET}")
            return driver
        except Exception as e2:
            logger.error(f"备用方法初始化浏览器也失败: {e2}")
            print(f"{Colors.RED}备用方法初始化也失败: {e2}{Colors.RESET}")
            raise Exception("无法初始化浏览器，请检查Chrome和ChromeDriver版本是否匹配")

def check_login_status(driver):
    """检查当前的登录状态
    
    Args:
        driver: WebDriver实例
        
    Returns:
        tuple: (是否已登录, 用户名)
    """
    try:
        # 方法1: 检查nickname元素
        try:
            nickname_element = driver.find_element(By.CLASS_NAME, "nickname")
            if nickname_element and nickname_element.text and "请登录" not in nickname_element.text:
                return True, nickname_element.text
        except (NoSuchElementException, TimeoutException):
            pass
        
        # 方法2: 检查其他可能的登录状态元素
        try:
            login_status = driver.find_element(By.CSS_SELECTOR, ".user-info .user-level")
            if login_status:
                # 尝试获取用户名
                try:
                    username_elem = driver.find_element(By.CSS_SELECTOR, ".user-info .nickname")
                    return True, username_elem.text
                except:
                    return True, "京东用户"
        except (NoSuchElementException, TimeoutException):
            pass
        
        # 方法3: 使用JavaScript检测
        try:
            # 注入并执行登录检测脚本
            js_code = """
            // 返回当前登录状态
            let nickname = document.querySelector('.nickname');
            return nickname && nickname.innerText ? nickname.innerText : "";
            """
            result = driver.execute_script(js_code)
            if result:
                return True, result
        except:
            pass
        
        # 方法4: 检查是否有"请登录"文本, 如果没有可能表示登录成功
        try:
            login_text = driver.find_element(By.LINK_TEXT, "请登录")
            # 仍有"请登录"文本，未登录
            return False, None
        except (NoSuchElementException, TimeoutException):
            # 检查页面上是否有其他登录状态指标
            page_source = driver.page_source
            if "欢迎登录" not in page_source and "已登录" in page_source:
                return True, "京东用户"
        
        return False, None
    except Exception as e:
        logger.error(f"检查登录状态失败: {e}")
        return False, None

def handle_login(driver):
    """处理登录
    
    Args:
        driver: WebDriver实例
        
    Returns:
        bool: 是否登录成功
    """
    # 如果不需要登录，直接返回
    if not SPIDER_CONFIG['LOGIN']['ENABLE']:
        logger.info("未启用登录功能")
        return True
    
    # 检查当前是否已经登录
    is_logged_in, username = check_login_status(driver)
    if is_logged_in:
        logger.info(f"已经登录，无需重新登录: {username}")
        print(f"\n{Colors.GREEN}已检测到登录状态: {username}{Colors.RESET}")
        return True
    
    # 初始化登录处理器
    login_handler = JDLogin(driver)
    
    # 自动登录
    username = SPIDER_CONFIG['LOGIN']['USERNAME']
    password = SPIDER_CONFIG['LOGIN']['PASSWORD']
    
    logger.info("开始登录京东...")
    print("\n" + "="*60)
    print(f"{Colors.BLUE}{Colors.BOLD}【开始登录】京东账号{Colors.RESET}")
    
    # 显示登录提示和说明
    print(f"{Colors.YELLOW}登录说明:{Colors.RESET}")
    if username and password:
        print(f"  • 将使用账号 {Colors.CYAN}{username}{Colors.RESET} 登录")
        print(f"  • 如遇验证码，系统会自动处理")
    else:
        print(f"  • 将使用{Colors.CYAN}扫码登录{Colors.RESET}方式")
        print(f"  • {Colors.YELLOW}请注意查看浏览器窗口，准备扫描二维码{Colors.RESET}")
        print(f"  • 请在30秒内完成扫码，否则二维码会失效")
        print(f"  • 系统会自动检测登录状态，登录成功后将继续爬取")
    
    print(f"{Colors.CYAN}正在准备登录环境，请稍候...{Colors.RESET}")
    print("="*60 + "\n")
    
    # 保存当前cookies日志，用于调试对比
    try:
        logger.info(f"登录前的cookies数: {len(driver.get_cookies())}")
        cookie_domains = set(cookie['domain'] for cookie in driver.get_cookies() if 'domain' in cookie)
        logger.info(f"登录前的cookie域: {cookie_domains}")
    except:
        logger.error("保存登录前cookies日志失败")
    
    # 添加进度条显示
    import sys
    print(f"{Colors.YELLOW}请在浏览器窗口进行操作，此处等待登录结果...{Colors.RESET}")
    sys.stdout.write(f"{Colors.CYAN}[            ] 等待登录中{Colors.RESET}")
    sys.stdout.flush()
    
    # 启动登录过程
    login_start_time = time.time()
    success = login_handler.auto_login(username, password)
    login_time = time.time() - login_start_time
    
    # 清除进度条
    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()
    
    if success:
        logger.info(f"登录成功，耗时 {login_time:.1f} 秒")
        print(f"\n{Colors.GREEN}✓ 登录成功！耗时 {login_time:.1f} 秒{Colors.RESET}")
        
        # 保存登录后的cookies日志
        try:
            cookies = driver.get_cookies()
            logger.info(f"登录后的cookies数: {len(cookies)}")
            cookie_domains = set(cookie['domain'] for cookie in cookies if 'domain' in cookie)
            logger.info(f"登录后的cookie域: {cookie_domains}")
            print(f"{Colors.GREEN}✓ 已保存{len(cookies)}个Cookie{Colors.RESET}")
        except Exception as e:
            logger.error(f"保存登录后cookies日志失败: {e}")
        
        # 在页面顶部添加登录状态提示
        try:
            is_logged_in, username = check_login_status(driver)
            if is_logged_in:
                print("\n" + "="*60)
                print(f"{Colors.GREEN}{Colors.BOLD}【登录状态】{Colors.RESET}当前登录账号: {Colors.YELLOW}{username}{Colors.RESET}")
                print(f"所有爬取的数据将基于该账号权限")
                print(f"{Colors.GREEN}✓ Cookie状态已保存，下次可直接使用{Colors.RESET}")
                print("="*60 + "\n")
            else:
                logger.warning("登录后状态检查失败，可能登录不成功")
        except Exception as e:
            logger.error(f"登录后状态检查异常: {e}")
    else:
        logger.warning(f"登录失败，耗时 {login_time:.1f} 秒")
        print("\n" + "="*60)
        print(f"{Colors.RED}{Colors.BOLD}【登录失败】{Colors.RESET}")
        print(f"登录尝试耗时 {login_time:.1f} 秒")
        
        if not username and not password:
            # 扫码登录失败的可能原因
            print(f"\n{Colors.YELLOW}可能的失败原因：{Colors.RESET}")
            print(f"1. 二维码过期（未在有效时间内完成扫描）")
            print(f"2. 取消了登录确认")
            print(f"3. 京东服务器繁忙")
            print("\n尝试以下解决方案:")
            print(f"• 重新运行程序: {Colors.GREEN}python run_jd_spider.py --login{Colors.RESET}")
            print(f"• 尝试使用账号密码登录（在config.py中配置）")
        else:
            # 账号密码登录失败的可能原因
            print(f"\n{Colors.YELLOW}可能的失败原因：{Colors.RESET}")
            print(f"1. 账号或密码错误")
            print(f"2. 需要手机验证码但未能正确处理")
            print(f"3. 账号被风控")
            print("\n尝试以下解决方案:")
            print(f"• 检查账号密码是否正确")
            print(f"• 尝试扫码登录: {Colors.GREEN}python run_jd_spider.py --login{Colors.RESET}")
        
        print(f"\n{Colors.YELLOW}将以未登录状态继续爬取，某些数据可能无法获取{Colors.RESET}")
        print("="*60 + "\n")
        
        # 再次提示
        print(f"{Colors.CYAN}正在以访客模式继续...{Colors.RESET}")
    
    return success

def simulate_human_behavior(driver):
    """模拟人类浏览行为"""
    logger.info("开始模拟人类浏览行为")
    print(f"{Colors.YELLOW}模拟真实用户行为...{Colors.RESET}")
    
    try:
        # 创建一个模拟真实行为的JavaScript函数
        js_human_simulation = """
        // 高级人类行为模拟器
        (function() {
            // 保存元素位置信息，避免太多DOM查询
            let interactiveElements = [];
            
            // 查找可交互元素
            function findInteractiveElements() {
                // 收集所有可能的交互元素
                const selectors = [
                    'a', 'button', '.btn', '.button', 
                    'li[role="tab"]', '[tabindex="0"]', '.tab', '.menu-item'
                ];
                
                let elements = [];
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => {
                        // 检查元素是否可见且在视口内
                        const rect = el.getBoundingClientRect();
                        const isVisible = rect.width > 0 && rect.height > 0 && 
                                         rect.top < window.innerHeight && rect.left < window.innerWidth &&
                                         getComputedStyle(el).display !== 'none' &&
                                         getComputedStyle(el).visibility !== 'hidden';
                        
                        // 排除类似登录、购买、添加购物车等敏感操作按钮
                        const text = (el.textContent || '').toLowerCase();
                        const isSensitive = /登录|注册|购买|下单|收藏|加入购物车|提交|支付/.test(text);
                        
                        if (isVisible && !isSensitive) {
                            elements.push({
                                element: el,
                                rect: rect
                            });
                        }
                    });
                });
                
                return elements;
            }
            
            // 模拟自然的鼠标移动
            function simulateMouseMovement(startX, startY, endX, endY, steps = 20) {
                return new Promise(resolve => {
                    const deltaX = (endX - startX) / steps;
                    const deltaY = (endY - startY) / steps;
                    
                    let step = 0;
                    
                    function moveStep() {
                        if (step <= steps) {
                            // 引入一些随机性，模拟人类不精确的移动
                            const wobbleX = Math.random() * 3 - 1.5;
                            const wobbleY = Math.random() * 3 - 1.5;
                            
                            const currentX = startX + deltaX * step + wobbleX;
                            const currentY = startY + deltaY * step + wobbleY;
                            
                            // 创建鼠标移动事件
                            const event = new MouseEvent('mousemove', {
                                view: window,
                                bubbles: true,
                                cancelable: true,
                                clientX: currentX,
                                clientY: currentY
                            });
                            
                            document.dispatchEvent(event);
                            
                            // 速度变化：开始慢，中间快，结束慢
                            let delay;
                            if (step < steps * 0.2 || step > steps * 0.8) {
                                delay = 15 + Math.random() * 25; // 慢
                            } else {
                                delay = 5 + Math.random() * 15; // 快
                            }
                            
                            step++;
                            setTimeout(moveStep, delay);
                        } else {
                            resolve();
                        }
                    }
                    
                    moveStep();
                });
            }
            
            // 模拟自然的滚动
            function simulateNaturalScrolling(distance, duration = 500) {
                return new Promise(resolve => {
                    const startTime = Date.now();
                    const endTime = startTime + duration;
                    
                    function scrollStep() {
                        const now = Date.now();
                        const elapsedTime = now - startTime;
                        const progress = Math.min(elapsedTime / duration, 1);
                        
                        // 平滑的缓动函数
                        const easeInOutCubic = progress < 0.5
                            ? 4 * progress * progress * progress
                            : 1 - Math.pow(-2 * progress + 2, 3) / 2;
                        
                        const currentScroll = easeInOutCubic * distance;
                        window.scrollBy(0, currentScroll - (window.scrollY || window.pageYOffset));
                        
                        if (now < endTime) {
                            requestAnimationFrame(scrollStep);
                        } else {
                            resolve();
                        }
                    }
                    
                    requestAnimationFrame(scrollStep);
                });
            }
            
            // 模拟眼睛扫描
            function simulateEyeScanning(duration = 2000) {
                // 这里我们只是等待一段时间，模拟用户在阅读页面内容
                return new Promise(resolve => {
                    setTimeout(resolve, duration);
                });
            }
            
            // 执行一系列人类行为模拟
            async function simulateHumanBehavior() {
                try {
                    // 收集页面上的交互元素
                    interactiveElements = findInteractiveElements();
                    
                    // 1. 初始扫视页面
                    await simulateEyeScanning(1000 + Math.random() * 1000);
                    
                    // 2. 缓慢滚动一点
                    const initialScroll = 100 + Math.random() * 300;
                    await simulateNaturalScrolling(initialScroll, 800 + Math.random() * 500);
                    
                    // 3. 暂停查看内容
                    await simulateEyeScanning(1500 + Math.random() * 1500);
                    
                    // 4. 随机移动鼠标到某个元素（但不点击）
                    if (interactiveElements.length > 0) {
                        const randomIndex = Math.floor(Math.random() * interactiveElements.length);
                        const targetElement = interactiveElements[randomIndex];
                        
                        const startX = Math.random() * window.innerWidth;
                        const startY = Math.random() * (window.innerHeight / 2);
                        const endX = targetElement.rect.x + targetElement.rect.width / 2;
                        const endY = targetElement.rect.y + targetElement.rect.height / 2;
                        
                        await simulateMouseMovement(startX, startY, endX, endY);
                    }
                    
                    // 5. 继续滚动查看更多内容
                    const mainScroll = 300 + Math.random() * 500;
                    await simulateNaturalScrolling(mainScroll, 1000 + Math.random() * 800);
                    
                    // 6. 暂停一会儿，假装在阅读
                    await simulateEyeScanning(2000 + Math.random() * 2000);
                    
                    // 7. 最终滚动（有时向上滚动一点）
                    const finalScroll = Math.random() > 0.3 ? (200 + Math.random() * 400) : -(100 + Math.random() * 200);
                    await simulateNaturalScrolling(finalScroll, 900 + Math.random() * 600);
                    
                    console.log("成功完成人类行为模拟");
                    return true;
                } catch (error) {
                    console.error("人类行为模拟过程中出错:", error);
                    return false;
                }
            }
            
            // 开始执行模拟
            return simulateHumanBehavior();
        })();
        """
        
        # 执行JavaScript代码
        driver.execute_script(js_human_simulation)
        
        # 等待JS执行完成
        time.sleep(random.uniform(3, 5))
        
        # Python层面额外的行为模拟
        
        # 1. 随机滚动页面 (更自然的模式)
        scroll_steps = random.randint(3, 8)
        for i in range(scroll_steps):
            # 每次滚动距离不同，形成一个更自然的浏览模式
            if i < scroll_steps / 2:
                # 前半段，滚动距离逐渐增加
                scroll_height = random.randint(100, 400) * (i + 1) / scroll_steps
            else:
                # 后半段，滚动距离逐渐减少
                scroll_height = random.randint(100, 400) * (scroll_steps - i) / scroll_steps
                
            # 有时会向上滚动一点，好像在回看内容
            if random.random() < 0.2 and i > 1:
                scroll_height = -scroll_height / 2
                
            driver.execute_script(f"window.scrollBy(0, {int(scroll_height)});")
            time.sleep(random.uniform(0.8, 2.2))
            
            # 有时候停下来"阅读"
            if random.random() < 0.4:
                read_time = random.uniform(1, 3)
                logger.info(f"模拟阅读，停留 {read_time:.1f} 秒")
                time.sleep(read_time)
        
        # 2. 随机移动鼠标到非敏感区域
        try:
            # 获取安全区域的元素（不包含登录、购买等操作的元素）
            safe_elements_js = """
            return Array.from(document.querySelectorAll('div, span, p, li, h1, h2, h3, img'))
                .filter(el => {
                    // 检查元素是否可见
                    const rect = el.getBoundingClientRect();
                    const isVisible = rect.width > 0 && rect.height > 0 && 
                                     rect.top < window.innerHeight && rect.left < window.innerWidth;
                    // 排除可能是按钮的元素
                    const isSafeElement = !el.tagName.match(/button|a/i) && 
                                          !el.getAttribute('role')?.match(/button|link/i);
                    // 排除包含敏感文字的元素
                    const text = (el.textContent || '').toLowerCase();
                    const isSafeText = !text.match(/登录|注册|购买|下单|收藏|加入购物车|提交|支付/);
                    
                    return isVisible && isSafeElement && isSafeText;
                })
                .map(el => {
                    const rect = el.getBoundingClientRect();
                    return {
                        x: rect.left + rect.width / 2,
                        y: rect.top + rect.height / 2,
                        width: rect.width,
                        height: rect.height
                    };
                })
                .filter(rect => rect.x > 0 && rect.y > 0);
            """
            
            safe_elements = driver.execute_script(safe_elements_js)
            
            if safe_elements and len(safe_elements) > 0:
                # 选择几个元素进行鼠标移动模拟
                num_moves = random.randint(2, 5)
                for _ in range(num_moves):
                    if not safe_elements:
                        break
                        
                    target = random.choice(safe_elements)
                    
                    # 模拟鼠标移动 (加点随机性)
                    target_x = target['x'] + random.randint(-10, 10)
                    target_y = target['y'] + random.randint(-10, 10)
                    
                    # 创建一个自然的鼠标移动
                    move_steps = random.randint(5, 15)
                    for step in range(move_steps):
                        # 计算当前步骤的位置 (使用缓动函数)
                        progress = step / move_steps
                        eased_progress = (2 * progress * progress) if progress < 0.5 else (1 - pow(-2 * progress + 2, 2) / 2)
                        
                        current_x = target_x * eased_progress
                        current_y = target_y * eased_progress
                        
                        # 添加一点随机抖动
                        wobble_x = random.randint(-2, 2)
                        wobble_y = random.randint(-2, 2)
                        
                        # 分发鼠标移动事件
                        driver.execute_script(f"""
                            document.dispatchEvent(new MouseEvent('mousemove', {{
                                clientX: {current_x + wobble_x},
                                clientY: {current_y + wobble_y},
                                bubbles: true
                            }}));
                        """)
                        time.sleep(random.uniform(0.01, 0.05))
                    
                    # 到达目标后暂停一下
                    time.sleep(random.uniform(0.3, 1.2))
        except Exception as e:
            logger.warning(f"模拟鼠标移动失败: {e}")
    
    except Exception as e:
        logger.error(f"模拟人类行为失败: {e}")
        print(f"{Colors.RED}模拟人类行为失败: {e}{Colors.RESET}")
    
    finally:
        # 最后再随机等待一段时间
        time.sleep(random.uniform(1, 3))

def extract_product_info(driver, url):
    """提取商品信息
    
    Args:
        driver: WebDriver实例
        url: 商品URL
        
    Returns:
        dict: 商品信息
    """
    logger.info("开始提取商品信息")
    
    # 创建进度跟踪系统
    import sys
    from tqdm import tqdm
    
    # 定义提取步骤
    extraction_steps = [
        "基本信息", "价格信息", "店铺信息", "商品规格", 
        "商品描述", "评价信息", "促销信息", "图片信息"
    ]
    
    # 初始化进度条
    print(f"\n{Colors.CYAN}{Colors.BOLD}【数据提取进度】{Colors.RESET}")
    progress_bar = tqdm(total=len(extraction_steps), desc="提取商品数据", 
                        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Colors.GREEN, Colors.RESET))
    
    # 初始化结果字典
    product_info = {
        'product_id': '',
        'title': '',
        'price': '',
        'original_price': '',
        'discount': '',
        'shop_name': '',
        'shop_score': '',
        'brand': '',
        'model': '',
        'category': '',
        'description': '',
        'specs': {},
        'images': [],
        'comments_count': '',
        'good_rate': '',
        'promotion': '',
        'stock_status': '',
        'url': url,
        'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # ====== 步骤1: 提取基本信息 ======
        try:
            # 提取商品ID
            product_id = url.split('/')[-1].split('.')[0]
            product_info['product_id'] = product_id
            logger.info(f"提取到商品ID: {product_id}")
            
            # 提取商品标题
            try:
                title_elem = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".sku-name"))
                )
                product_info['title'] = title_elem.text.strip()
                logger.info(f"提取到商品标题: {product_info['title'][:30]}...")
            except (NoSuchElementException, TimeoutException):
                logger.warning("未找到商品标题元素")
                # 尝试其他可能的标题选择器
                try:
                    title_elem = driver.find_element(By.CSS_SELECTOR, ".product-intro .name")
                    product_info['title'] = title_elem.text.strip()
                except:
                    logger.warning("备用标题选择器也失败")
                    product_info['title'] = "提取失败"
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_description(f"提取{extraction_steps[0]}完成")
        except Exception as e:
            logger.error(f"提取基本信息失败: {e}")
            progress_bar.set_description(f"{Colors.RED}提取{extraction_steps[0]}失败{Colors.RESET}")
            progress_bar.update(1)
        
        # ====== 步骤2: 提取价格信息 ======
        try:
            # 首先检查是否有价格元素
            try:
                # 尝试获取价格
                price_elem = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".price .p-price .price"))
                )
                price = price_elem.text.strip()
                if price:
                    product_info['price'] = price
                    logger.info(f"提取到商品价格: {price}")
                else:
                    # 如果价格为空，可能需要登录
                    product_info['price'] = "需要登录查看"
                    logger.warning("价格为空，可能需要登录")
            except (NoSuchElementException, TimeoutException):
                # 尝试其他价格选择器
                try:
                    js_code = """
                    return document.querySelector('.price-type-1 .price') ? 
                           document.querySelector('.price-type-1 .price').innerText : '';
                    """
                    price = driver.execute_script(js_code)
                    if price:
                        product_info['price'] = price
                        logger.info(f"通过JS提取到商品价格: {price}")
                    else:
                        product_info['price'] = "提取失败"
                        logger.warning("通过JS提取价格失败")
                except Exception as e:
                    logger.error(f"尝试通过JS提取价格失败: {e}")
                    product_info['price'] = "提取失败"
            
            # 尝试提取原价和折扣信息
            try:
                original_price_elem = driver.find_element(By.CSS_SELECTOR, ".price .p-price-origin .price")
                product_info['original_price'] = original_price_elem.text.strip()
                
                # 计算折扣
                if product_info['price'] != "提取失败" and product_info['price'] != "需要登录查看" and product_info['original_price']:
                    try:
                        current_price = float(product_info['price'].replace('¥', '').replace(',', ''))
                        original_price = float(product_info['original_price'].replace('¥', '').replace(',', ''))
                        if original_price > 0:
                            discount = round((current_price / original_price) * 10, 1)
                            product_info['discount'] = f"{discount}折"
                    except:
                        pass
            except:
                pass
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_description(f"提取{extraction_steps[1]}完成")
        except Exception as e:
            logger.error(f"提取价格信息失败: {e}")
            progress_bar.set_description(f"{Colors.RED}提取{extraction_steps[1]}失败{Colors.RESET}")
            progress_bar.update(1)
        
        # ====== 步骤3: 提取店铺信息 ======
        try:
            # 提取店铺名称
            try:
                shop_name_elem = driver.find_element(By.CSS_SELECTOR, ".J-hove-shop-name .shop-name")
                product_info['shop_name'] = shop_name_elem.text.strip()
                logger.info(f"提取到店铺名称: {product_info['shop_name']}")
            except NoSuchElementException:
                try:
                    # 尝试其他可能的店铺名称选择器
                    shop_name_elem = driver.find_element(By.CSS_SELECTOR, "#shop-name")
                    product_info['shop_name'] = shop_name_elem.text.strip()
                except:
                    product_info['shop_name'] = "自营"  # 假设是京东自营
            
            # 提取店铺评分
            try:
                shop_score_elem = driver.find_element(By.CSS_SELECTOR, ".score-part .score-detail")
                product_info['shop_score'] = shop_score_elem.text.strip()
            except:
                pass
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_description(f"提取{extraction_steps[2]}完成")
        except Exception as e:
            logger.error(f"提取店铺信息失败: {e}")
            progress_bar.set_description(f"{Colors.RED}提取{extraction_steps[2]}失败{Colors.RESET}")
            progress_bar.update(1)
        
        # ====== 步骤4: 提取商品规格 ======
        try:
            specs = {}
            
            # 尝试提取规格参数
            try:
                # 点击规格参数标签
                try:
                    spec_tab = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "#detail .tab-main li:nth-child(2)"))
                    )
                    spec_tab.click()
                    time.sleep(1)
                except:
                    # 尝试其他可能的规格参数标签
                    try:
                        spec_tab = driver.find_element(By.XPATH, "//div[contains(@class, 'tab-main')]//li[contains(text(), '规格')]")
                        spec_tab.click()
                        time.sleep(1)
                    except:
                        logger.warning("未找到规格参数标签")
                
                # 提取规格参数表格
                spec_items = driver.find_elements(By.CSS_SELECTOR, "#detail .Ptable .Ptable-item")
                
                for item in spec_items:
                    try:
                        # 获取规格组名称
                        group_name = item.find_element(By.CSS_SELECTOR, "h3").text.strip()
                        specs[group_name] = {}
                        
                        # 获取该组下的所有规格项
                        spec_rows = item.find_elements(By.CSS_SELECTOR, ".Ptable-sub dl")
                        for row in spec_rows:
                            try:
                                key = row.find_element(By.CSS_SELECTOR, "dt").text.strip()
                                value = row.find_element(By.CSS_SELECTOR, "dd").text.strip()
                                specs[group_name][key] = value
                            except:
                                continue
                    except:
                        continue
                
                if specs:
                    product_info['specs'] = specs
                    logger.info(f"提取到{len(specs)}组规格参数")
                else:
                    logger.warning("未找到规格参数")
            except Exception as e:
                logger.error(f"提取规格参数失败: {e}")
            
            # 提取品牌和型号信息
            try:
                # 从规格参数中提取品牌
                if "主体" in specs and "品牌" in specs["主体"]:
                    product_info['brand'] = specs["主体"]["品牌"]
                elif "基本信息" in specs and "品牌" in specs["基本信息"]:
                    product_info['brand'] = specs["基本信息"]["品牌"]
                
                # 从规格参数中提取型号
                if "主体" in specs and "型号" in specs["主体"]:
                    product_info['model'] = specs["主体"]["型号"]
                elif "基本信息" in specs and "型号" in specs["基本信息"]:
                    product_info['model'] = specs["基本信息"]["型号"]
            except Exception as e:
                logger.error(f"提取品牌和型号失败: {e}")
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_description(f"提取{extraction_steps[3]}完成")
        except Exception as e:
            logger.error(f"提取商品规格失败: {e}")
            progress_bar.set_description(f"{Colors.RED}提取{extraction_steps[3]}失败{Colors.RESET}")
            progress_bar.update(1)
        
        # ====== 步骤5: 提取商品描述 ======
        try:
            # 点击商品介绍标签
            try:
                desc_tab = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "#detail .tab-main li:nth-child(1)"))
                )
                desc_tab.click()
                time.sleep(1)
            except:
                logger.warning("未找到商品介绍标签")
            
            # 提取商品描述
            try:
                desc_elem = driver.find_element(By.CSS_SELECTOR, "#detail .p-parameter")
                product_info['description'] = desc_elem.text.strip()
                logger.info("提取到商品描述")
            except:
                logger.warning("未找到商品描述元素")
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_description(f"提取{extraction_steps[4]}完成")
        except Exception as e:
            logger.error(f"提取商品描述失败: {e}")
            progress_bar.set_description(f"{Colors.RED}提取{extraction_steps[4]}失败{Colors.RESET}")
            progress_bar.update(1)
        
        # ====== 步骤6: 提取评价信息 ======
        try:
            # 点击商品评价标签
            try:
                comment_tab = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "#detail .tab-main li:nth-child(4)"))
                )
                comment_tab.click()
                time.sleep(2)
            except:
                # 尝试其他可能的评价标签
                try:
                    comment_tab = driver.find_element(By.XPATH, "//div[contains(@class, 'tab-main')]//li[contains(text(), '评价')]")
                    comment_tab.click()
                    time.sleep(2)
                except:
                    logger.warning("未找到商品评价标签")
            
            # 提取评价数量
            try:
                comments_count_elem = driver.find_element(By.CSS_SELECTOR, "#detail .tab-con .count")
                product_info['comments_count'] = comments_count_elem.text.strip()
                logger.info(f"提取到评价数量: {product_info['comments_count']}")
            except:
                logger.warning("未找到评价数量元素")
            
            # 提取好评率
            try:
                good_rate_elem = driver.find_element(By.CSS_SELECTOR, "#detail .tab-con .percent")
                product_info['good_rate'] = good_rate_elem.text.strip()
                logger.info(f"提取到好评率: {product_info['good_rate']}")
            except:
                logger.warning("未找到好评率元素")
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_description(f"提取{extraction_steps[5]}完成")
        except Exception as e:
            logger.error(f"提取评价信息失败: {e}")
            progress_bar.set_description(f"{Colors.RED}提取{extraction_steps[5]}失败{Colors.RESET}")
            progress_bar.update(1)
        
        # ====== 步骤7: 提取促销信息 ======
        try:
            # 提取促销信息
            try:
                promo_elem = driver.find_element(By.CSS_SELECTOR, "#summary-promotion .hl_red")
                product_info['promotion'] = promo_elem.text.strip()
                logger.info(f"提取到促销信息: {product_info['promotion']}")
            except:
                logger.warning("未找到促销信息元素")
            
            # 提取库存状态
            try:
                stock_elem = driver.find_element(By.CSS_SELECTOR, "#store-prompt")
                product_info['stock_status'] = stock_elem.text.strip()
                logger.info(f"提取到库存状态: {product_info['stock_status']}")
            except:
                logger.warning("未找到库存状态元素")
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_description(f"提取{extraction_steps[6]}完成")
        except Exception as e:
            logger.error(f"提取促销信息失败: {e}")
            progress_bar.set_description(f"{Colors.RED}提取{extraction_steps[6]}失败{Colors.RESET}")
            progress_bar.update(1)
        
        # ====== 步骤8: 提取图片信息 ======
        try:
            # 提取商品图片
            try:
                img_elements = driver.find_elements(By.CSS_SELECTOR, "#spec-list img")
                for img in img_elements:
                    try:
                        # 获取原始图片URL（替换缩略图URL为大图URL）
                        img_url = img.get_attribute("src")
                        if img_url:
                            # 替换缩略图URL为大图URL
                            img_url = img_url.replace("/n5/", "/n1/").replace("/s54x54_", "/s450x450_")
                            product_info['images'].append(img_url)
                    except:
                        continue
                
                logger.info(f"提取到{len(product_info['images'])}张商品图片")
            except Exception as e:
                logger.error(f"提取商品图片失败: {e}")
            
            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_description(f"提取{extraction_steps[7]}完成")
        except Exception as e:
            logger.error(f"提取图片信息失败: {e}")
            progress_bar.set_description(f"{Colors.RED}提取{extraction_steps[7]}失败{Colors.RESET}")
            progress_bar.update(1)
        
    except Exception as e:
        logger.error(f"提取商品信息失败: {e}")
        print(f"\n{Colors.RED}提取商品信息时发生错误: {e}{Colors.RESET}")
    
    # 关闭进度条
    progress_bar.close()
    
    # 显示提取结果摘要
    print(f"\n{Colors.GREEN}{Colors.BOLD}【数据提取完成】{Colors.RESET}")
    print(f"• 商品ID: {Colors.CYAN}{product_info['product_id']}{Colors.RESET}")
    print(f"• 商品标题: {Colors.CYAN}{product_info['title'][:50]}...{Colors.RESET}" if len(product_info['title']) > 50 else f"• 商品标题: {Colors.CYAN}{product_info['title']}{Colors.RESET}")
    print(f"• 商品价格: {Colors.CYAN}{product_info['price']}{Colors.RESET}")
    print(f"• 店铺名称: {Colors.CYAN}{product_info['shop_name']}{Colors.RESET}")
    print(f"• 评价数量: {Colors.CYAN}{product_info['comments_count']}{Colors.RESET}")
    print(f"• 好评率: {Colors.CYAN}{product_info['good_rate']}{Colors.RESET}")
    print(f"• 提取规格: {Colors.CYAN}{len(product_info['specs'])}组{Colors.RESET}")
    print(f"• 提取图片: {Colors.CYAN}{len(product_info['images'])}张{Colors.RESET}")
    
    return product_info

def save_product_info(product_info, output_dir):
    """保存商品信息"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    product_id = product_info['id']
    
    # 保存JSON文件
    json_path = os.path.join(output_dir, f"product_{product_id}_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(product_info, f, ensure_ascii=False, indent=2)
    
    # 保存CSV文件
    csv_path = os.path.join(output_dir, f"product_{product_id}_{timestamp}.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        # 写入CSV头部
        headers = ['id', 'title', 'price', 'brand', 'shop', 'category', 'description', 'comments_count', 'good_rate', 'url', 'crawl_time']
        f.write(','.join(headers) + '\n')
        
        # 写入CSV数据
        row = [
            product_info['id'],
            product_info['title'].replace(',', '，'),
            product_info['price'],
            product_info['brand'].replace(',', '，'),
            product_info['shop'].replace(',', '，'),
            product_info['category'].replace(',', '，'),
            product_info['description'].replace(',', '，').replace('\n', ' '),
            product_info['comments']['count'],
            product_info['comments']['good_rate'],
            product_info['url'],
            product_info['timestamp']
        ]
        f.write(','.join([str(item) for item in row]) + '\n')
    
    logger.info(f"商品信息已保存到: {json_path}")
    logger.info(f"商品信息已保存到: {csv_path}")
    
    # 如果需要保存图片
    if SPIDER_CONFIG['OUTPUT']['SAVE_IMAGES'] and product_info['images']:
        import requests
        from urllib.parse import urlparse
        
        # 创建图片目录
        image_dir = os.path.join(output_dir, 'images', product_id)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        
        # 下载图片
        for i, img_url in enumerate(product_info['images']):
            try:
                # 获取图片文件名
                img_filename = os.path.basename(urlparse(img_url).path)
                if not img_filename:
                    img_filename = f"image_{i+1}.jpg"
                
                # 下载图片
                img_path = os.path.join(image_dir, img_filename)
                
                # 添加请求头，模拟浏览器
                headers = {
                    'User-Agent': UserAgent().random,
                    'Referer': product_info['url'],
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Connection': 'keep-alive'
                }
                
                response = requests.get(img_url, headers=headers, timeout=10)
                with open(img_path, 'wb') as img_file:
                    img_file.write(response.content)
                
                logger.info(f"图片已保存到: {img_path}")
                
                # 随机延迟，避免被检测
                time.sleep(random.uniform(0.5, 1.5))
                
            except Exception as e:
                logger.error(f"下载图片失败 {img_url}: {e}")
        
        logger.info(f"所有图片已保存到: {image_dir}")

def run_spider(url):
    """运行爬虫"""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}【京东爬虫】启动{Colors.RESET}")
    print(f"目标URL: {Colors.BLUE}{url}{Colors.RESET}")
    print("="*60 + "\n")
    
    logger.info(f"开始爬取: {url}")
    
    try:
        # ====== 步骤1: 初始化浏览器 ======
        driver = init_browser()
        
        # ====== 步骤2: 访问目标URL ======
        logger.info(f"正在访问: {url}")
        print(f"{Colors.BLUE}正在加载页面...{Colors.RESET}")
        try:
            driver.get(url)
            time.sleep(random.uniform(3, 5))  # 等待页面加载完成
        except Exception as e:
            logger.error(f"访问URL失败: {e}")
            raise Exception(f"访问目标URL失败: {e}")
        
        # 保存原始URL以便后续使用
        original_url = url
        logger.info(f"保存原始URL: {original_url}")
        try:
            driver.execute_script(f"localStorage.setItem('jd_original_url', '{original_url}');")
            if "original_product_url" not in globals():
                global original_product_url
                original_product_url = original_url
        except Exception as e:
            logger.warning(f"保存URL到localStorage失败: {e}")
        
        # ====== 步骤3: 创建验证处理器，处理可能的URL重定向和验证 ======
        verification_handler = JDVerificationHandler(driver)
        
        # 处理重定向
        try:
            current_url = driver.current_url
            if current_url != original_url:
                logger.info(f"检测到URL重定向: {original_url} -> {current_url}")
                print(f"{Colors.YELLOW}检测到URL重定向，正在处理...{Colors.RESET}")
                
                # 使用验证处理器处理URL重定向
                updated_url = verification_handler.handle_url_redirect(original_url)
                
                # 如果URL发生变化，更新全局变量
                if updated_url != original_url:
                    logger.info(f"URL已更新: {updated_url}")
                    print(f"{Colors.GREEN}已更新到正确的URL: {updated_url}{Colors.RESET}")
                    url = updated_url  # 更新当前使用的URL
        except Exception as e:
            logger.error(f"处理URL重定向时出错: {e}")
            print(f"{Colors.YELLOW}处理URL重定向时出错，将继续使用原始URL{Colors.RESET}")
        
        # 处理验证码/滑块等验证
        try:
            verification_handler.handle_verification()
        except Exception as e:
            logger.error(f"处理验证时出错: {e}")
            print(f"{Colors.YELLOW}处理验证时出错: {e}，尝试继续...{Colors.RESET}")
        
        # ====== 步骤4: 检查是否需要登录 ======
        login_success = False
        try:
            # 检查当前是否已登录
            is_logged_in, username = check_login_status(driver)
            if is_logged_in:
                logger.info(f"检测到已登录状态: {username}")
                print(f"\n{Colors.GREEN}✓ 已登录: {username}{Colors.RESET}")
                login_success = True
            else:
                # 检查页面是否需要登录
                need_login = verification_handler.check_login_required()
                
                if need_login:
                    logger.info("检测到页面需要登录")
                    print(f"\n{Colors.YELLOW}{Colors.BOLD}【需要登录】{Colors.RESET}")
                    print(f"检测到该页面需要登录才能获取完整数据")
                    print(f"正在为您处理登录流程...\n")
                    
                    # 自动启用登录功能，即使用户没有设置--login参数
                    if not SPIDER_CONFIG['LOGIN']['ENABLE']:
                        logger.info("检测到需要登录，自动启用登录功能")
                        print(f"{Colors.GREEN}已自动启用登录功能{Colors.RESET}")
                        SPIDER_CONFIG['LOGIN']['ENABLE'] = True
                    
                    # ====== 步骤5: 处理登录 ======
                    # 进行登录操作
                    login_success = handle_login(driver)
                    
                    # 登录后再次检查验证码，有时登录会触发新的验证
                    try:
                        verification_handler.handle_verification()
                    except Exception as e:
                        logger.error(f"登录后处理验证码失败: {e}")
                        print(f"{Colors.YELLOW}处理验证时出错，尝试继续...{Colors.RESET}")
                    
                    # ====== 步骤6: 确认登录状态 ======
                    # 重新检查登录状态
                    time.sleep(2)  # 等待一下确保状态已更新
                    is_logged_in, username = check_login_status(driver)
                    if is_logged_in:
                        print("\n" + "="*60)
                        print(f"{Colors.GREEN}{Colors.BOLD}【登录状态确认】{Colors.RESET}")
                        print(f"京东账号 {Colors.YELLOW}{username}{Colors.RESET} 已登录成功!")
                        print(f"现在开始获取完整的商品数据...")
                        print("="*60 + "\n")
                        login_success = True
                else:
                    logger.info("页面不需要登录，以访客模式继续")
                    print(f"{Colors.CYAN}页面不需要登录，以访客模式继续...{Colors.RESET}")
        except Exception as e:
            logger.error(f"检查登录状态时出错: {e}")
            print(f"{Colors.YELLOW}检查登录状态时出错，尝试以访客模式继续...{Colors.RESET}")
        
        # ====== 步骤7: 确保回到商品页面 ======
        try:
            current_url = driver.current_url
            
            # 如果当前不在商品页面，尝试回到商品页
            if "item.jd.com" not in current_url:
                logger.info(f"当前不在商品页面 ({current_url})，尝试返回商品页")
                print(f"{Colors.YELLOW}返回商品页...{Colors.RESET}")
                
                # 尝试加载原始商品页
                try:
                    driver.get(url)
                    time.sleep(3)
                except Exception as e:
                    logger.error(f"返回原始商品页失败: {e}")
                
                # 如果还是不在商品页，尝试从localStorage恢复
                if "item.jd.com" not in driver.current_url:
                    try:
                        saved_url = driver.execute_script("return localStorage.getItem('jd_original_url');")
                        if saved_url and "item.jd.com" in saved_url:
                            logger.info(f"从localStorage恢复URL: {saved_url}")
                            print(f"{Colors.YELLOW}从缓存恢复商品页...{Colors.RESET}")
                            driver.get(saved_url)
                            time.sleep(3)
                    except Exception as e:
                        logger.error(f"从localStorage恢复URL失败: {e}")
                
                # 如果上述方法都失败，尝试构建新的商品URL
                if "item.jd.com" not in driver.current_url:
                    try:
                        # 从原始URL提取商品ID
                        product_id = url.split("/")[-1].split(".")[0]
                        new_url = f"https://item.jd.com/{product_id}.html"
                        logger.info(f"尝试构建新URL: {new_url}")
                        print(f"{Colors.YELLOW}尝试访问: {new_url}{Colors.RESET}")
                        driver.get(new_url)
                        time.sleep(3)
                    except Exception as e:
                        logger.error(f"构建和访问新URL失败: {e}")
                
                # 访问商品页后可能再次出现验证码
                try:
                    verification_handler.handle_verification()
                except Exception as e:
                    logger.error(f"返回商品页后处理验证失败: {e}")
            
            # 检查最终是否成功返回商品页
            if "item.jd.com" in driver.current_url:
                print(f"{Colors.GREEN}✓ 已成功返回商品页面{Colors.RESET}")
                # 查找几个关键元素，确认页面加载正常
                try:
                    title_elem = driver.find_element(By.CSS_SELECTOR, ".sku-name")
                    if title_elem and title_elem.text:
                        print(f"{Colors.GREEN}✓ 商品标题已加载：{title_elem.text[:30]}...{Colors.RESET}")
                except:
                    print(f"{Colors.YELLOW}⚠️ 商品标题加载异常{Colors.RESET}")
            else:
                print(f"{Colors.RED}⚠️ 无法返回商品页面，可能影响数据抓取{Colors.RESET}")
        except Exception as e:
            logger.error(f"尝试返回商品页面时出错: {e}")
            print(f"{Colors.YELLOW}尝试返回商品页面时出错，继续尝试抓取数据...{Colors.RESET}")
        
        # ====== 步骤8: 抓取商品数据 ======
        try:
            print(f"\n{Colors.BLUE}正在提取商品信息...{Colors.RESET}")
            product_info = extract_product_info(driver, url)
            
            # 显示登录状态提醒
            if not login_success and (product_info['price'] == "需要登录查看" or product_info['price'] == "提取失败"):
                print("\n" + "="*60)
                print(f"{Colors.YELLOW}{Colors.BOLD}【数据缺失提醒】{Colors.RESET}")
                print("部分商品数据无法提取，原因：未登录")
                print("如需完整数据，请使用以下命令重新运行并启用登录功能：")
                print(f"{Colors.GREEN}python run_jd_spider.py --login{Colors.RESET}")
                print("="*60 + "\n")
            
            # 保存商品信息
            save_product_info(product_info, SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'])
            
            # 显示抓取结果
            print("\n" + "="*60)
            print(f"{Colors.GREEN}{Colors.BOLD}【爬取完成】{Colors.RESET}")
            print(f"商品: {Colors.YELLOW}{product_info['title'][:30] + ('...' if len(product_info['title']) > 30 else '')}{Colors.RESET}")
            if login_success:
                print(f"价格: {Colors.GREEN}{product_info['price']}{Colors.RESET} (登录账号查看)")
            else:
                print(f"价格: {product_info['price']} (访客查看)")
            print(f"数据已保存到: {Colors.BLUE}{SPIDER_CONFIG['OUTPUT']['RESULTS_DIR']}{Colors.RESET}")
            print("="*60 + "\n")
            
            logger.info("爬取完成")
            return True
            
        except Exception as e:
            logger.error(f"提取商品信息出错: {e}")
            print("\n" + "="*60)
            print(f"{Colors.RED}{Colors.BOLD}【错误】提取商品数据失败{Colors.RESET}")
            print(f"错误信息: {e}")
            print("="*60 + "\n")
            
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"爬虫运行失败: {e}")
        print("\n" + "="*60)
        print(f"{Colors.RED}{Colors.BOLD}【错误】爬虫运行失败{Colors.RESET}")
        print(f"错误信息: {e}")
        print("="*60 + "\n")
        
        import traceback
        logger.error(traceback.format_exc())
        return False
        
    finally:
        # 安全关闭浏览器
        try:
            # 随机等待，避免立即关闭浏览器
            time.sleep(random.uniform(1, 3))
            
            if 'driver' in locals() and driver:
                # 检查浏览器会话是否仍然有效
                try:
                    current_url = driver.current_url  # 尝试访问，如果会话无效会抛出异常
                    driver.quit()
                    logger.info("已关闭浏览器")
                except Exception:
                    logger.info("浏览器会话已失效，无需关闭")
        except Exception as e:
            logger.error(f"关闭浏览器时出错: {e}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 更新配置
    update_config(args)
    
    # 创建输出目录
    if not os.path.exists(SPIDER_CONFIG['OUTPUT']['RESULTS_DIR']):
        os.makedirs(SPIDER_CONFIG['OUTPUT']['RESULTS_DIR'])
    
    # 显示爬取信息
    logger.info("=" * 50)
    logger.info("京东商品爬虫启动")
    logger.info(f"目标URL: {SPIDER_CONFIG['TARGET_URL']}")
    logger.info(f"输出目录: {SPIDER_CONFIG['OUTPUT']['RESULTS_DIR']}")
    logger.info(f"无头模式: {'是' if SPIDER_CONFIG['BROWSER']['HEADLESS'] else '否'}")
    logger.info(f"启用登录: {'是' if SPIDER_CONFIG['LOGIN']['ENABLE'] else '否'}")
    logger.info("=" * 50)
    
    # 运行爬虫
    try:
        run_spider(SPIDER_CONFIG['TARGET_URL'])
        logger.info("爬虫运行完成")
        return 0
    except Exception as e:
        logger.error(f"爬虫运行失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 