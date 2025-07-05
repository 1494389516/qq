#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
京东登录模块
支持两种登录方式：
1. 扫码登录（推荐）
2. 账号密码登录（需要处理验证码）
"""

import os
import time
import json
import base64
import logging
import pickle
import threading
import tempfile
import subprocess
from PIL import Image
from io import BytesIO
import webbrowser
import platform
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from config import SPIDER_CONFIG
import colorama
from colorama import Fore, Style, Back
from selenium import webdriver

# 初始化colorama
colorama.init(autoreset=True)

# 全局变量，存储原始商品URL
original_product_url = None

# 配置日志记录
logger = logging.getLogger(__name__)

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
    CYAN = "\033[96m" if HAS_COLORAMA else ""  # 添加青色

class LoginStatusMonitor:
    """登录状态监控类"""
    
    def __init__(self, driver):
        self.driver = driver
        self.is_logged_in = False
        self.username = None
        self.monitor_active = False
        self.monitor_thread = None
        self.cookies_saved = False
        self.on_login_callback = None
        self.login_detected_time = None  # 记录检测到登录的时间
        self.login_detection_methods = [
            self._check_nickname_element,
            self._check_user_info,
            self._check_please_login_text,
            self._check_js_login_state,
            self._check_logout_button
        ]
    
    def start_monitoring(self, callback=None):
        """开始监控登录状态
        
        Args:
            callback: 登录成功后的回调函数, 传入参数(username)
        """
        self.monitor_active = True
        self.on_login_callback = callback
        self.monitor_thread = threading.Thread(target=self._monitor_login_status)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("登录状态监控器已启动")
        print(f"\n{Colors.CYAN}登录状态监控已启动，正在等待扫码登录...{Colors.RESET}")
    
    def stop_monitoring(self):
        """停止监控登录状态"""
        self.monitor_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("登录状态监控器已停止")
    
    def _monitor_login_status(self):
        """监控登录状态线程"""
        check_interval = 0.5  # 每0.5秒检查一次
        check_count = 0  # 检查次数
        status_update_interval = 20  # 每20次检查输出一次状态日志（约10秒）
        
        # 初始状态为未登录
        self.is_logged_in = False
        self.username = None
        
        while self.monitor_active:
            try:
                check_count += 1
                
                # 每隔一定次数显示状态日志
                if check_count % status_update_interval == 0:
                    logger.info(f"登录状态监控中... 当前状态: {'已登录' if self.is_logged_in else '未登录'}")
                    if not self.is_logged_in and check_count >= 40:  # 约20秒后开始提示
                        print(f"\r{Colors.YELLOW}正在等待登录，请扫描并确认二维码... {check_count // 2}秒{Colors.RESET}", end="")
                
                # 使用多种方法检查登录状态
                logged_in = False
                detected_username = None
                login_method = None
                
                for method in self.login_detection_methods:
                    try:
                        method_name = method.__name__
                        result, username = method()
                        if result:
                            logged_in = True
                            detected_username = username
                            login_method = method_name
                            break
                    except Exception as e:
                        logger.debug(f"登录检测方法 {method.__name__} 失败: {e}")
                
                # 状态变化：从未登录到已登录
                if logged_in and not self.is_logged_in:
                    self.is_logged_in = True
                    self.username = detected_username or "京东用户"
                    self.login_detected_time = time.time()
                    logger.info(f"状态监控器检测到登录成功: {self.username} (方法: {login_method})")
                    
                    # 清除等待提示
                    if check_count >= 40:
                        print("\r" + " " * 60)
                        
                    self._show_login_success()
                    if self.on_login_callback:
                        self.on_login_callback(self.username)
                
                # 如果已登录超过5秒，但还没保存cookies，显示提示
                if self.is_logged_in and self.login_detected_time and not self.cookies_saved:
                    elapsed = time.time() - self.login_detected_time
                    if elapsed > 5 and int(elapsed) % 3 == 0:  # 每3秒提示一次
                        print(f"\r{Colors.YELLOW}正在等待保存登录状态，请不要关闭窗口...{Colors.RESET}", end="")
                
            except Exception as e:
                logger.error(f"监控登录状态出错: {e}")
            
            time.sleep(check_interval)
    
    def _check_nickname_element(self):
        """检查昵称元素"""
        nickname_element = self.driver.find_element(By.CLASS_NAME, "nickname")
        if nickname_element and nickname_element.text and "请登录" not in nickname_element.text:
            return True, nickname_element.text
        return False, None
    
    def _check_user_info(self):
        """检查用户信息元素"""
        login_status = self.driver.find_element(By.CSS_SELECTOR, ".user-info .user-level")
        if login_status:
            try:
                username_elem = self.driver.find_element(By.CSS_SELECTOR, ".user-info .nickname")
                return True, username_elem.text
            except:
                return True, "京东用户"
        return False, None
    
    def _check_please_login_text(self):
        """检查'请登录'文本是否存在"""
        try:
            self.driver.find_element(By.LINK_TEXT, "请登录")
            # 仍有"请登录"文本，未登录
            return False, None
        except:
            # 检查页面上是否有其他登录状态指标
            page_source = self.driver.page_source.lower()
            if "欢迎登录" not in page_source and ("已登录" in page_source or "hi," in page_source):
                return True, "京东用户"
            return False, None
    
    def _check_js_login_state(self):
        """使用JS检测登录状态"""
        is_logged = self.driver.execute_script("""
            try {
                // 检查是否有登录用户信息
                if (document.querySelector('.nickname') && 
                    document.querySelector('.nickname').innerText && 
                    document.querySelector('.nickname').innerText !== '请登录') {
                    return true;
                }
                
                // 检查是否存在注销按钮
                if (document.querySelector('a[href*="logout"]')) {
                    return true;
                }
                
                return false;
            } catch (e) {
                return false;
            }
        """)
        
        if is_logged:
            # 尝试获取用户名
            username_elem = self.driver.execute_script("""
                return document.querySelector('.nickname') ? 
                    document.querySelector('.nickname').innerText : '京东用户';
            """)
            username = username_elem if username_elem != '请登录' else '京东用户'
            return True, username
        return False, None
    
    def _check_logout_button(self):
        """检查是否存在注销按钮"""
        try:
            logout_button = self.driver.find_element(By.CSS_SELECTOR, "a[href*='logout']")
            if logout_button:
                return True, "京东用户"
        except:
            pass
        return False, None
    
    def _show_login_success(self):
        """显示登录成功信息"""
        print("\n" + "="*70)
        print(f"{Colors.GREEN}{Colors.BOLD}【登录状态检测】登录成功!{Colors.RESET}")
        print(f"京东账号 {Colors.YELLOW}{self.username}{Colors.RESET} 已登录!")
        print(f"{Colors.CYAN}正在保存登录状态，请稍候...{Colors.RESET}")
        print(f"{Colors.YELLOW}提示: 请不要关闭浏览器窗口，等待系统自动处理{Colors.RESET}")
        print("="*70 + "\n")

class JDLogin:
    """京东登录处理类"""
    
    def __init__(self, driver, timeout=30):
        """初始化登录处理器
        
        Args:
            driver: WebDriver实例
            timeout: 超时时间（秒）
        """
        self.driver = driver
        self.timeout = timeout
        self.cookies_dir = SPIDER_CONFIG['LOGIN']['COOKIES_DIR'] if 'COOKIES_DIR' in SPIDER_CONFIG['LOGIN'] else "cookies"
        self.qrcode_timeout = SPIDER_CONFIG['LOGIN']['QRCODE_TIMEOUT'] if 'QRCODE_TIMEOUT' in SPIDER_CONFIG['LOGIN'] else 60
        self.qrcode_auto_refresh = SPIDER_CONFIG['LOGIN']['QRCODE_REFRESH'] if 'QRCODE_REFRESH' in SPIDER_CONFIG['LOGIN'] else True
        self.qrcode_refresh_count = 0  # 记录二维码刷新次数
        self.max_qrcode_refresh = SPIDER_CONFIG['LOGIN'].get('MAX_QRCODE_REFRESH', 3)  # 最大刷新次数
        
        # 创建cookies目录
        if not os.path.exists(self.cookies_dir):
            os.makedirs(self.cookies_dir)
            
        # 初始化状态监控器
        self.status_monitor = LoginStatusMonitor(driver)
        
        # 注入JavaScript状态监听器
        self.inject_login_detector()
    
    def _on_login_success_callback(self, username, event=None):
        """登录成功的回调处理
        
        Args:
            username: 登录用户名
            event: 可选的事件对象，用于通知等待线程
            
        Returns:
            bool: 是否处理成功
        """
        logger.info(f"登录成功回调触发: {username}")
        
        try:
            # 保存cookies到文件
            self._save_cookies(username)
            
            # 设置cookies已保存标志，通知监控器不再提示
            self.status_monitor.cookies_saved = True
            
            # 清除等待提示（如果有）
            print("\r" + " " * 80)
            
            # 显示登录成功信息
            print("\n" + "="*70)
            print(f"{Colors.GREEN}{Colors.BOLD}【登录成功】{Colors.RESET}")
            print(f"京东账号 {Colors.YELLOW}{username}{Colors.RESET} 已成功登录!")
            print(f"{Colors.GREEN}✓ cookies已保存，下次运行将自动使用cookies登录{Colors.RESET}")
            print(f"{Colors.GREEN}✓ 登录状态将会保持，除非清除cookies或手动退出登录{Colors.RESET}")
            print("="*70 + "\n")
            
            # 如果提供了事件对象，设置事件以通知等待线程
            if event:
                event.set()
            
            logger.info("登录成功回调完成")
            return True
            
        except Exception as e:
            logger.error(f"登录成功回调处理失败: {e}")
            
            # 如果回调处理失败，也设置事件，以避免无限等待
            if event:
                event.set()
                
            return False
    
    def _login_status_announcement(self, is_logged_in, username=None):
        """显示当前登录状态的公告
        
        Args:
            is_logged_in: 是否已登录
            username: 用户名（如果已登录）
        """
        if is_logged_in and username:
            print("\n" + "="*60)
            print(f"{Colors.GREEN}{Colors.BOLD}【京东账号状态】已登录{Colors.RESET}")
            print(f"当前登录账号: {Colors.YELLOW}{username}{Colors.RESET}")
            print(f"登录状态正常，可以获取完整数据")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print(f"{Colors.YELLOW}{Colors.BOLD}【京东账号状态】未登录{Colors.RESET}")
            print(f"需要登录才能获取完整数据")
            print("="*60 + "\n")
            
    def _save_cookies(self, username):
        """保存cookies
        
        Args:
            username: 用户名，用于生成cookies文件名
        """
        cookies_file = os.path.join(self.cookies_dir, f"{username}.pkl")
        cookies = self.driver.get_cookies()
        
        if not cookies:
            logger.warning("没有cookies可保存")
            return
        
        # 打印cookie信息(调试用)
        logger.info(f"获取到{len(cookies)}个cookies")
        
        with open(cookies_file, "wb") as f:
            pickle.dump(cookies, f)
            
        logger.info(f"已保存cookies到: {cookies_file}")
        print(f"\n{Colors.GREEN}✓ Cookies已保存至 {cookies_file}{Colors.RESET}")
        print(f"{Colors.GREEN}✓ 下次运行时将自动使用此登录状态{Colors.RESET}")
        
        # 打印cookie的域名(调试用)
        domains = set(cookie['domain'] for cookie in cookies if 'domain' in cookie)
        logger.info(f"Cookie域: {domains}")
        
        # 保存cookies.json文件以便查看
        try:
            cookies_json = os.path.join(self.cookies_dir, f"{username}.json")
            with open(cookies_json, "w", encoding='utf-8') as f:
                # 将日期转换为字符串
                safe_cookies = []
                for cookie in cookies:
                    safe_cookie = cookie.copy()
                    if 'expiry' in safe_cookie:
                        safe_cookie['expiry'] = str(safe_cookie['expiry'])
                    safe_cookies.append(safe_cookie)
                json.dump(safe_cookies, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存cookies JSON到: {cookies_json}")
        except Exception as e:
            logger.error(f"保存cookies JSON失败: {e}")
    
    def _validate_cookies(self, url=None):
        """验证cookies是否有效
        
        Args:
            url: 要验证的URL，如果提供，会导航到此URL检查登录状态
            
        Returns:
            tuple: (是否有效, 用户名)
        """
        try:
            # 如果提供了URL，导航到该URL
            if url and url.startswith("http"):
                logger.info(f"导航到URL验证cookies: {url}")
                self.driver.get(url)
                time.sleep(3)
            
            # 检查登录状态
            nickname_element = None
            try:
                nickname_element = self.driver.find_element(By.CLASS_NAME, "nickname")
            except:
                # 尝试其他可能的用户名元素
                try:
                    nickname_element = self.driver.find_element(By.ID, "ttbar-login")
                    if nickname_element and "请登录" in nickname_element.text:
                        return False, None
                except:
                    pass
            
            # 如果找到用户名元素并且有内容
            if nickname_element and nickname_element.text and "请登录" not in nickname_element.text:
                logger.info(f"通过nickname元素验证cookies有效: {nickname_element.text}")
                return True, nickname_element.text
            
            # 尝试使用JavaScript检测登录状态
            logged_in, username = self.inject_login_detector()
            if logged_in and username:
                logger.info(f"通过JS检测验证cookies有效: {username}")
                return True, username
            
            # 检查页面内容是否表明已登录
            page_source = self.driver.page_source.lower()
            has_login_text = "请登录" in page_source or "login" in page_source
            has_user_info = "user-info" in page_source and "nickname" in page_source
            
            if not has_login_text and has_user_info:
                logger.info("通过页面内容验证cookies可能有效")
                return True, "京东用户"
            
            logger.info("cookies验证结果: 无效")
            return False, None
        except Exception as e:
            logger.error(f"验证cookies失效: {e}")
            return False, None
    
    def _load_cookies(self, username):
        """加载cookies
        
        Args:
            username: 用户名，用于查找cookies文件
            
        Returns:
            bool: 是否成功加载cookies
        """
        cookies_file = os.path.join(self.cookies_dir, f"{username}.pkl")
        if not os.path.exists(cookies_file):
            logger.warning(f"未找到cookies文件: {cookies_file}")
            return False
        
        try:
            print(f"\n{Colors.BLUE}尝试使用保存的Cookies登录...{Colors.RESET}")
            with open(cookies_file, "rb") as f:
                cookies = pickle.load(f)
            
            if not cookies:
                logger.warning("cookies文件存在但为空")
                os.remove(cookies_file)
                return False
            
            logger.info(f"已加载{len(cookies)}个cookies")
            
            # 保存当前URL，以便验证后返回
            original_url = self.driver.current_url
            
            # 先访问京东首页
            self.driver.get("https://www.jd.com/")
            time.sleep(2)
            
            # 保存原始URL到localStorage
            if original_url and "item.jd.com" in original_url:
                self.driver.execute_script(f"localStorage.setItem('jd_original_url', '{original_url}');")
            
            # 添加cookies
            for cookie in cookies:
                try:
                    if 'expiry' in cookie:
                        del cookie['expiry']
                    self.driver.add_cookie(cookie)
                except Exception as e:
                    logger.error(f"添加cookie失败: {e}")
            
            # 刷新页面
            self.driver.refresh()
            time.sleep(3)
            
            # 检查是否登录成功
            is_valid, detected_username = self._validate_cookies()
            if is_valid:
                logger.info(f"通过cookies登录成功！账号: {detected_username}")
                print(f"\n{Colors.GREEN}✓ Cookie有效，已自动登录为: {detected_username}{Colors.RESET}")
                
                # 如果是在商品页，返回原始URL
                if original_url and "item.jd.com" in original_url:
                    logger.info(f"导航回原始URL: {original_url}")
                    self.driver.get(original_url)
                    time.sleep(3)
                
                return True
            else:
                logger.warning("通过cookies登录失败，cookies可能已过期，需要重新登录")
                print(f"\n{Colors.YELLOW}保存的登录信息已过期，需要重新登录{Colors.RESET}")
                
                # 删除无效的cookie文件
                try:
                    os.remove(cookies_file)
                    logger.info(f"已删除无效的cookie文件: {cookies_file}")
                except:
                    pass
                
                return False
        except Exception as e:
            logger.error(f"加载cookies失败: {e}")
            return False
    
    def is_logged_in(self):
        """检查当前是否已登录
        
        Returns:
            bool: 是否已登录
        """
        max_retries = 3  # 最大重试次数
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 检查是否有用户名元素
                try:
                    nickname_element = self.driver.find_element(By.CLASS_NAME, "nickname")
                    if nickname_element and nickname_element.text and "请登录" not in nickname_element.text:
                        logger.info(f"检测到已登录，用户名: {nickname_element.text}")
                        return True
                except (NoSuchElementException, TimeoutException):
                    pass
                    
                # 尝试其他登录检测方法
                try:
                    # 检查用户信息区域
                    user_info = self.driver.find_element(By.CSS_SELECTOR, ".user-info")
                    if user_info:
                        logger.info("检测到用户信息区域，可能已登录")
                        return True
                except (NoSuchElementException, TimeoutException):
                    pass
                    
                # 尝试通过JavaScript检测
                try:
                    result = self.driver.execute_script("""
                        if (document.querySelector('.nickname') && 
                            document.querySelector('.nickname').innerText && 
                            document.querySelector('.nickname').innerText !== '请登录') {
                            return true;
                        }
                        return false;
                    """)
                    if result:
                        logger.info("通过JavaScript检测到已登录")
                        return True
                except Exception:
                    pass
                
                logger.info("未检测到登录状态")
                return False
            
            except Exception as e:
                retry_count += 1
                logger.warning(f"检测登录状态异常 (尝试 {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(1)  # 等待一秒后重试
                else:
                    logger.error(f"检测登录状态失败，达到最大重试次数: {e}")
                    # 最后一次尝试，直接检查URL或cookies
                    try:
                        cookies = self.driver.get_cookies()
                        domains = [cookie['domain'] for cookie in cookies if 'domain' in cookie]
                        logger.info(f"通过检查cookie域名判断登录状态: {domains}")
                        
                        # 检查是否含有登录相关的cookie域
                        login_domains = ['.jd.com', 'passport.jd.com']
                        if any(domain in domains for domain in login_domains):
                            return True  # 可能已登录
                    except:
                        pass
                    return False  # 默认假设未登录
    
    def inject_login_detector(self):
        """注入JavaScript登录状态检测器"""
        js_code = """
        // 创建一个状态监听器
        if (!window._JD_LOGIN_MONITOR) {
            window._JD_LOGIN_MONITOR = true;
            
            // 监听DOM变化
            let observer = new MutationObserver(function(mutations) {
                // 检查是否显示用户名
                let nickname = document.querySelector('.nickname');
                if (nickname && nickname.innerText) {
                    // 设置自定义标记表示已登录
                    document.body.setAttribute('data-jd-logged-in', 'true');
                    document.body.setAttribute('data-jd-username', nickname.innerText);
                }
            });
            
            // 开始监听DOM变化
            observer.observe(document.body, { childList: true, subtree: true });
            
            // 立即执行一次检查
            let nickname = document.querySelector('.nickname');
            if (nickname && nickname.innerText) {
                document.body.setAttribute('data-jd-logged-in', 'true');
                document.body.setAttribute('data-jd-username', nickname.innerText);
            }
        }
        
        // 返回当前登录状态
        let nickname = document.querySelector('.nickname');
        return nickname && nickname.innerText ? nickname.innerText : "";
        """
        
        try:
            result = self.driver.execute_script(js_code)
            if result:
                return True, result
            return False, None
        except Exception as e:
            logger.error(f"注入登录检测器失败: {e}")
            return False, None
    
    def open_image(self, image_path):
        """打开图片文件
        
        Args:
            image_path: 图片路径
        """
        try:
            logger.info(f"尝试打开图片: {image_path}")
            if not os.path.exists(image_path):
                logger.error(f"图片文件不存在: {image_path}")
                return
            
            # 根据操作系统选择不同的打开方式
            if platform.system() == "Windows":
                os.startfile(image_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", image_path])
            else:  # Linux
                subprocess.call(["xdg-open", image_path])
                
            logger.info(f"图片已打开: {image_path}")
        except Exception as e:
            logger.error(f"打开图片失败: {e}")
            print(f"{Colors.YELLOW}无法自动打开图片: {e}{Colors.RESET}")
            print(f"{Colors.YELLOW}请手动打开文件: {image_path}{Colors.RESET}")
            
    def check_qrcode_status(self):
        """检查二维码状态
        
        Returns:
            str: 状态码
                'valid': 二维码有效
                'scanned': 二维码已扫描
                'expired': 二维码已过期
                'confirmed': 二维码已确认
        """
        # 首先检查是否已登录（通过JavaScript注入检测）
        logged_in, username = self.inject_login_detector()
        if logged_in:
            return "confirmed"
        
        try:
            # 检查二维码是否过期
            try:
                expired_selectors = [".qrcode-error-mask", ".qr-expired", ".expired-wrapper"]
                for selector in expired_selectors:
                    try:
                        expired = self.driver.find_element(By.CSS_SELECTOR, selector)
                        if expired.is_displayed():
                            return "expired"
                    except:
                        continue
                
                # 检查页面是否有过期文本
                page_source = self.driver.page_source.lower()
                if "二维码已过期" in page_source or "已失效" in page_source:
                    return "expired"
            except:
                pass
            
            # 检查二维码是否已扫描
            try:
                scanned_selectors = [".qrcode-scanned", ".qr-scanned", ".scanned-wrapper", ".scan-area-scanned"]
                for selector in scanned_selectors:
                    try:
                        scanned = self.driver.find_element(By.CSS_SELECTOR, selector)
                        if scanned.is_displayed():
                            return "scanned"
                    except:
                        continue
                
                # 检查页面是否有扫描成功的文本
                page_source = self.driver.page_source.lower()
                if "已扫描" in page_source or "扫描成功" in page_source:
                    return "scanned"
            except:
                pass
            
            # 检查是否已登录（传统方式）
            if self.is_logged_in():
                return "confirmed"
                
            # 通过JavaScript检测登录状态
            try:
                logged_in = self.driver.execute_script("return document.body.getAttribute('data-jd-logged-in') === 'true';")
                if logged_in:
                    return "confirmed"
            except:
                pass
            
            # 默认二维码有效
            return "valid"
        except Exception as e:
            logger.error(f"检查二维码状态失败: {e}")
            return "valid"
    
    def show_progress_bar(self, elapsed, total, status=""):
        """显示进度条
        
        Args:
            elapsed: 已用时间
            total: 总时间
            status: 状态文本
        """
        progress = min(1.0, elapsed / total)
        bar_length = 40
        block = int(round(bar_length * progress))
        
        # 使用不同颜色的进度条
        if progress < 0.5:
            progress_bar = f"{Colors.GREEN}{'▓' * block}{Colors.RESET}{'░' * (bar_length - block)}"
        elif progress < 0.8:
            progress_bar = f"{Colors.YELLOW}{'▓' * block}{Colors.RESET}{'░' * (bar_length - block)}"
        else:
            progress_bar = f"{Colors.RED}{'▓' * block}{Colors.RESET}{'░' * (bar_length - block)}"
        
        remaining = max(0, total - elapsed)
        
        # 显示百分比和剩余秒数
        percent = int(progress*100)
        time_info = f"{int(remaining)}秒"
        if remaining < 10:
            time_info = f"{Colors.RED}{int(remaining)}秒{Colors.RESET}"
        
        print(f"\r{Colors.BOLD}二维码状态{Colors.RESET} [{progress_bar}] {percent}% {status} 剩余{time_info} ", end="")
    
    def login_by_qrcode(self):
        """扫码登录
        
        Returns:
            bool: 是否登录成功
        """
        try:
            # 检查是否超过最大刷新次数
            if self.qrcode_refresh_count >= self.max_qrcode_refresh:
                logger.warning(f"二维码已刷新{self.qrcode_refresh_count}次，达到最大刷新次数")
                print(f"\n{Colors.YELLOW}已达到最大二维码刷新次数({self.max_qrcode_refresh})，请重新启动程序再试{Colors.RESET}")
                return False
            
            # 增加刷新计数
            self.qrcode_refresh_count += 1
            
            if self.qrcode_refresh_count > 1:
                print(f"\n{Colors.CYAN}正在获取新的二维码 (第{self.qrcode_refresh_count}次尝试){Colors.RESET}")
            
            # 保存当前URL，以便登录后返回
            original_url = self.driver.current_url
            logger.info(f"保存原始URL: {original_url}")
            
            # 强制保存原始URL，确保是商品页面
            if "item.jd.com" in original_url:
                # 将URL保存到localStorage
                self.driver.execute_script(f"localStorage.setItem('jd_original_url', '{original_url}');")
                logger.info(f"商品URL已保存到localStorage: {original_url}")
                
                # 额外打印，确认保存成功
                saved_url = self.driver.execute_script("return localStorage.getItem('jd_original_url');")
                logger.info(f"确认保存的URL: {saved_url}")
                
                # 为了确保可以找回，也保存到全局变量
                global original_product_url
                original_product_url = original_url
                logger.info(f"商品URL已保存到全局变量: {original_product_url}")
            else:
                logger.warning(f"当前URL不是商品页面: {original_url}")
            
            # 显示准备登录信息
            print(f"\n{Colors.CYAN}正在准备京东登录环境...{Colors.RESET}")
            
            # 直接访问京东登录页面（使用新的URL，可能更稳定）
            login_url = "https://passport.jd.com/new/login.aspx?ReturnUrl=https%3A%2F%2Fwww.jd.com%2F"
            logger.info(f"正在打开京东登录页面: {login_url}")
            self.driver.get(login_url)
            
            # 等待页面加载
            print(f"{Colors.YELLOW}正在加载登录页面...{Colors.RESET}")
            time.sleep(3)
            
            # 尝试关闭可能出现的弹窗
            try:
                close_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".close-btn, .ui-dialog-close")
                for btn in close_buttons:
                    if btn.is_displayed():
                        btn.click()
                        logger.info("已关闭弹窗")
                        time.sleep(1)
            except:
                pass
            
            # 启动登录状态监视器 - 传入登录成功回调
            login_success_event = threading.Event()
            self.status_monitor.start_monitoring(
                callback=lambda username: self._on_login_success_callback(username, login_success_event)
            )
            
            # 注入登录检测器
            self.inject_login_detector()
            
            # 切换到扫码登录
            try:
                # 尝试多个可能的选择器
                qrcode_selectors = [
                    ".login-tab-r", 
                    ".login-box-tabs a:nth-child(2)",
                    ".login-method-list li:nth-child(2)",
                    "[data-tab='qrcode']"
                ]
                
                switched = False
                for selector in qrcode_selectors:
                    try:
                        qrcode_tab = WebDriverWait(self.driver, 2).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        qrcode_tab.click()
                        logger.info(f"已切换到扫码登录 (使用选择器: {selector})")
                        time.sleep(1)
                        switched = True
                        break
                    except:
                        continue
                
                if switched:
                    print(f"{Colors.GREEN}已切换到扫码登录模式{Colors.RESET}")
            except:
                logger.info("可能已经在扫码登录页面")
            
            # 获取二维码图片
            print(f"{Colors.YELLOW}正在获取二维码...{Colors.RESET}")
            qrcode_selectors = ["#qrcode", ".qrcode-img img", ".qr-code-img", ".qrcode-img", ".QRCode img"]
            qrcode_img = None
            
            for selector in qrcode_selectors:
                try:
                    qrcode_img = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info(f"找到二维码元素 (使用选择器: {selector})")
                    break
                except:
                    continue
            
            if not qrcode_img:
                logger.error("无法找到二维码元素")
                print(f"\n{Colors.RED}错误: 无法找到二维码元素，请检查网页结构是否变化{Colors.RESET}")
                
                # 尝试截取整个页面，帮助分析
                try:
                    screenshot_path = "login_page_screenshot.png"
                    self.driver.save_screenshot(screenshot_path)
                    print(f"{Colors.YELLOW}已保存登录页面截图: {screenshot_path}{Colors.RESET}")
                    print(f"{Colors.YELLOW}请手动查看截图，确认页面状态{Colors.RESET}")
                    self.open_image(screenshot_path)
                except:
                    pass
                
                self.status_monitor.stop_monitoring()
                return False
            
            # 提取二维码图片
            qrcode_url = None
            try:
                # 尝试获取img标签
                qrcode_url = qrcode_img.get_attribute("src")
            except:
                # 如果不是img标签，尝试找子元素
                try:
                    img_element = qrcode_img.find_element(By.TAG_NAME, "img")
                    qrcode_url = img_element.get_attribute("src")
                except:
                    pass
            
            # 二维码文件名
            qrcode_file = "jd_qrcode.png"
            
            if not qrcode_url:
                # 尝试截图方式获取二维码
                qrcode_img.screenshot(qrcode_file)
                logger.info(f"已截图保存二维码: {qrcode_file}")
                
                # 自动打开二维码图片
                self.open_image(qrcode_file)
                
                # 显示清晰的提示信息
                print("\n" + "="*70)
                print(f"{Colors.CYAN}{Colors.BOLD}【京东扫码登录】{Colors.RESET}")
                print(f"{Colors.GREEN}✅ 二维码已自动打开，请使用京东APP扫码{Colors.RESET}")
                print("-"*70)
                print(f"1️⃣ 二维码保存位置: {Colors.YELLOW}{qrcode_file}{Colors.RESET}")
                print(f"2️⃣ {Colors.YELLOW}请打开京东APP，点击右下角\"我的\"，再点右上角扫码图标{Colors.RESET}")
                print(f"3️⃣ 扫描二维码并在手机上点击\"确认登录\"按钮")
                print(f"4️⃣ {Colors.RED}注意: 二维码有效期仅{self.qrcode_timeout}秒，请尽快完成扫码!{Colors.RESET}")
                print("="*70)
            elif qrcode_url.startswith("data:image"):
                # 处理base64编码的图片
                img_data = qrcode_url.split(",")[1]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_bytes))
                
                # 保存二维码图片
                img.save(qrcode_file)
                
                # 自动打开二维码图片
                self.open_image(qrcode_file)
                
                # 显示清晰的提示信息
                print("\n" + "="*70)
                print(f"{Colors.CYAN}{Colors.BOLD}【京东扫码登录】{Colors.RESET}")
                print(f"{Colors.GREEN}✅ 二维码已自动打开，请使用京东APP扫码{Colors.RESET}")
                print("-"*70)
                print(f"1️⃣ 二维码保存位置: {Colors.YELLOW}{qrcode_file}{Colors.RESET}")
                print(f"2️⃣ {Colors.YELLOW}请打开京东APP，点击右下角\"我的\"，再点右上角扫码图标{Colors.RESET}")
                print(f"3️⃣ 扫描二维码并在手机上点击\"确认登录\"按钮")
                print(f"4️⃣ {Colors.RED}注意: 二维码有效期仅{self.qrcode_timeout}秒，请尽快完成扫码!{Colors.RESET}")
                print("="*70)
            else:
                logger.warning("无法获取二维码图片")
                print(f"\n{Colors.RED}错误: 无法获取二维码图片{Colors.RESET}")
                self.status_monitor.stop_monitoring()
                return False
            
            # 等待扫码登录
            logger.info("等待扫码登录...")
            
            # 循环检查是否登录成功
            start_time = time.time()
            countdown_shown = False  # 是否已显示倒计时
            last_status = "valid"  # 上次的状态
            
            # 设置事件等待，等待登录成功或超时
            print(f"\n{Colors.YELLOW}等待扫码登录，请在{self.qrcode_timeout}秒内完成操作...{Colors.RESET}")
            success = login_success_event.wait(self.timeout)
            
            if success:
                logger.info("登录成功事件触发")
                # 登录状态监控器已处理登录成功逻辑
                # 此时cookies已保存，无需额外操作
                
                # 清除进度条
                print("\r" + " " * 100)
                
                # 返回原来的商品页面
                self.return_to_product_page()
                return True
            
            # 未通过事件触发登录成功，继续使用循环检测
            while time.time() - start_time < self.timeout:
                # 检查二维码状态
                status = self.check_qrcode_status()
                
                # 状态发生变化时显示提示
                if status != last_status:
                    # 清除进度条
                    print("\r" + " " * 100)
                    
                    if status == "scanned" and last_status != "scanned":
                        print("\n" + "="*70)
                        print(f"{Colors.GREEN}{Colors.BOLD}【二维码已扫描成功】{Colors.RESET}")
                        print(f"{Colors.YELLOW}请在手机京东APP上点击\"确认登录\"按钮完成登录{Colors.RESET}")
                        print("="*70)
                    elif status == "expired" and last_status != "expired":
                        print("\n" + "="*70)
                        print(f"{Colors.RED}{Colors.BOLD}【二维码已过期】{Colors.RESET}")
                        print(f"{Colors.YELLOW}正在准备刷新获取新的二维码...{Colors.RESET}")
                        print("="*70)
                    last_status = status
                
                # 根据状态进行处理
                elapsed_time = time.time() - start_time
                remaining_time = max(0, self.qrcode_timeout - elapsed_time)
                
                if status == "confirmed":
                    # 登录成功，保存cookies
                    # 清除进度条
                    print("\r" + " " * 100)
                    
                    logged_in, username = self.inject_login_detector()
                    username = username or "qrcode_login"
                    self._save_cookies(username)
                    print("\n\n" + "="*70)
                    print(f"{Colors.GREEN}{Colors.BOLD}【登录成功】{Colors.RESET}")
                    print(f"京东账号 {Colors.YELLOW}{username}{Colors.RESET} 已成功登录!")
                    print(f"{Colors.GREEN}✓ cookies已保存，下次运行将自动使用cookies登录{Colors.RESET}")
                    print(f"{Colors.GREEN}✓ 登录状态将会保持，除非清除cookies或手动退出登录{Colors.RESET}")
                    print("="*70)
                    print(f"\n{Colors.GREEN}{Colors.BOLD}正在返回原商品页面...{Colors.RESET}")
                    
                    # 返回原来的商品页面
                    self.return_to_product_page()
                    
                    self.status_monitor.stop_monitoring()
                    return True
                elif status == "scanned":
                    # 二维码已扫描，等待确认
                    self.show_progress_bar(elapsed_time, self.timeout, f"{Colors.GREEN}已扫描，等待确认登录{Colors.RESET}")
                elif status == "expired":
                    # 二维码已过期
                    # 清除进度条
                    print("\r" + " " * 100)
                    
                    print("\n\n" + "="*70)
                    print(f"{Colors.RED}{Colors.BOLD}【二维码已过期】{Colors.RESET}")
                    
                    if self.qrcode_auto_refresh and self.qrcode_refresh_count < self.max_qrcode_refresh:
                        remaining_attempts = self.max_qrcode_refresh - self.qrcode_refresh_count
                        print(f"{Colors.YELLOW}系统将自动刷新获取新二维码... (剩余{remaining_attempts}次尝试){Colors.RESET}")
                        print("="*70 + "\n")
                        self.status_monitor.stop_monitoring()
                        return self.login_by_qrcode()  # 递归调用，重新获取二维码
                    else:
                        if self.qrcode_refresh_count >= self.max_qrcode_refresh:
                            print(f"{Colors.RED}已达到最大二维码刷新次数({self.max_qrcode_refresh})，请重新启动程序再试{Colors.RESET}")
                        else:
                            print(f"{Colors.RED}二维码已失效，请重新启动程序进行登录{Colors.RESET}")
                        print("="*70 + "\n")
                        self.status_monitor.stop_monitoring()
                        return False
                else:
                    # 显示等待扫码的进度条
                    status_text = f"{Colors.YELLOW}等待扫码{Colors.RESET}"
                    if remaining_time <= 10:
                        status_text = f"{Colors.RED}即将过期，请尽快扫码{Colors.RESET}"
                    elif remaining_time <= 30:
                        status_text = f"{Colors.YELLOW}请尽快扫码{Colors.RESET}"
                    self.show_progress_bar(elapsed_time, self.qrcode_timeout, status_text)
                
                # 二维码过期检查
                if elapsed_time >= self.qrcode_timeout and status == "valid":
                    # 清除进度条
                    print("\r" + " " * 100)
                    
                    print("\n\n" + "="*70)
                    print(f"{Colors.RED}{Colors.BOLD}【二维码已过期】{Colors.RESET}")
                    
                    if self.qrcode_auto_refresh and self.qrcode_refresh_count < self.max_qrcode_refresh:
                        remaining_attempts = self.max_qrcode_refresh - self.qrcode_refresh_count
                        print(f"{Colors.YELLOW}系统将自动刷新获取新二维码... (剩余{remaining_attempts}次尝试){Colors.RESET}")
                        print("="*70 + "\n")
                        self.status_monitor.stop_monitoring()
                        return self.login_by_qrcode()  # 递归调用，重新获取二维码
                    else:
                        if self.qrcode_refresh_count >= self.max_qrcode_refresh:
                            print(f"{Colors.RED}已达到最大二维码刷新次数({self.max_qrcode_refresh})，请重新启动程序再试{Colors.RESET}")
                        else:
                            print(f"{Colors.RED}二维码已失效，请重新启动程序进行登录{Colors.RESET}")
                        print("="*70 + "\n")
                        self.status_monitor.stop_monitoring()
                        return False
                
                time.sleep(0.1)  # 使进度条更新更平滑
            
            print("\n\n" + "="*70)
            print(f"{Colors.RED}{Colors.BOLD}【错误】扫码登录超时{Colors.RESET}")
            print("请重启程序再试")
            print("="*70 + "\n")
            self.status_monitor.stop_monitoring()
            return False
            
        except Exception as e:
            logger.error(f"扫码登录失败: {e}")
            print("\n" + "="*70)
            print(f"{Colors.RED}{Colors.BOLD}【错误】扫码登录失败: {e}{Colors.RESET}")
            print("请重启程序再试")
            print("="*70 + "\n")
            self.status_monitor.stop_monitoring()
            return False
            
    def return_to_product_page(self):
        """返回到原来的商品页面"""
        try:
            logger.info("尝试返回原始商品页面")
            
            # 尝试多种方法获取原始URL
            original_url = None
            
            # 方法1: 从全局变量获取
            global original_product_url
            if "original_product_url" in globals() and original_product_url and "item.jd.com" in original_product_url:
                original_url = original_product_url
                logger.info(f"从全局变量获取商品URL: {original_url}")
                
            # 方法2: 从localStorage获取
            if not original_url:
                try:
                    saved_url = self.driver.execute_script("return localStorage.getItem('jd_original_url');")
                    if saved_url and "item.jd.com" in saved_url:
                        original_url = saved_url
                        logger.info(f"从localStorage获取商品URL: {original_url}")
                except Exception as e:
                    logger.error(f"从localStorage获取URL失败: {e}")
            
            # 如果找到了原始URL，则返回该页面
            if original_url:
                current_url = self.driver.current_url
                if original_url != current_url:
                    print(f"\n{Colors.GREEN}{Colors.BOLD}正在返回原商品页面...{Colors.RESET}")
                    self.driver.get(original_url)
                    time.sleep(3)
                    logger.info("已返回原始商品页面")
                    return True
                else:
                    logger.info("当前已在商品页面，无需跳转")
                    return True
            else:
                logger.warning("找不到原始商品URL，无法返回")
                print(f"\n{Colors.YELLOW}无法找到原始商品页面链接，请手动导航{Colors.RESET}")
                return False
        except Exception as e:
            logger.error(f"返回原始商品页面失败: {e}")
            print(f"\n{Colors.RED}返回商品页面失败: {e}{Colors.RESET}")
            return False
    
    def login_by_username(self, username, password):
        """账号密码登录
        
        Args:
            username: 用户名/手机号/邮箱
            password: 密码
            
        Returns:
            bool: 是否登录成功
        """
        # 先尝试加载cookies
        if self._load_cookies(username):
            return True
        
        try:
            # 保存当前URL，以便登录后返回
            original_url = self.driver.current_url
            logger.info(f"保存原始URL: {original_url}")
            
            # 强制保存原始URL，确保是商品页面
            if "item.jd.com" in original_url:
                # 将URL保存到localStorage
                self.driver.execute_script(f"localStorage.setItem('jd_original_url', '{original_url}');")
                logger.info(f"商品URL已保存到localStorage: {original_url}")
                
                # 额外打印，确认保存成功
                saved_url = self.driver.execute_script("return localStorage.getItem('jd_original_url');")
                logger.info(f"确认保存的URL: {saved_url}")
                
                # 为了确保可以找回，也保存到全局变量
                global original_product_url
                original_product_url = original_url
                logger.info(f"商品URL已保存到全局变量: {original_product_url}")
            else:
                logger.warning(f"当前URL不是商品页面: {original_url}")
            
            # 访问京东登录页
            self.driver.get("https://passport.jd.com/new/login.aspx")
            logger.info("正在加载登录页面...")
            time.sleep(3)
            
            # 启动登录状态监视器 - 传入登录成功回调
            login_success_event = threading.Event()
            self.status_monitor.start_monitoring(
                callback=lambda username: self._on_login_success_callback(username, login_success_event)
            )
            
            # 注入登录检测器
            self.inject_login_detector()
            
            # 切换到账号登录
            try:
                account_tab = self.driver.find_element(By.CLASS_NAME, "login-tab-l")
                account_tab.click()
                logger.info("已切换到账号登录")
                time.sleep(1)
            except (NoSuchElementException, TimeoutException):
                logger.info("已经是账号登录页面")
            
            # 输入账号密码
            username_input = self.driver.find_element(By.ID, "loginname")
            password_input = self.driver.find_element(By.ID, "nloginpwd")
            
            username_input.clear()
            username_input.send_keys(username)
            
            password_input.clear()
            password_input.send_keys(password)
            
            logger.info("已输入账号密码")
            
            # 点击登录按钮
            login_button = self.driver.find_element(By.ID, "loginsubmit")
            login_button.click()
            logger.info("已点击登录按钮")
            
            # 等待可能出现的滑块验证码
            time.sleep(3)
            
            # 处理滑块验证码
            try:
                slider = self.driver.find_element(By.CLASS_NAME, "JDJRV-slide-btn")
                if slider.is_displayed():
                    print("\n" + "="*70)
                    print(f"{Colors.YELLOW}{Colors.BOLD}【需要验证】{Colors.RESET}")
                    print("请在浏览器中完成滑块验证码")
                    print("完成验证后程序将继续运行")
                    print("="*70 + "\n")
                    
                    # 等待用户手动处理验证码
                    start_time = time.time()
                    while time.time() - start_time < self.timeout:
                        status = self.check_qrcode_status()
                        if status == "confirmed":
                            break
                            
                        # 显示等待验证的进度条
                        elapsed_time = time.time() - start_time
                        self.show_progress_bar(elapsed_time, self.timeout, f"{Colors.YELLOW}等待验证完成{Colors.RESET}")
                        time.sleep(0.1)
            except (NoSuchElementException, TimeoutException):
                pass
            
            # 等待登录成功
            print("\n" + "="*70)
            print(f"{Colors.BOLD}【登录中】{Colors.RESET}请稍候...")
            print("="*70)
            
            # 设置事件等待，等待登录成功或超时
            success = login_success_event.wait(self.timeout)
            
            if success:
                logger.info("登录成功事件触发")
                # 登录状态监控器已处理登录成功逻辑
                # 此时cookies已保存，无需额外操作
                
                # 返回原来的商品页面
                self.return_to_product_page()
                return True
            
            # 未通过事件触发登录成功，继续使用循环检测
            start_time = time.time()
            login_success = False
            
            while time.time() - start_time < self.timeout:
                # 通过多种方式检测登录状态
                status = self.check_qrcode_status()
                
                if status == "confirmed":
                    login_success = True
                    break
                
                # 显示等待登录的进度条
                elapsed_time = time.time() - start_time
                self.show_progress_bar(elapsed_time, self.timeout, f"{Colors.BLUE}等待登录确认{Colors.RESET}")
                time.sleep(0.1)
            
            if login_success:
                # 登录成功，保存cookies
                logged_in, detected_username = self.inject_login_detector()
                save_username = detected_username or username
                self._save_cookies(save_username)
                
                print("\n\n" + "="*70)
                print(f"{Colors.GREEN}{Colors.BOLD}【登录成功】{Colors.RESET}")
                print(f"京东账号 {Colors.YELLOW}{save_username}{Colors.RESET} 已成功登录!")
                print(f"{Colors.GREEN}✓ cookies已保存，下次运行将自动使用cookies登录{Colors.RESET}")
                print(f"{Colors.GREEN}✓ 登录状态将会保持，除非清除cookies或手动退出登录{Colors.RESET}")
                print("="*70)
                print(f"\n{Colors.GREEN}{Colors.BOLD}正在返回原商品页面...{Colors.RESET}")
                
                # 返回原来的商品页面
                self.return_to_product_page()
                
                self.status_monitor.stop_monitoring()
                return True
            else:
                print("\n\n" + "="*70)
                print(f"{Colors.RED}{Colors.BOLD}【错误】账号密码登录超时{Colors.RESET}")
                print("请检查账号密码是否正确，或者尝试扫码登录")
                print("="*70 + "\n")
                self.status_monitor.stop_monitoring()
                return False
            
        except Exception as e:
            logger.error(f"账号密码登录失败: {e}")
            print("\n" + "="*70)
            print(f"{Colors.RED}{Colors.BOLD}【错误】账号密码登录失败: {e}{Colors.RESET}")
            print("请重启程序再试")
            print("="*70 + "\n")
            self.status_monitor.stop_monitoring()
            return False
    
    def auto_login(self, username=None, password=None):
        """自动选择登录方式
        
        根据提供的参数选择登录方式：
        1. 如果有username和password，尝试账号密码登录
        2. 否则，使用扫码登录
        
        Args:
            username: 用户名/手机号/邮箱（可选）
            password: 密码（可选）
            
        Returns:
            bool: 是否登录成功
        """
        print("\n" + "="*70)
        print(f"{Colors.BOLD}【京东爬虫】登录流程启动{Colors.RESET}")
        print("="*70)
        
        # 添加全局异常处理
        try:
            # 先检查是否已经登录
            try:
                if self.is_logged_in():
                    logger.info("检测到已登录状态，无需重新登录")
                    print(f"\n{Colors.GREEN}检测到已登录状态，无需重新登录{Colors.RESET}")
                    return True
            except Exception as e:
                logger.warning(f"登录状态检查异常，将尝试重新登录: {e}")
                # 继续进行登录流程
            
            # 首先，如果有保存的cookies，尝试加载
            if username:
                try:
                    logger.info(f"尝试加载保存的cookies: {username}")
                    if self._load_cookies(username):
                        return True
                except Exception as e:
                    logger.warning(f"加载cookies失败，将尝试重新登录: {e}")
                    # 继续进行登录流程
            
            # 根据提供的参数选择登录方式
            if username and password:
                # 使用账号密码登录
                return self.login_by_username(username, password)
            else:
                # 使用扫码登录
                return self.login_by_qrcode()
                
        except Exception as e:
            logger.error(f"自动登录过程失败: {e}")
            print(f"\n{Colors.RED}登录过程中发生错误: {e}{Colors.RESET}")
            
            # 尝试恢复到商品页面
            try:
                # 检查是否可以找到原始商品URL
                global original_product_url
                if "original_product_url" in globals() and original_product_url and "jd.com" in original_product_url:
                    logger.info(f"尝试返回原始商品页面: {original_product_url}")
                    print(f"\n{Colors.YELLOW}尝试返回原始商品页面...{Colors.RESET}")
                    self.driver.get(original_product_url)
                    time.sleep(3)
            except:
                pass
                
            return False 