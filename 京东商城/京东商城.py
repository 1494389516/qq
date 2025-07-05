import os
import time
import random
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pyautogui
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from config import SPIDER_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JDSliderCracker:
    """处理京东滑块验证码"""
    
    def __init__(self, driver):
        self.driver = driver
        
    def handle_slider(self):
        """处理滑块验证码"""
        try:
            # 等待滑块元素出现
            logger.info("等待滑块验证码出现...")
            
            # 尝试多个可能的滑块选择器
            slider_selectors = [
                ".JDJRV-slide-btn",
                ".slide-btn",
                ".sliderBlock",
                ".slider_block",
                "#nc_1_n1z"  # 阿里云盾滑块
            ]
            
            slider = None
            for selector in slider_selectors:
                try:
                    slider = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info(f"找到滑块元素: {selector}")
                    break
                except:
                    continue
            
            if not slider:
                logger.error("未找到滑块元素")
                return False
            
            # 获取滑块的位置和大小
            slider_location = slider.location
            slider_size = slider.size
            
            # 计算滑动起点和终点
            start_x = slider_location['x'] + slider_size['width'] // 2
            start_y = slider_location['y'] + slider_size['height'] // 2
            
            # 模拟人工滑动：先快后慢
            pyautogui.moveTo(start_x, start_y)
            pyautogui.mouseDown()
            
            # 计算滑动距离（这里需要根据实际情况调整）
            # 京东的滑块距离通常在100-200像素之间
            distance = random.randint(
                SPIDER_CONFIG['VERIFICATION']['SLIDER_DISTANCE_MIN'], 
                SPIDER_CONFIG['VERIFICATION']['SLIDER_DISTANCE_MAX']
            )
            
            # 分段滑动，模拟人工操作
            steps = SPIDER_CONFIG['VERIFICATION']['SLIDER_STEPS']
            
            # 使用更真实的轨迹
            track = self._generate_track(distance, steps)
            
            # 按照轨迹移动
            current_x = 0
            for x_offset in track:
                current_x += x_offset
                pyautogui.moveTo(start_x + current_x, start_y + random.randint(-2, 2))
                time.sleep(random.uniform(0.01, 0.03))
            
            # 最后释放鼠标
            pyautogui.mouseUp()
            logger.info("滑块验证完成")
            
            # 等待验证结果
            time.sleep(2)
            
            # 检查验证是否成功
            try:
                # 尝试检测成功或失败的提示
                success_elements = self.driver.find_elements(By.CSS_SELECTOR, ".verify-success, .success-tip")
                if any(elem.is_displayed() for elem in success_elements):
                    logger.info("滑块验证成功")
                    return True
                
                # 检查是否有失败提示
                fail_elements = self.driver.find_elements(By.CSS_SELECTOR, ".verify-fail, .fail-tip")
                if any(elem.is_displayed() for elem in fail_elements):
                    logger.warning("滑块验证失败，将重试")
                    time.sleep(1)
                    return self.handle_slider()  # 递归重试
            except:
                pass
            
            return True
        
        except Exception as e:
            logger.error(f"处理滑块验证码失败: {e}")
            return False
    
    def _generate_track(self, distance, steps):
        """生成更真实的轨迹
        
        Args:
            distance: 总距离
            steps: 步数
            
        Returns:
            list: 每一步的位移
        """
        # 创建一个加速-减速的轨迹
        track = []
        
        # 加速阶段
        mid = steps * 0.7
        for i in range(int(mid)):
            factor = i / mid
            offset = int(distance * 0.6 * factor * factor)
            track.append(offset - sum(track))
        
        # 减速阶段
        for i in range(int(mid), steps):
            factor = (steps - i) / (steps - mid)
            offset = int(distance * 0.4 * factor * factor) + int(distance * 0.6)
            track.append(offset - sum(track))
        
        # 微调，确保总和等于距离
        track[-1] += (distance - sum(track))
        
        return track

class JDVerificationHandler:
    """处理京东各种验证机制"""
    
    def __init__(self, driver):
        self.driver = driver
        self.slider_cracker = JDSliderCracker(driver)
    
    def handle_quick_verification(self):
        """处理快速验证按钮"""
        try:
            # 尝试多个可能的验证按钮选择器
            verify_button_selectors = [
                "#JDJRV-wrap-loginsubmit",
                ".JDJRV-verify-btn",
                ".verify-btn",
                ".verify-button",
                ".btn-verify",
                ".btn-primary"
            ]
            
            for selector in verify_button_selectors:
                try:
                    logger.info(f"尝试查找验证按钮: {selector}")
                    verify_button = WebDriverWait(self.driver, 2).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )
                    
                    # 点击验证按钮
                    verify_button.click()
                    logger.info(f"已点击验证按钮: {selector}")
                    time.sleep(1)
                    return True
                except:
                    continue
            
            logger.info("未发现验证按钮，可能不需要验证")
            return False
        except Exception as e:
            logger.error(f"处理快速验证失败: {e}")
            return False
    
    def handle_image_verification(self):
        """处理图形验证码"""
        try:
            # 检查是否有图形验证码
            image_selectors = [
                ".verify-img-panel img",
                ".verify-image",
                ".captcha-img",
                "#captcha_img"
            ]
            
            for selector in image_selectors:
                try:
                    image_element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if image_element.is_displayed():
                        logger.info(f"检测到图形验证码: {selector}")
                        
                        # 保存验证码图片
                        image_path = "jd_captcha.png"
                        image_element.screenshot(image_path)
                        
                        # 提示用户手动处理
                        print("\n" + "="*60)
                        print("检测到图形验证码，需要手动输入")
                        print(f"验证码图片已保存到: {image_path}")
                        print("请查看图片并在下方输入验证码")
                        print("="*60)
                        
                        # 打开图片
                        try:
                            import webbrowser
                            webbrowser.open(image_path)
                        except:
                            pass
                        
                        # 获取用户输入
                        captcha_code = input("请输入验证码: ")
                        
                        # 查找验证码输入框
                        input_selectors = [
                            ".verify-input",
                            ".captcha-input",
                            "#captcha_input",
                            "input[name='captcha']",
                            "input[type='text']"
                        ]
                        
                        for input_selector in input_selectors:
                            try:
                                input_element = self.driver.find_element(By.CSS_SELECTOR, input_selector)
                                input_element.clear()
                                input_element.send_keys(captcha_code)
                                logger.info(f"已输入验证码: {captcha_code}")
                                
                                # 查找提交按钮
                                submit_selectors = [
                                    ".verify-submit",
                                    ".submit-btn",
                                    ".btn-submit",
                                    "button[type='submit']",
                                    ".btn-primary"
                                ]
                                
                                for submit_selector in submit_selectors:
                                    try:
                                        submit_button = self.driver.find_element(By.CSS_SELECTOR, submit_selector)
                                        submit_button.click()
                                        logger.info("已提交验证码")
                                        time.sleep(2)
                                        return True
                                    except:
                                        continue
                                
                                # 如果没有找到提交按钮，尝试按回车键
                                input_element.send_keys("\n")
                                logger.info("已通过回车键提交验证码")
                                time.sleep(2)
                                return True
                            except:
                                continue
                        
                        logger.warning("未找到验证码输入框")
                        return False
                except:
                    continue
            
            return False
        except Exception as e:
            logger.error(f"处理图形验证码失败: {e}")
            return False
    
    def handle_verification(self):
        """处理所有验证流程"""
        # 检查是否需要处理验证
        need_verification = False
        
        # 保存页面截图，用于分析
        try:
            screenshot_path = "verification_page.png"
            self.driver.save_screenshot(screenshot_path)
            logger.info(f"已保存页面截图: {screenshot_path}")
        except:
            pass
        
        # 首先检查是否在登录页面，如果是普通登录页面则不需要处理验证
        current_url = self.driver.current_url.lower()
        if "passport.jd.com/new/login.aspx" in current_url and "verify" not in current_url:
            try:
                # 检查是否有登录表单元素，如果有则说明是正常登录页面
                login_elements = self.driver.find_elements(By.CSS_SELECTOR, ".login-tab, .login-box, .login-form")
                if login_elements and any(elem.is_displayed() for elem in login_elements):
                    logger.info("检测到普通登录页面，无需验证处理")
                    return
            except:
                pass
        
        try:
            # 检查是否有明确的验证元素
            verification_selectors = [
                ".JDJRV-suspend",
                ".verify-wrap",
                ".verify-box",
                ".captcha-wrap",
                "#captcha-box",
                ".slider-tips",  # 滑块提示文字
                ".verify-title"  # 验证标题
            ]
            
            for selector in verification_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements and any(elem.is_displayed() for elem in elements):
                    need_verification = True
                    logger.info(f"检测到验证元素: {selector}")
                    break
        except:
            pass
            
        try:
            # 检查URL是否包含明确的验证相关参数
            verification_keywords = ["verify", "captcha", "validate", "authcode"]
            if any(keyword in current_url for keyword in verification_keywords):
                need_verification = True
                logger.info(f"检测到验证页面URL: {current_url}")
        except:
            pass
        
        # 检查页面内容是否包含验证相关文本
        try:
            page_source = self.driver.page_source.lower()
            verification_texts = ["请完成验证", "安全验证", "滑动验证", "拖动滑块", "拼图验证"]
            if any(text in page_source for text in verification_texts):
                need_verification = True
                logger.info(f"检测到验证相关文本")
        except:
            pass
            
        if not need_verification:
            logger.info("无需验证")
            return
        
        print("\n" + "="*60)
        print("检测到京东验证页面，正在尝试自动处理...")
        print("="*60)
            
        # 尝试处理快速验证
        if self.handle_quick_verification():
            logger.info("快速验证处理完成")
            time.sleep(2)
        
        # 尝试处理图形验证码
        if self.handle_image_verification():
            logger.info("图形验证码处理完成")
            time.sleep(2)
        
        # 尝试处理滑块验证码
        if self.slider_cracker.handle_slider():
            logger.info("滑块验证码处理成功")
        else:
            logger.warning("滑块验证码处理可能失败")
            
            # 如果自动处理失败，提示用户手动处理
            print("\n" + "="*60)
            print("自动处理验证失败，请手动完成验证")
            print("请在浏览器窗口中手动完成验证操作")
            print("完成后请按回车键继续...")
            print("="*60)
            input()
        
        # 等待页面加载完成
        time.sleep(3)
        
        # 检查是否被重定向到其他页面
        if "item.jd.com" not in self.driver.current_url:
            logger.warning(f"验证后被重定向到: {self.driver.current_url}")
            # 这里不做处理，让主程序处理重定向
    
    def handle_url_redirect(self, original_url):
        """处理URL重定向
        
        Args:
            original_url: 原始URL
            
        Returns:
            str: 处理后的URL
        """
        try:
            current_url = self.driver.current_url
            config = SPIDER_CONFIG['URL_REDIRECT']
            
            # 如果当前URL与原始URL不同，可能发生了重定向
            if current_url != original_url:
                logger.info(f"检测到URL重定向: {original_url} -> {current_url}")
                
                # 检查是否重定向到了移动端页面（item.m.jd.com）
                if "item.m.jd.com" in current_url and config.get('FORCE_PC', True):
                    logger.info("检测到重定向到移动版页面，尝试强制使用PC版")
                    
                    # 从URL中提取商品ID
                    try:
                        # 提取商品ID的不同方式
                        product_id = None
                        
                        # 移动版URL格式: https://item.m.jd.com/product/100050401004.html
                        if "product/" in current_url:
                            product_id = current_url.split("product/")[1].split(".")[0]
                            logger.info(f"从移动版URL提取到商品ID: {product_id}")
                        
                        # 尝试从原始URL提取
                        if not product_id and "item.jd.com" in original_url:
                            product_id = original_url.split("/")[-1].split(".")[0]
                            logger.info(f"从原始URL提取到商品ID: {product_id}")
                        
                        if product_id:
                            # 构建PC版URL
                            pc_url = f"https://item.jd.com/{product_id}.html"
                            logger.info(f"强制使用PC版URL: {pc_url}")
                            
                            # 设置PC版User-Agent
                            if config.get('USER_AGENT_ROTATION', True):
                                self.driver.execute_script("""
                                    Object.defineProperty(navigator, 'userAgent', {
                                        get: function () { 
                                            return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'; 
                                        }
                                    });
                                """)
                                logger.info("已注入PC版User-Agent")
                            
                            # 尝试访问PC版URL
                            self.driver.get(pc_url)
                            time.sleep(3)
                            
                            # 如果还是被重定向到移动版，尝试更强的方法
                            if "item.m.jd.com" in self.driver.current_url and config.get('DIRECT_PC_DOMAIN', True):
                                logger.info("仍然被重定向到移动版，尝试通过修改请求头访问")
                                
                                # 通过执行JavaScript来修改请求头并访问
                                js_code = f"""
                                    var xhr = new XMLHttpRequest();
                                    xhr.open('GET', '{pc_url}', false);
                                    xhr.setRequestHeader('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36');
                                    xhr.setRequestHeader('Accept', 'text/html,application/xhtml+xml,application/xml');
                                    xhr.setRequestHeader('Accept-Language', 'zh-CN,zh;q=0.9,en;q=0.8');
                                    xhr.setRequestHeader('Cache-Control', 'no-cache');
                                    xhr.setRequestHeader('Pragma', 'no-cache');
                                    xhr.setRequestHeader('Sec-Ch-Ua', '"Chromium";v="122", "Google Chrome";v="122"');
                                    xhr.setRequestHeader('Sec-Ch-Ua-Mobile', '?0');
                                    xhr.setRequestHeader('Sec-Ch-Ua-Platform', '"Windows"');
                                    xhr.setRequestHeader('Sec-Fetch-Dest', 'document');
                                    xhr.setRequestHeader('Sec-Fetch-Mode', 'navigate');
                                    xhr.setRequestHeader('Sec-Fetch-Site', 'none');
                                    xhr.setRequestHeader('Sec-Fetch-User', '?1');
                                    xhr.setRequestHeader('Upgrade-Insecure-Requests', '1');
                                    xhr.send();
                                    document.open();
                                    document.write(xhr.responseText);
                                    document.close();
                                    window.history.pushState({{}},'', '{pc_url}');
                                """
                                
                                try:
                                    self.driver.execute_script(js_code)
                                    time.sleep(2)
                                    logger.info(f"已尝试通过JS方式强制访问PC版: 当前URL={self.driver.current_url}")
                                except Exception as e:
                                    logger.error(f"JS方式访问PC版失败: {e}")
                                
                                # 如果还是失败，最后一次尝试直接访问
                                if "item.m.jd.com" in self.driver.current_url:
                                    # 为后续处理保存商品ID
                                    self.driver.execute_script(f"localStorage.setItem('jd_product_id', '{product_id}');")
                                    
                                    # 再尝试加载一次PC版URL
                                    self.driver.get(pc_url)
                                    time.sleep(3)
                            
                            return self.driver.current_url
                    except Exception as e:
                        logger.error(f"强制PC版处理失败: {e}")
                
                # 检查是否重定向到了PC商品页面
                if "item.jd.com" in current_url and "item.m.jd.com" not in current_url:
                    logger.info(f"重定向到了其他PC商品页面")
                    return current_url
                
                # 检查是否重定向到了登录页面
                if "passport.jd.com" in current_url:
                    logger.info(f"重定向到了登录页面，可能需要登录")
                    return original_url  # 返回原始URL，让登录模块处理
                
                # 检查是否重定向到了验证页面
                if "verify" in current_url:
                    logger.info(f"重定向到了验证页面，需要处理验证")
                    # 处理验证
                    self.handle_verification()
                    # 验证后尝试返回原始页面
                    self.driver.get(original_url)
                    time.sleep(3)
                    return self.driver.current_url
                
                # 其他情况，尝试从页面中提取商品ID
                try:
                    # 尝试从URL中提取商品ID
                    if "item.jd.com" in original_url:
                        product_id = original_url.split("/")[-1].split(".")[0]
                        logger.info(f"从原始URL提取到商品ID: {product_id}")
                        
                        # 构建新的URL
                        new_url = f"https://item.jd.com/{product_id}.html"
                        logger.info(f"尝试访问新构建的URL: {new_url}")
                        
                        # 访问新URL
                        self.driver.get(new_url)
                        time.sleep(3)
                        
                        # 检查是否成功访问
                        if "item.jd.com" in self.driver.current_url:
                            logger.info(f"成功访问商品页面: {self.driver.current_url}")
                            return self.driver.current_url
                except Exception as e:
                    logger.error(f"处理商品ID失败: {e}")
            
            return original_url
        except Exception as e:
            logger.error(f"处理URL重定向失败: {e}")
            return original_url
    
    def check_login_required(self):
        """检查是否需要登录
        
        Returns:
            bool: 是否需要登录
        """
        try:
            # 方法1：检查"请登录"文字
            try:
                login_text_selectors = [
                    "//div[contains(text(), '请登录') or contains(text(), '登录')]",
                    "//a[contains(text(), '请登录') or contains(text(), '登录')]",
                    "//span[contains(text(), '请登录') or contains(text(), '登录')]"
                ]
                
                for selector in login_text_selectors:
                    try:
                        login_text = WebDriverWait(self.driver, 2).until(
                            EC.presence_of_element_located((By.XPATH, selector))
                        )
                        if login_text and login_text.is_displayed():
                            logger.info(f"检测到需要登录文本: {login_text.text}")
                            return True
                    except:
                        continue
            except:
                pass
                
            # 方法2：检查登录按钮
            login_button_selectors = ["请登录", "立即登录", "登录", "账号登录"]
            for text in login_button_selectors:
                try:
                    login_button = self.driver.find_element(By.LINK_TEXT, text)
                    if login_button and login_button.is_displayed():
                        logger.info(f"检测到登录按钮: {text}")
                        return True
                except:
                    continue
                
            # 方法3：检查商品价格是否可见
            try:
                # 尝试多个价格选择器
                price_selectors = [
                    ".price",
                    ".p-price",
                    "#jd-price",
                    ".J-p-*"  # 动态ID
                ]
                
                price_found = False
                for selector in price_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for price_element in elements:
                            price_text = price_element.text.strip() if price_element else ""
                            
                            if price_text and ("¥" in price_text or "￥" in price_text):
                                price_found = True
                                logger.info(f"找到价格元素: {price_text}")
                                break
                        
                        if price_found:
                            break
                    except:
                        continue
                
                if not price_found:
                    logger.info("价格信息不可见，可能需要登录")
                    return True
            except:
                # 找不到价格元素，可能需要登录
                logger.info("找不到价格元素，可能需要登录")
                return True
                
            # 方法4：检查是否已登录
            try:
                # 检查是否有用户名显示
                nickname_selectors = [
                    ".nickname",
                    ".user-name",
                    ".login-user",
                    ".user-info"
                ]
                
                for selector in nickname_selectors:
                    try:
                        nickname = self.driver.find_element(By.CSS_SELECTOR, selector)
                        if nickname and nickname.text and "请登录" not in nickname.text:
                            logger.info(f"已经登录: {nickname.text}")
                            return False
                    except:
                        continue
            except:
                # 找不到用户名，可能需要登录
                pass
            
            logger.info("未明确检测到登录需求")
            return False
            
        except Exception as e:
            logger.error(f"登录检测失败: {e}")
            return False

class JDProductItem(scrapy.Item):
    """京东商品信息Item"""
    product_id = scrapy.Field()
    title = scrapy.Field()
    price = scrapy.Field()
    shop_name = scrapy.Field()
    category = scrapy.Field()
    description = scrapy.Field()
    images = scrapy.Field()
    specs = scrapy.Field()
    comments_count = scrapy.Field()
    good_rate = scrapy.Field()
    url = scrapy.Field()
    crawl_time = scrapy.Field()

class JDSeleniumMiddleware:
    """Scrapy中间件，使用Selenium处理动态内容和验证"""
    
    def __init__(self):
        # 设置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-extensions")
        
        # 如果配置为无头模式，则添加无头参数
        if SPIDER_CONFIG['BROWSER']['HEADLESS']:
            chrome_options.add_argument("--headless")
        
        # 设置窗口大小
        window_size = SPIDER_CONFIG['BROWSER']['WINDOW_SIZE']
        chrome_options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
        
        # 禁用自动化控制特征
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # 使用当前目录下的chromedriver
        driver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromedriver.exe")
        service = Service(driver_path)
        
        # 初始化WebDriver
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # 设置页面加载超时和隐式等待时间
        self.driver.set_page_load_timeout(SPIDER_CONFIG['BROWSER']['PAGE_LOAD_TIMEOUT'])
        self.driver.implicitly_wait(SPIDER_CONFIG['BROWSER']['IMPLICIT_WAIT'])
        
        # 执行JavaScript代码以绕过反爬
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """
        })
        
        # 初始化验证处理器
        self.verification_handler = JDVerificationHandler(self.driver)
        
    def process_request(self, request, spider):
        """处理请求，使用Selenium加载页面并处理验证"""
        url = request.url
        logger.info(f"正在处理URL: {url}")
        
        try:
            self.driver.get(url)
            # 等待页面加载
            time.sleep(random.uniform(2, 4))
            
            # 处理各种验证
            self.verification_handler.handle_verification()
            
            # 检查是否需要登录
            if self.verification_handler.check_login_required():
                logger.warning("需要登录才能查看完整信息，部分数据可能无法获取")
            
            # 返回页面内容
            body = self.driver.page_source
            return scrapy.http.HtmlResponse(
                url=url,
                body=body,
                encoding='utf-8',
                request=request
            )
        except Exception as e:
            logger.error(f"Selenium处理请求失败: {e}")
            # 重试机制
            if request.meta.get('retry_times', 0) < SPIDER_CONFIG['CRAWLER']['RETRY_TIMES']:
                logger.info(f"尝试重新请求 {url}")
                request.meta['retry_times'] = request.meta.get('retry_times', 0) + 1
                time.sleep(SPIDER_CONFIG['CRAWLER']['RETRY_DELAY'])
                return request
            else:
                logger.error(f"达到最大重试次数，放弃请求 {url}")
                return None
    
    def close_spider(self, spider):
        """关闭Spider时关闭浏览器"""
        if self.driver:
            self.driver.quit()
            logger.info("已关闭浏览器")

class JDProductSpider(scrapy.Spider):
    """京东商品爬虫"""
    name = 'jd_product'
    
    def __init__(self, url=None, *args, **kwargs):
        super(JDProductSpider, self).__init__(*args, **kwargs)
        self.start_urls = [url] if url else [SPIDER_CONFIG['TARGET_URL']]
    
    def parse(self, response):
        """解析商品页面"""
        from datetime import datetime
        import re
        
        item = JDProductItem()
        
        # 提取商品ID
        product_id = response.url.split('/')[-1].split('.')[0]
        item['product_id'] = product_id
        item['url'] = response.url
        item['crawl_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 提取商品标题
        try:
            # 方法1: 使用CSS选择器
            title = response.css('div.sku-name::text').get()
            if title and title.strip():
                item['title'] = title.strip()
            else:
                # 方法2: 使用XPath
                title = response.xpath('//div[contains(@class, "sku-name")]/text()').get()
                if title and title.strip():
                    item['title'] = title.strip()
                else:
                    # 方法3: 查找h1标题
                    title = response.css('h1::text').get()
                    if title and title.strip():
                        item['title'] = title.strip()
                    else:
                        # 方法4: 在页面源码中查找title标签
                        title_match = re.search(r'<title>(.*?)-(京东|JD).*?</title>', response.text)
                        if title_match:
                            item['title'] = title_match.group(1).strip()
                        else:
                            item['title'] = "无法获取标题"
        except Exception as e:
            logger.error(f"提取标题失败: {e}")
            item['title'] = "提取失败"
        
        # 提取价格
        try:
            # 方法1: 页面上的价格元素
            price = response.css('span.price::text').get() or response.css('.p-price .price::text').get()
            if price and price.strip():
                item['price'] = price.strip()
            else:
                # 方法2: 尝试从JS变量中提取
                price_match = re.search(r'"p":"([^"]+)"', response.text)
                if price_match:
                    item['price'] = price_match.group(1)
                else:
                    # 方法3: 尝试从其他可能的元素中提取
                    price = response.css('.J-p-{} .price::text'.format(product_id)).get()
                    if price and price.strip():
                        item['price'] = price.strip()
                    else:
                        item['price'] = "需要登录查看"
        except Exception as e:
            logger.error(f"提取价格失败: {e}")
            item['price'] = "提取失败"
        
        # 提取店铺名称
        try:
            # 方法1
            shop_name = response.css('div.J-hove-wrap.EDropdown.fr a::text').get()
            if shop_name and shop_name.strip():
                item['shop_name'] = shop_name.strip()
            else:
                # 方法2
                shop_name = response.css('div.shopName a::text').get()
                if shop_name and shop_name.strip():
                    item['shop_name'] = shop_name.strip()
                else:
                    # 方法3
                    shop_name = response.css('#popbox .mt a::text').get()
                    if shop_name and shop_name.strip():
                        item['shop_name'] = shop_name.strip()
                    else:
                        # 方法4: 从JS变量中提取
                        shop_match = re.search(r'"shopName":"([^"]+)"', response.text)
                        if shop_match:
                            item['shop_name'] = shop_match.group(1)
                        else:
                            item['shop_name'] = "无法获取店铺名称"
        except Exception as e:
            logger.error(f"提取店铺名称失败: {e}")
            item['shop_name'] = "提取失败"
        
        # 提取商品分类
        try:
            categories = response.css('div.crumb-wrap .item a::text').getall()
            if categories:
                item['category'] = '>'.join([cat.strip() for cat in categories if cat.strip()])
            else:
                # 尝试其他可能的选择器
                categories = response.css('div.crumbs-nav .item a::text').getall()
                if categories:
                    item['category'] = '>'.join([cat.strip() for cat in categories if cat.strip()])
                else:
                    item['category'] = "无法获取分类"
        except Exception as e:
            logger.error(f"提取分类失败: {e}")
            item['category'] = "提取失败"
        
        # 提取商品描述
        try:
            descriptions = response.css('div.p-parameter ul.parameter2 li::text').getall()
            if descriptions:
                item['description'] = '\n'.join([desc.strip() for desc in descriptions if desc.strip()])
            else:
                # 尝试其他可能的选择器
                descriptions = response.xpath('//div[contains(@class, "p-parameter")]//li/text()').getall()
                if descriptions:
                    item['description'] = '\n'.join([desc.strip() for desc in descriptions if desc.strip()])
                else:
                    # 尝试从详情页提取
                    description = response.css('#detail .detail-content-item::text').get()
                    if description:
                        item['description'] = description.strip()
                    else:
                        item['description'] = "无法获取描述"
        except Exception as e:
            logger.error(f"提取描述失败: {e}")
            item['description'] = "提取失败"
        
        # 提取商品图片
        try:
            # 方法1
            images = response.css('div#spec-list img::attr(src)').getall()
            if images:
                item['images'] = ['https:' + img if not img.startswith('http') else img for img in images]
            else:
                # 方法2
                images = response.css('ul.lh li img::attr(src)').getall()
                if images:
                    item['images'] = ['https:' + img if not img.startswith('http') else img for img in images]
                else:
                    # 方法3
                    images = response.css('#spec-list .img-hover img::attr(src)').getall()
                    if images:
                        item['images'] = ['https:' + img if not img.startswith('http') else img for img in images]
                    else:
                        # 方法4: 尝试从JS变量中提取
                        image_match = re.search(r'"imageList":\[(.*?)\]', response.text)
                        if image_match:
                            images_json = "[" + image_match.group(1) + "]"
                            try:
                                import json
                                img_list = json.loads(images_json.replace("'", '"'))
                                item['images'] = ['https:' + img if not img.startswith('http') else img for img in img_list]
                            except:
                                item['images'] = []
                        else:
                            item['images'] = []
        except Exception as e:
            logger.error(f"提取图片失败: {e}")
            item['images'] = []
        
        # 提取规格参数
        try:
            specs = {}
            # 方法1
            spec_items = response.css('div.p-parameter ul.parameter2 li')
            for spec in spec_items:
                spec_text = spec.css('::text').get()
                if spec_text and ':' in spec_text:
                    key, value = spec_text.split(':', 1)
                    specs[key.strip()] = value.strip()
            
            # 如果没有找到规格，尝试其他选择器
            if not specs:
                # 方法2
                spec_items = response.xpath('//div[contains(@class, "Ptable")]//dl')
                for spec in spec_items:
                    key = spec.xpath('./dt/text()').get()
                    value = spec.xpath('./dd/text()').get()
                    if key and value:
                        specs[key.strip()] = value.strip()
            
            item['specs'] = specs
        except Exception as e:
            logger.error(f"提取规格参数失败: {e}")
            item['specs'] = {}
        
        # 提取评论数和好评率
        try:
            # 方法1
            comments_count = response.css('div#comment div.count a::text').get()
            if comments_count and comments_count.strip():
                item['comments_count'] = comments_count.strip()
            else:
                # 方法2
                comments_match = re.search(r'"commentCount":(\d+)', response.text)
                if comments_match:
                    item['comments_count'] = comments_match.group(1)
                else:
                    item['comments_count'] = "无法获取评论数"
        except Exception as e:
            logger.error(f"提取评论数失败: {e}")
            item['comments_count'] = "提取失败"
        
        try:
            # 方法1
            good_rate = response.css('div#comment div.percent em::text').get()
            if good_rate and good_rate.strip():
                item['good_rate'] = good_rate.strip()
            else:
                # 方法2
                rate_match = re.search(r'"goodRate":([0-9.]+)', response.text)
                if rate_match:
                    item['good_rate'] = rate_match.group(1) + '%'
                else:
                    item['good_rate'] = "无法获取好评率"
        except Exception as e:
            logger.error(f"提取好评率失败: {e}")
            item['good_rate'] = "提取失败"
        
        logger.info(f"成功提取商品信息: {item['title']}")
        yield item

# 如果直接运行此文件，则使用默认配置爬取
if __name__ == "__main__":
    from jd_pipeline import JDProductPipeline
    
    # 目标URL
    target_url = SPIDER_CONFIG['TARGET_URL']
    
    try:
        # 配置Scrapy设置
        settings = {
            'USER_AGENT': SPIDER_CONFIG['BROWSER']['USER_AGENT'],
            'ROBOTSTXT_OBEY': False,
            'DOWNLOAD_DELAY': SPIDER_CONFIG['CRAWLER']['DOWNLOAD_DELAY'],
            'COOKIES_ENABLED': True,
            'DOWNLOADER_MIDDLEWARES': {
                '京东商城.JDSeleniumMiddleware': 543,
            },
            'ITEM_PIPELINES': {
                'jd_pipeline.JDProductPipeline': 300,
            },
            'FEED_EXPORT_ENCODING': 'utf-8',
            'LOG_LEVEL': 'INFO',
        }
        
        # 运行爬虫
        logger.info(f"开始爬取: {target_url}")
        process = CrawlerProcess(settings)
        process.crawl(JDProductSpider, url=target_url)
        process.start()
    except Exception as e:
        logger.error(f"爬虫运行失败: {e}")
