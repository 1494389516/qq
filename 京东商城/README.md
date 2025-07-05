# 京东商城爬虫

一个功能强大的京东商品数据爬虫工具，支持自动登录、批量爬取、数据可视化及打包为独立可执行程序。

## ⚠️ 特别声明

**请注意：京东对爬虫有严格的检测和防护机制，随时可能更新。**

- 本工具可能会因京东网站更新而失效，不保证长期有效
- 频繁、大量爬取可能导致IP被封禁或账号风控
- 建议合理设置爬取间隔和批量数量，避免集中爬取
- 爬取行为应遵守京东用户协议和相关法律法规
- 本工具仅供学习研究使用，请勿用于商业用途
- 使用本工具产生的任何法律责任由使用者自行承担

如遇工具失效，请关注本项目更新或提交Issue反馈。

## 功能特点

- **自动登录**：支持扫码登录和账号密码登录，自动保存和复用登录状态
- **批量爬取**：支持从文件批量读取URL进行爬取，可配置线程数和延迟时间
- **数据提取**：提取商品标题、价格、规格、评价等完整信息
- **进度显示**：实时显示爬取和数据提取进度
- **数据处理**：自动验证和清洗数据，确保数据质量
- **多格式导出**：支持JSON和CSV格式保存数据
- **错误处理**：智能重试机制和详细的错误日志
- **防检测**：增强的浏览器指纹模拟和反爬机制
- **可执行程序**：支持打包为独立的EXE文件，无需Python环境即可运行

## 安装说明

### 环境要求

- Python 3.8+
- Chrome浏览器
- ChromeDriver（与Chrome版本匹配）

### 安装步骤

1. 克隆项目到本地

```bash
git clone https://github.com/yourusername/jd-spider.git
cd jd-spider
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 下载ChromeDriver

确保下载的ChromeDriver版本与您的Chrome浏览器版本匹配，并将其放置在项目根目录下。

## 使用说明

### 基本用法

```bash
python run_jd_spider.py --url https://item.jd.com/100050401004.html
```

### 命令行参数

- `--url`: 指定要爬取的商品URL
- `--output`: 指定结果保存目录
- `--headless`: 使用无头浏览器模式（不显示浏览器窗口）
- `--login`: 启用登录功能（默认启用）
- `--no-login`: 禁用登录功能（访客模式）
- `--username`: 指定京东账号
- `--password`: 指定京东密码
- `--debug`: 启用调试模式
- `--show-workflow`: 显示爬虫工作流程

### 批量爬取

```bash
# 从文件批量爬取
python run_jd_spider.py --batch --url-file urls.txt --output results/phones

# 指定线程数和延迟
python run_jd_spider.py --batch --url-file urls.txt --max-workers 2 --delay 10
```

### 批量爬取参数

- `--batch`: 启用批量爬取模式
- `--url-file`: 包含多个URL的文件路径（每行一个URL）
- `--max-workers`: 批量爬取时的最大线程数（默认为1，建议不超过3）
- `--delay`: 批量爬取时每个URL之间的延迟（秒）
- `--category`: 爬取的商品分类（用于结果保存）
- `--retry`: 失败时重试次数

## 打包为可执行文件

本项目支持打包为独立的可执行文件（.exe），无需安装Python环境即可运行。

### Windows系统打包

```bash
# 一键打包
build.bat
```

### Linux/macOS系统打包

```bash
# 添加执行权限
chmod +x build.sh

# 执行打包脚本
./build.sh
```

### 手动打包

```bash
# 安装PyInstaller
pip install pyinstaller

# 执行打包
python build_exe.py
```

### 打包后的文件

打包完成后，可执行文件将位于`dist`目录下：

- `jd_spider.exe`: 主程序
- `chromedriver.exe`: Chrome驱动
- `启动爬虫.bat`: 批处理启动文件
- `使用说明.txt`: 简易使用说明

详细的打包说明请参考[PACKAGING.md](PACKAGING.md)文件。

## 项目结构

```
京东商城/
  ├── chromedriver.exe        # Chrome驱动
  ├── config.py               # 配置文件
  ├── cookies/                # 保存登录状态的cookies
  ├── crawl_jd_categories.py  # 分类爬取脚本
  ├── crawl_jd_phones.py      # 手机爬取示例脚本
  ├── jd_data_visualize.py    # 数据可视化工具
  ├── jd_login.py             # 登录处理模块
  ├── jd_main.py              # 爬虫主要逻辑
  ├── jd_pipeline.py          # 数据处理管道
  ├── logs/                   # 日志目录
  ├── requirements.txt        # 项目依赖
  ├── results/                # 结果保存目录
  ├── run_jd_spider.py        # 爬虫运行脚本
  ├── build_exe.py            # EXE打包脚本
  ├── pyinstaller_hooks.py    # PyInstaller钩子脚本
  ├── jd_icon.py              # 图标生成脚本
  ├── build.bat               # Windows打包批处理
  ├── build.sh                # Linux/macOS打包脚本
  ├── PACKAGING.md            # 打包说明文档
  ├── README_EXE.md           # EXE版本使用说明
  ├── test_login_crawler.py   # 登录测试脚本
  ├── test_url_redirect.py    # URL重定向测试脚本
  └── 京东商城.py              # 验证处理模块
```

## 配置说明

配置文件`config.py`包含多种可自定义的设置：

### 浏览器配置

```python
'BROWSER': {
    'HEADLESS': False,  # 是否使用无头模式
    'WINDOW_SIZE': (1920, 1080),  # 窗口大小
    'PAGE_LOAD_TIMEOUT': 30,  # 页面加载超时时间（秒）
    'IMPLICIT_WAIT': 10,  # 隐式等待时间（秒）
    'STEALTH_MODE': True,  # 是否使用隐身模式
}
```

### 登录配置

```python
'LOGIN': {
    'ENABLE': True,  # 是否启用登录
    'USERNAME': '',  # 京东账号
    'PASSWORD': '',  # 京东密码
    'COOKIES_DIR': 'cookies',  # cookies保存目录
    'QRCODE_TIMEOUT': 60,  # 二维码超时时间（秒）
    'QRCODE_REFRESH': True,  # 是否自动刷新二维码
    'MAX_QRCODE_REFRESH': 3,  # 最大二维码刷新次数
    'AUTO_OPEN_QRCODE': True,  # 是否自动打开二维码图片
}
```

### 输出配置

```python
'OUTPUT': {
    'RESULTS_DIR': 'results/huawei',  # 结果保存目录
    'SAVE_IMAGES': False,  # 是否保存商品图片
    'DEBUG_DIR': 'results/debug',  # 调试信息保存目录
}
```

## 数据格式

爬取的商品数据将保存为JSON和CSV格式，包含以下字段：

- `product_id`: 商品ID
- `title`: 商品标题
- `price`: 商品价格
- `original_price`: 原价
- `discount`: 折扣
- `shop_name`: 店铺名称
- `shop_score`: 店铺评分
- `brand`: 品牌
- `model`: 型号
- `category`: 分类
- `description`: 描述
- `specs`: 规格参数
- `images`: 图片URL列表
- `comments_count`: 评价数量
- `good_rate`: 好评率
- `promotion`: 促销信息
- `stock_status`: 库存状态
- `url`: 商品URL
- `crawl_time`: 爬取时间

## 登录说明

程序支持两种登录方式：

1. **扫码登录（推荐）**：程序会显示二维码，使用京东APP扫码登录
2. **账号密码登录**：使用`--username`和`--password`参数指定账号密码

登录成功后，程序会保存Cookie到`cookies`目录，下次运行时会自动使用保存的登录状态。

详细登录说明请参考[LOGIN_GUIDE.md](LOGIN_GUIDE.md)文件。

## 数据可视化

项目包含数据可视化功能，可以生成多种图表：

- 价格分布直方图
- 品牌商品数量对比
- 价格区间分析
- 评价与价格关系图

使用方法：

```bash
python jd_data_visualize.py --input results/phones --output analysis/phones
```

## 注意事项

1. 首次运行会要求扫码登录
2. 登录成功后会保存Cookie，下次运行可自动使用
3. 如需强制重新登录，可删除cookies目录下的对应文件
4. 批量爬取时建议设置适当的延迟，避免被封IP
5. 爬取结果将保存在results目录下，可通过--output参数自定义
6. 打包为EXE后，请确保ChromeDriver与Chrome浏览器版本匹配

## 常见问题

### 1. 登录问题

如果遇到登录相关问题，请参考[LOGIN_GUIDE.md](LOGIN_GUIDE.md)中的常见问题解决方案。

### 2. 打包问题

如果遇到打包相关问题，请参考[PACKAGING.md](PACKAGING.md)中的常见问题解决方案。

### 3. 爬取失败

- 检查网络连接是否正常
- 确认URL是否有效
- 检查是否需要登录才能查看完整信息
- 尝试增加延迟时间和重试次数

### 4. ChromeDriver问题

- 确保ChromeDriver版本与Chrome浏览器版本匹配
- 将ChromeDriver放置在项目根目录下
- 检查ChromeDriver是否有执行权限

## 更新日志

详细更新记录请查看[UPDATE_LOG.md](UPDATE_LOG.md)

## 贡献指南

欢迎贡献代码或提出建议：

1. Fork 本仓库
2. 创建新分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建 Pull Request

## 许可证

MIT License

## 免责声明

本项目仅供学习研究使用，请勿用于商业用途。使用本程序产生的任何法律责任由使用者自行承担。

## 反爬应对策略

京东的反爬机制非常严格，包括但不限于以下措施：

1. **IP频率限制**：短时间内大量请求会触发封IP
   - 应对：设置随机延迟，使用代理IP池

2. **用户行为分析**：检测非正常浏览行为
   - 应对：模拟真实用户行为，随机停顿，自然滚动

3. **浏览器指纹识别**：检测自动化工具特征
   - 应对：使用undetected_chromedriver，隐藏WebDriver特征

4. **验证码与滑块验证**：频繁访问触发验证
   - 应对：本工具提供自动或手动处理机制

5. **页面结构变化**：经常更改页面元素和结构
   - 应对：定期更新选择器，使用多种定位策略

如果您发现工具失效，请检查以下可能的原因：
- 京东页面结构已更新，需要更新选择器
- 反爬策略升级，需要调整防检测方法
- 登录机制变更，需要更新登录流程
- ChromeDriver版本与Chrome不匹配

建议定期关注本项目更新，或根据自身需求修改代码适应最新变化。 