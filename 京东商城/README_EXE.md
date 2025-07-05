# 京东商城爬虫 (EXE版本)

这是京东商城爬虫的独立可执行文件版本，无需安装Python环境即可运行。

## 安装说明

1. 从发布页面下载最新的`jd_spider_vX.X.X.zip`文件
2. 解压缩到任意文件夹
3. 确保`chromedriver.exe`与`jd_spider.exe`在同一目录下
4. 双击`启动爬虫.bat`或直接运行`jd_spider.exe`

## 目录结构

```
jd_spider/
  ├── jd_spider.exe        # 主程序
  ├── chromedriver.exe     # Chrome驱动
  ├── 启动爬虫.bat          # 批处理启动文件
  ├── 使用说明.txt          # 简易使用说明
  ├── README.md            # 详细文档
  ├── LOGIN_GUIDE.md       # 登录指南
  ├── example_urls.txt     # 示例URL文件
  ├── cookies/             # 保存登录状态的cookies
  ├── logs/                # 日志目录
  └── results/             # 结果保存目录
```

## 使用方法

### 图形界面启动

1. 双击`启动爬虫.bat`
2. 根据菜单提示选择操作：
   - 爬取单个商品
   - 批量爬取商品
   - 查看使用说明

### 命令行启动

程序支持与Python版本相同的命令行参数：

```
jd_spider.exe [参数]
```

#### 常用参数

- `--url URL`: 指定要爬取的商品URL
- `--output DIR`: 指定结果保存目录
- `--login`: 启用登录功能（默认启用）
- `--no-login`: 禁用登录功能（访客模式）
- `--headless`: 使用无头浏览器模式（不显示浏览器窗口）
- `--debug`: 启用调试模式

#### 批量爬取参数

- `--batch`: 启用批量爬取模式
- `--url-file FILE`: 包含多个URL的文件路径（每行一个URL）
- `--max-workers N`: 批量爬取时的最大线程数（默认为1）
- `--delay N`: 批量爬取时每个URL之间的延迟（秒）

### 示例

```
# 爬取单个商品
jd_spider.exe --url https://item.jd.com/100050401004.html

# 批量爬取
jd_spider.exe --batch --url-file example_urls.txt --max-workers 2 --delay 10

# 不登录爬取
jd_spider.exe --url https://item.jd.com/100050401004.html --no-login
```

## 登录说明

程序支持两种登录方式：

1. **扫码登录（推荐）**：程序会显示二维码，使用京东APP扫码登录
2. **账号密码登录**：使用`--username`和`--password`参数指定账号密码

登录成功后，程序会保存Cookie到`cookies`目录，下次运行时会自动使用保存的登录状态。

详细登录说明请参考`LOGIN_GUIDE.md`文件。

## 常见问题

### 1. 启动时提示"Windows已保护你的电脑"

这是因为程序是自打包的，没有数字签名。点击"更多信息"，然后点击"仍要运行"即可。

### 2. 提示找不到chromedriver.exe

请确保`chromedriver.exe`与`jd_spider.exe`在同一目录下。

### 3. 提示"无法启动此程序，因为计算机中丢失VCRUNTIME140.dll"

需要安装Visual C++ Redistributable for Visual Studio 2015-2022：
- [下载链接](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### 4. 登录后没有返回商品页面

这是一个已知问题，可以尝试重新运行程序，或者手动复制商品URL重新访问。

### 5. 无法正常显示二维码

如果二维码显示异常，可以查看程序目录下的`jd_qrcode.png`文件。

## 注意事项

1. 请确保您的Chrome浏览器已安装且版本与chromedriver匹配
2. 爬取频率过高可能导致IP被封，建议设置适当的延迟时间
3. 程序运行日志将保存在`logs`目录下，方便排查问题
4. 爬取结果将保存在`results`目录下，包括JSON和CSV格式

## 更新日志

详细更新记录请查看`UPDATE_LOG.md`文件。

## 免责声明

本程序仅供学习研究使用，请勿用于商业用途。使用本程序产生的任何法律责任由使用者自行承担。 