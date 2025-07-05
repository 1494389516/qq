# 京东商城爬虫打包说明

本文档详细说明如何将京东商城爬虫项目打包成独立的可执行文件（.exe）。

## 准备工作

### 1. 安装必要的依赖

首先，确保已安装所有必要的依赖：

```bash
# 安装项目依赖
pip install -r requirements.txt

# 安装PyInstaller
pip install pyinstaller
```

### 2. 检查项目结构

确保项目结构完整，包含以下关键文件：

- `run_jd_spider.py`：主入口文件
- `jd_main.py`：爬虫主逻辑
- `jd_login.py`：登录处理模块
- `jd_pipeline.py`：数据处理管道
- `config.py`：配置文件
- `京东商城.py`：验证处理模块
- `chromedriver.exe`：Chrome驱动

## 打包方法

### 方法一：使用自动化脚本（推荐）

项目提供了自动化打包脚本，可以一键完成打包过程。

#### Windows系统

双击运行`build.bat`文件，或在命令行中执行：

```cmd
build.bat
```

#### Linux/macOS系统

在终端中执行：

```bash
chmod +x build.sh
./build.sh
```

### 方法二：手动打包

如果自动化脚本不适用于您的环境，可以按照以下步骤手动打包：

1. **生成图标**（可选）

   ```bash
   python jd_icon.py
   ```

2. **清理旧的构建文件**

   ```bash
   # Windows
   rmdir /s /q build dist
   del jd_spider.spec

   # Linux/macOS
   rm -rf build dist
   rm -f jd_spider.spec
   ```

3. **执行PyInstaller打包**

   ```bash
   pyinstaller --name=jd_spider --onefile --windowed --icon=jd_icon.ico ^
   --add-data="config.py;." --add-data="README.md;." --add-data="requirements.txt;." ^
   --add-data="example_urls.txt;." --add-data="UPDATE_LOG.md;." --add-data="LOGIN_GUIDE.md;." ^
   --add-data="cookies;cookies" --add-data="logs;logs" --add-data="results;results" ^
   --hidden-import=PIL._tkinter_finder --hidden-import=colorama --hidden-import=selenium ^
   --hidden-import=undetected_chromedriver --hidden-import=tqdm --hidden-import=fake_useragent ^
   run_jd_spider.py
   ```

   > 注意：Linux/macOS系统中，路径分隔符应使用`:`而非`;`

4. **复制ChromeDriver**

   ```bash
   # Windows
   copy chromedriver.exe dist\chromedriver.exe

   # Linux/macOS
   cp chromedriver.exe dist/chromedriver.exe
   ```

5. **创建辅助文件**（可选）

   ```bash
   # 创建使用说明
   copy README_EXE.md dist\使用说明.txt

   # 创建批处理启动文件
   # 请参考build_exe.py中的create_batch_file函数
   ```

## 打包配置说明

### PyInstaller参数说明

- `--name=jd_spider`：指定生成的可执行文件名称
- `--onefile`：将所有依赖打包成单个可执行文件
- `--windowed`：不显示控制台窗口（Windows系统）
- `--icon=jd_icon.ico`：指定应用图标
- `--add-data`：添加数据文件
- `--hidden-import`：添加隐式导入的模块

### 特殊处理说明

1. **处理undetected_chromedriver**

   undetected_chromedriver模块需要特殊处理，确保正确导入：

   ```python
   # pyinstaller_hooks.py中的处理方法
   def hook_undetected_chromedriver(hook_api):
       hook_api.add_imports('undetected_chromedriver')
       hook_api.add_imports('undetected_chromedriver.patcher')
       hook_api.add_imports('undetected_chromedriver.options')
       hook_api.add_imports('undetected_chromedriver.cdp')
       hook_api.add_imports('undetected_chromedriver.v2')
   ```

2. **处理fake_useragent**

   fake_useragent模块需要包含其数据文件：

   ```python
   # pyinstaller_hooks.py中的处理方法
   def hook_fake_useragent(hook_api):
       hook_api.add_imports('fake_useragent')
       hook_api.add_imports('fake_useragent.utils')
       hook_api.add_imports('fake_useragent.settings')
       datas = collect_data_files('fake_useragent')
       hook_api.add_datas(datas)
   ```

## 常见问题

### 1. 找不到模块错误

如果打包后运行时提示找不到某个模块，请确保在PyInstaller命令中添加了相应的`--hidden-import`参数。

### 2. 资源文件缺失

如果程序运行时找不到某些资源文件，请确保在PyInstaller命令中使用`--add-data`参数正确包含了这些文件。

### 3. ChromeDriver问题

确保打包后的目录中包含与Chrome浏览器版本匹配的ChromeDriver。

### 4. Windows Defender警告

由于PyInstaller打包的程序没有数字签名，可能会被Windows Defender标记为可疑文件。这是正常现象，可以通过点击"更多信息"和"仍要运行"来运行程序。

## 发布说明

打包完成后，建议将以下文件一起发布：

1. `jd_spider.exe`：主程序
2. `chromedriver.exe`：Chrome驱动
3. `启动爬虫.bat`：批处理启动文件
4. `使用说明.txt`：简易使用说明
5. `README.md`：详细文档
6. `LOGIN_GUIDE.md`：登录指南
7. `example_urls.txt`：示例URL文件

可以将这些文件打包成ZIP文件进行发布，文件名格式建议为：`jd_spider_v版本号.zip` 