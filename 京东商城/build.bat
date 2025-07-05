@echo off
echo ========================================
echo      京东商城爬虫 - EXE打包工具
echo ========================================
echo.

REM 检查Python环境
echo 检查Python环境...
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

REM 检查是否已安装依赖
echo 检查依赖...
python -c "import PyInstaller" > nul 2>&1
if %errorlevel% neq 0 (
    echo 安装PyInstaller...
    pip install pyinstaller
    if %errorlevel% neq 0 (
        echo 错误: 安装PyInstaller失败
        pause
        exit /b 1
    )
)

REM 检查requirements.txt
if exist requirements.txt (
    echo 安装项目依赖...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 警告: 安装部分依赖失败，但将继续构建
    )
)

REM 生成图标
echo 生成图标...
python jd_icon.py
if %errorlevel% neq 0 (
    echo 警告: 生成图标失败，将使用默认图标
)

REM 清理旧的构建文件
echo 清理旧的构建文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist jd_spider.spec del jd_spider.spec

REM 构建EXE
echo.
echo 开始构建可执行文件...
python build_exe.py
if %errorlevel% neq 0 (
    echo 错误: 构建失败
    pause
    exit /b 1
)

REM 检查构建结果
if not exist dist\jd_spider.exe (
    echo 错误: 构建过程完成，但未找到生成的EXE文件
    pause
    exit /b 1
)

REM 创建发布包
echo.
echo 创建发布包...
cd dist
set VERSION=1.4.0
for /f "tokens=3" %%i in ('findstr /C:"v" ..\UPDATE_LOG.md ^| findstr /C:"##" ^| findstr /B /C:"## " ^| head -1') do set VERSION=%%i
set VERSION=%VERSION:v=%
set ZIPFILE=jd_spider_v%VERSION%.zip

REM 创建ZIP文件
echo 打包文件到 %ZIPFILE% ...
powershell Compress-Archive -Path * -DestinationPath %ZIPFILE% -Force
if %errorlevel% neq 0 (
    echo 警告: 创建ZIP文件失败
) else (
    echo 成功创建发布包: %ZIPFILE%
)

cd ..

echo.
echo ========================================
echo 构建完成!
echo.
echo 可执行文件位置: %CD%\dist\jd_spider.exe
echo 发布包位置: %CD%\dist\%ZIPFILE%
echo ========================================
echo.

pause 