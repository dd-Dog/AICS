@echo off
chcp 65001 >nul
REM 快速编译运行脚本
REM 用法: run.bat <文件名> [调试模式]
REM 示例: run.bat problems\01_utra_quicksort.c
REM 示例: run.bat problems\01_utra_quicksort.c debug

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo 用法: run.bat ^<文件名^> [debug]
    echo 示例: run.bat problems\01_utra_quicksort.c
    echo 示例: run.bat problems\01_utra_quicksort.c debug
    exit /b 1
)

REM 获取完整路径，避免相对路径问题
set "SOURCE_FILE=%~f1"
set "DEBUG_MODE=%~2"

REM 检查文件是否存在
if not exist "%SOURCE_FILE%" (
    echo 错误: 文件不存在: %~1
    exit /b 1
)

REM 检查编译器
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    where g++ >nul 2>&1
    if %errorlevel% neq 0 (
        echo 错误: 未找到 gcc 或 g++ 编译器
        echo 请安装 MinGW 或将其添加到 PATH
        exit /b 1
    )
    set "COMPILER=g++"
) else (
    REM 根据文件扩展名选择编译器
    if /i "%SOURCE_FILE:~-2%"==".c" (
        set "COMPILER=gcc"
    ) else (
        set "COMPILER=g++"
    )
)

REM 获取文件名（不含扩展名），使用完整路径避免中文问题
for %%f in ("%SOURCE_FILE%") do set "OUTPUT_NAME=%%~nf"

REM 创建输出目录
if not exist "build\bin" mkdir build\bin

REM 设置编译选项
if /i "%DEBUG_MODE%"=="debug" (
    echo [调试模式] 编译 %~1...
    set "FLAGS=-Wall -Wextra -g -O0 -I./utils"
    set "OUTPUT=build\bin\%OUTPUT_NAME%_debug.exe"
) else (
    echo [发布模式] 编译 %~1...
    set "FLAGS=-Wall -Wextra -O2 -I./utils"
    set "OUTPUT=build\bin\%OUTPUT_NAME%.exe"
)

REM 如果是C++文件，添加C++标准
if /i not "%SOURCE_FILE:~-2%"==".c" (
    set "FLAGS=-std=c++17 %FLAGS%"
) else (
    REM C文件在Windows上需要链接psapi库用于内存统计
    set "FLAGS=%FLAGS% -lpsapi"
)

REM 检查源文件是否使用了utils工具类，如果使用了则自动链接
REM 简化方案：如果源文件是C++且utils文件存在，就尝试链接（编译时会自动处理未使用的链接）
set "UTILS_FILES="
if exist "utils\mnist_loader.cpp" (
    if /i not "%SOURCE_FILE:~-2%"==".c" (
        REM C++文件自动链接utils工具类（如果编译报错说明没用到，可以忽略）
        set "UTILS_FILES=utils\mnist_loader.cpp"
        echo [自动链接utils工具类] !UTILS_FILES!
    )
)

REM 编译（如果使用了utils，一起编译链接）
if defined UTILS_FILES (
    %COMPILER% %FLAGS% -o "%OUTPUT%" "%SOURCE_FILE%" %UTILS_FILES%
) else (
    %COMPILER% %FLAGS% -o "%OUTPUT%" "%SOURCE_FILE%"
)

if %errorlevel% neq 0 (
    echo 编译失败！
    exit /b 1
)

echo 编译成功: %OUTPUT%
echo.
echo ========== 运行程序 ==========
echo.

REM 运行程序（使用延迟扩展变量避免中文路径问题）
setlocal disabledelayedexpansion
"%OUTPUT%"
set RUN_EXIT_CODE=%errorlevel%
endlocal

echo.
echo ========== 程序结束 ==========
echo 退出代码: %RUN_EXIT_CODE%
exit /b %RUN_EXIT_CODE%
