// 第十一关 - Hook open 函数（高级版）
// 适配 Frida 17.x

console.log("[★] 第十一关：Hook open() 函数 - 高级版");
console.log("=".repeat(70) + "\n");

// ═════════════════════════════════════════════════════════════
// 配置
// ═════════════════════════════════════════════════════════════

var config = {
    blockKeywords: ["hack", "frida", "xposed"],  // 阻止包含这些关键字的文件
    logAll: true,                                 // 记录所有 open 调用
    showFlags: true,                              // 显示打开标志
    showBacktrace: false                          // 显示调用栈
};

var stats = {
    totalCalls: 0,
    blocked: 0,
    allowed: 0
};

// ═════════════════════════════════════════════════════════════
// Hook open() 函数
// ═════════════════════════════════════════════════════════════

console.log("[模块 1] Hook libc.so 的 open() 函数");
console.log("─".repeat(70) + "\n");

var openPtr = Module.findExportByName("libc.so", "open");

if (!openPtr) {
    console.log("[-] 未找到 open 函数");
} else {
    console.log("[✓] open 函数地址: " + openPtr);
    console.log("");
    
    Interceptor.attach(openPtr, {
        onEnter: function(args) {
            stats.totalCalls++;
            
            // 读取文件路径
            var pathname = args[0].readCString();
            var flags = args[1].toInt32();
            
            this.pathname = pathname;
            this.flags = flags;
            this.blocked = false;
            
            // 检查是否需要阻止
            var shouldBlock = false;
            var matchedKeyword = "";
            
            config.blockKeywords.forEach(function(keyword) {
                if (pathname && pathname.toLowerCase().indexOf(keyword) !== -1) {
                    shouldBlock = true;
                    matchedKeyword = keyword;
                }
            });
            
            if (shouldBlock) {
                this.blocked = true;
                stats.blocked++;
                
                console.log("\n🚫 [阻止] open() 调用");
                console.log("─".repeat(70));
                console.log("  文件: " + pathname);
                console.log("  关键字: " + matchedKeyword);
                
                if (config.showFlags) {
                    console.log("  标志: 0x" + flags.toString(16));
                    this.parseFlags(flags);
                }
                
                console.log("  操作: 返回 -1 (失败)");
                console.log("─".repeat(70) + "\n");
                
            } else if (config.logAll) {
                stats.allowed++;
                
                console.log("\n✓ [允许] open() 调用");
                console.log("  文件: " + pathname);
                
                if (config.showFlags) {
                    console.log("  标志: 0x" + flags.toString(16));
                    this.parseFlags(flags);
                }
                
                console.log("");
            }
            
            // 显示调用栈（可选）
            if (config.showBacktrace && (shouldBlock || config.logAll)) {
                console.log("  [调用栈]");
                var backtrace = Thread.backtrace(this.context, Backtracer.ACCURATE);
                backtrace.slice(0, 5).forEach(function(addr, i) {
                    try {
                        var symbol = DebugSymbol.fromAddress(addr);
                        console.log("    [" + i + "] " + symbol);
                    } catch (e) {
                        console.log("    [" + i + "] " + addr);
                    }
                });
                console.log("");
            }
        },
        
        onLeave: function(retval) {
            if (this.blocked) {
                // 修改返回值为 -1 (失败)
                retval.replace(-1);
            } else if (config.logAll) {
                console.log("  返回值: " + retval);
                
                if (retval.toInt32() >= 0) {
                    console.log("  状态: ✓ 成功 (fd = " + retval + ")");
                } else {
                    console.log("  状态: ✗ 失败");
                }
                console.log("");
            }
        },
        
        // 辅助函数：解析 open 标志
        parseFlags: function(flags) {
            var O_RDONLY = 0x0000;
            var O_WRONLY = 0x0001;
            var O_RDWR = 0x0002;
            var O_CREAT = 0x0040;
            var O_EXCL = 0x0080;
            var O_TRUNC = 0x0200;
            var O_APPEND = 0x0400;
            
            var flagStr = "  模式: ";
            
            var mode = flags & 0x3;
            if (mode === O_RDONLY) flagStr += "O_RDONLY";
            else if (mode === O_WRONLY) flagStr += "O_WRONLY";
            else if (mode === O_RDWR) flagStr += "O_RDWR";
            
            if (flags & O_CREAT) flagStr += " | O_CREAT";
            if (flags & O_EXCL) flagStr += " | O_EXCL";
            if (flags & O_TRUNC) flagStr += " | O_TRUNC";
            if (flags & O_APPEND) flagStr += " | O_APPEND";
            
            console.log(flagStr);
        }
    });
    
    console.log("[✓] Hook 完成\n");
}

// ═════════════════════════════════════════════════════════════
// 同时 Hook openat() 函数
// ═════════════════════════════════════════════════════════════

console.log("[模块 2] Hook openat() 函数");
console.log("─".repeat(70) + "\n");

var openatPtr = Module.findExportByName("libc.so", "openat");

if (openatPtr) {
    console.log("[✓] openat 函数地址: " + openatPtr);
    
    Interceptor.attach(openatPtr, {
        onEnter: function(args) {
            var dirfd = args[0].toInt32();
            var pathname = args[1].readCString();
            var flags = args[2].toInt32();
            
            console.log("\n[openat] 调用");
            console.log("  dirfd: " + dirfd);
            console.log("  path: " + pathname);
            console.log("  flags: 0x" + flags.toString(16));
            console.log("");
        }
    });
    
    console.log("[✓] Hook 完成\n");
} else {
    console.log("[-] 未找到 openat 函数\n");
}

// ═════════════════════════════════════════════════════════════
// 统计信息定时显示
// ═════════════════════════════════════════════════════════════

setInterval(function() {
    if (stats.totalCalls > 0) {
        console.log("═".repeat(70));
        console.log("📊 [统计信息]");
        console.log("═".repeat(70));
        console.log("  总调用次数: " + stats.totalCalls);
        console.log("  ✓ 允许: " + stats.allowed);
        console.log("  🚫 阻止: " + stats.blocked);
        console.log("  阻止率: " + (stats.blocked / stats.totalCalls * 100).toFixed(2) + "%");
        console.log("═".repeat(70) + "\n");
    }
}, 30000);  // 每 30 秒显示一次

// ═════════════════════════════════════════════════════════════
// 说明
// ═════════════════════════════════════════════════════════════

console.log("═".repeat(70));
console.log("[配置说明]");
console.log("═".repeat(70));
console.log("阻止关键字: " + config.blockKeywords.join(", "));
console.log("记录所有调用: " + (config.logAll ? "是" : "否"));
console.log("显示标志: " + (config.showFlags ? "是" : "否"));
console.log("显示调用栈: " + (config.showBacktrace ? "是" : "否"));
console.log("═".repeat(70) + "\n");

console.log("💡 提示:");
console.log("  • 包含阻止关键字的文件将无法打开");
console.log("  • 返回 -1 表示文件打开失败");
console.log("  • 可用于绕过文件检测\n");

