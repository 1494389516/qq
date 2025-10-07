// 第九关 - 获取 libfrida.so 基地址（高级版）
// 适配 Frida 17.x

console.log("[★] 第九关：获取 Native 库基地址 - 高级版");
console.log("=".repeat(70) + "\n");

// ═════════════════════════════════════════════════════════════
// 方法 1: 使用 Process API
// ═════════════════════════════════════════════════════════════

console.log("[方法 1] 使用 Process.enumerateModules()");
console.log("─".repeat(70));

var modules = Process.enumerateModules();
console.log("总模块数: " + modules.length + "\n");

var libfridaModule = null;

modules.forEach(function(module) {
    if (module.name === "libfrida.so") {
        libfridaModule = module;
        
        console.log("[✓] 找到 libfrida.so");
        console.log("  名称: " + module.name);
        console.log("  基地址: " + module.base);
        console.log("  大小: " + module.size + " 字节 (" + (module.size / 1024).toFixed(2) + " KB)");
        console.log("  路径: " + module.path);
        console.log("");
    }
});

// ═════════════════════════════════════════════════════════════
// 方法 2: 使用 Module API
// ═════════════════════════════════════════════════════════════

console.log("[方法 2] 使用 Module.findBaseAddress()");
console.log("─".repeat(70));

var baseAddr = Module.findBaseAddress("libfrida.so");
if (baseAddr) {
    console.log("[✓] 基地址: " + baseAddr);
    console.log("  十六进制: 0x" + baseAddr.toString(16));
    console.log("");
}

// ═════════════════════════════════════════════════════════════
// 方法 3: 读取 /proc/self/maps
// ═════════════════════════════════════════════════════════════

console.log("[方法 3] 读取 /proc/self/maps");
console.log("─".repeat(70));

try {
    var File = Java.use("java.io.File");
    var FileReader = Java.use("java.io.FileReader");
    var BufferedReader = Java.use("java.io.BufferedReader");
    
    var file = File.$new("/proc/self/maps");
    var reader = BufferedReader.$new(FileReader.$new(file));
    
    var line;
    var found = false;
    var count = 0;
    
    while ((line = reader.readLine()) != null && count < 5) {
        if (line.indexOf("libfrida.so") !== -1) {
            if (!found) {
                console.log("[✓] 从 /proc/self/maps 找到:");
                found = true;
            }
            console.log("  " + line);
            count++;
        }
    }
    
    reader.close();
    console.log("");
    
} catch (e) {
    console.log("[-] 读取 maps 失败: " + e);
}

// ═════════════════════════════════════════════════════════════
// 方法 4: 枚举导出函数
// ═════════════════════════════════════════════════════════════

console.log("[方法 4] 枚举 libfrida.so 的导出函数");
console.log("─".repeat(70));

if (libfridaModule) {
    var exports = Module.enumerateExports("libfrida.so");
    console.log("总导出函数: " + exports.length);
    console.log("\n前 10 个导出函数:");
    
    for (var i = 0; i < Math.min(10, exports.length); i++) {
        var exp = exports[i];
        var offset = exp.address.sub(baseAddr);
        console.log("  [" + (i + 1) + "] " + exp.name);
        console.log("      地址: " + exp.address);
        console.log("      偏移: 0x" + offset.toString(16));
        console.log("      类型: " + exp.type);
        console.log("");
    }
}

// ═════════════════════════════════════════════════════════════
// 方法 5: 枚举内存范围
// ═════════════════════════════════════════════════════════════

console.log("[方法 5] 枚举 libfrida.so 的内存范围");
console.log("─".repeat(70));

if (libfridaModule) {
    var ranges = Process.enumerateRanges({
        protection: 'r--',
        coalesce: false
    });
    
    var libfridaRanges = ranges.filter(function(range) {
        return range.file && range.file.path && range.file.path.indexOf("libfrida.so") !== -1;
    });
    
    console.log("libfrida.so 内存段数量: " + libfridaRanges.length);
    console.log("\n内存段详情:");
    
    libfridaRanges.forEach(function(range, idx) {
        console.log("  [" + (idx + 1) + "] " + range.base + " - " + range.base.add(range.size));
        console.log("      大小: " + range.size + " 字节");
        console.log("      保护: " + range.protection);
        console.log("");
    });
}

// ═════════════════════════════════════════════════════════════
// Hook Java 层验证方法
// ═════════════════════════════════════════════════════════════

console.log("[方法 6] Hook MainActivity.getLibFridaBaseAddress()");
console.log("─".repeat(70) + "\n");

Java.perform(function() {
    try {
        var MainActivity = Java.use("cn.binary.frida.MainActivity");
        
        MainActivity.getLibFridaBaseAddress.implementation = function() {
            console.log("[getLibFridaBaseAddress 被调用]");
            
            var result = this.getLibFridaBaseAddress();
            
            console.log("  Java 层返回值: 0x" + result.toString(16));
            console.log("  Frida 获取的基地址: " + baseAddr);
            
            if (result.toString(16) === baseAddr.toString(16).replace("0x", "")) {
                console.log("  ✓ 地址匹配！\n");
            } else {
                console.log("  ⚠️  地址不匹配\n");
            }
            
            return result;
        };
        
        console.log("[✓] Hook 完成\n");
        
    } catch (e) {
        console.log("[-] Hook 失败: " + e + "\n");
    }
});

// ═════════════════════════════════════════════════════════════
// 总结
// ═════════════════════════════════════════════════════════════

console.log("═".repeat(70));
console.log("[总结]");
console.log("═".repeat(70));
console.log("✓ 方法 1: Process API - 推荐使用，最简单");
console.log("✓ 方法 2: Module API - 直接获取地址");
console.log("✓ 方法 3: /proc/maps - 系统级别查看");
console.log("✓ 方法 4: 枚举导出 - 查看所有函数");
console.log("✓ 方法 5: 内存范围 - 详细的内存布局");
console.log("✓ 方法 6: Hook验证 - 与Java层对比");
console.log("═".repeat(70) + "\n");

console.log("💡 基地址的作用:");
console.log("  • 计算函数偏移: 函数地址 - 基地址 = 偏移");
console.log("  • 定位函数: 基地址 + 偏移 = 函数地址");
console.log("  • 绕过 ASLR: 相对偏移始终不变\n");

