// 第十关 - hexdump 打印内存（高级版）
// 适配 Frida 17.x

console.log("[★] 第十关：hexdump 打印内存 - 高级版");
console.log("=".repeat(70) + "\n");

// ═════════════════════════════════════════════════════════════
// 步骤 1: 获取基地址
// ═════════════════════════════════════════════════════════════

console.log("[步骤 1] 获取 libc.so 基地址");
console.log("─".repeat(70));

var base = Module.findBaseAddress("libc.so");
if (!base) {
    console.log("[-] 未找到 libc.so");
} else {
    console.log("[✓] libc.so 基地址: " + base);
    console.log("    十六进制: 0x" + base.toString(16));
    
    var module = Process.getModuleByName("libc.so");
    console.log("    大小: " + module.size + " 字节");
    console.log("    路径: " + module.path);
    console.log("");
    
    // ═════════════════════════════════════════════════════════════
    // 步骤 2: 计算目标地址
    // ═════════════════════════════════════════════════════════════
    
    console.log("[步骤 2] 计算目标地址");
    console.log("─".repeat(70));
    
    var offset = 0x200;
    var target = base.add(offset);
    
    console.log("  基地址: " + base);
    console.log("  偏移量: 0x" + offset.toString(16) + " (" + offset + " 字节)");
    console.log("  目标地址: " + target);
    console.log("  目标地址(hex): 0x" + target.toString(16));
    console.log("");
    
    // ═════════════════════════════════════════════════════════════
    // 步骤 3: 多种 hexdump 方式
    // ═════════════════════════════════════════════════════════════
    
    console.log("[步骤 3] Hexdump 输出");
    console.log("═".repeat(70));
    
    // 方式 1: 标准 hexdump
    console.log("\n[方式 1] 标准 hexdump (128 字节)");
    console.log("─".repeat(70));
    console.log(hexdump(target, {
        length: 128,
        header: true,
        ansi: true,
        offset: 0
    }));
    
    // 方式 2: 简洁模式
    console.log("\n[方式 2] 简洁模式 (64 字节)");
    console.log("─".repeat(70));
    console.log(hexdump(target, {
        length: 64,
        header: false,
        ansi: false
    }));
    
    // 方式 3: 查找特定模式
    console.log("\n[方式 3] 分析内存内容");
    console.log("─".repeat(70));
    
    try {
        // 读取前 4 个字节检查 ELF 头
        var magic = target.readU32();
        console.log("前 4 字节: 0x" + magic.toString(16));
        
        if (magic === 0x464c457f) {
            console.log("  ✓ 检测到 ELF 文件头 (0x7f 'E' 'L' 'F')");
        } else {
            console.log("  ℹ️  不是 ELF 头");
        }
        
        // 读取字符串
        try {
            var str = target.readCString(32);
            if (str && str.length > 0) {
                console.log("  字符串内容: \"" + str + "\"");
            }
        } catch (e) {}
        
    } catch (e) {
        console.log("  读取失败: " + e.message);
    }
    
    // ═════════════════════════════════════════════════════════════
    // 步骤 4: 扫描更多位置
    // ═════════════════════════════════════════════════════════════
    
    console.log("\n[步骤 4] 扫描多个偏移位置");
    console.log("═".repeat(70));
    
    var offsets = [0x0, 0x100, 0x200, 0x400, 0x800];
    
    offsets.forEach(function(off) {
        var addr = base.add(off);
        console.log("\n[偏移 0x" + off.toString(16) + "] 地址: " + addr);
        console.log("─".repeat(70));
        console.log(hexdump(addr, {
            length: 32,
            header: false,
            ansi: true
        }));
    });
    
    // ═════════════════════════════════════════════════════════════
    // 步骤 5: 对比不同库
    // ═════════════════════════════════════════════════════════════
    
    console.log("\n[步骤 5] 对比其他库的内存");
    console.log("═".repeat(70));
    
    var otherLibs = ["libfrida.so", "libc++.so", "libm.so"];
    
    otherLibs.forEach(function(libName) {
        try {
            var libBase = Module.findBaseAddress(libName);
            if (libBase) {
                var libTarget = libBase.add(0x200);
                console.log("\n[" + libName + " + 0x200]");
                console.log("  基地址: " + libBase);
                console.log("  目标: " + libTarget);
                console.log("─".repeat(70));
                console.log(hexdump(libTarget, {
                    length: 32,
                    header: false,
                    ansi: true
                }));
            }
        } catch (e) {}
    });
    
    // ═════════════════════════════════════════════════════════════
    // 总结
    // ═════════════════════════════════════════════════════════════
    
    console.log("\n" + "═".repeat(70));
    console.log("[hexdump 使用技巧]");
    console.log("═".repeat(70));
    console.log("• length: 控制显示字节数");
    console.log("• header: 显示地址头");
    console.log("• ansi: 使用颜色高亮");
    console.log("• offset: 自定义偏移显示");
    console.log("");
    console.log("常见用途:");
    console.log("  ✓ 查看内存数据");
    console.log("  ✓ 检查文件头");
    console.log("  ✓ 查找字符串");
    console.log("  ✓ 分析数据结构");
    console.log("  ✓ 调试内存问题");
    console.log("═".repeat(70) + "\n");
}

