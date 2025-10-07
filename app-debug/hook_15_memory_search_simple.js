// 第十五关 - 内存搜索 Flag（简洁版）
// 适配 Frida 17.x

console.log("[★] 第十五关：内存搜索 Flag\n");

Java.perform(function() {
    console.log("[*] 开始内存搜索...\n");
    
    // 方法 1: 搜索字符串 "flag"
    console.log("=".repeat(60));
    console.log("[方法 1] 搜索字符串 'flag'");
    console.log("=".repeat(60));
    
    var pattern = "66 6c 61 67";  // "flag" 的十六进制
    
    Process.enumerateRanges('r--').forEach(function(range) {
        try {
            Memory.scan(range.base, range.size, pattern, {
                onMatch: function(address, size) {
                    console.log("\n[找到匹配]");
                    console.log("  地址: " + address);
                    console.log("  范围: " + range.base + " - " + range.base.add(range.size));
                    
                    // 读取周围的内容
                    try {
                        var str = address.readCString(100);
                        if (str && str.length > 0) {
                            console.log("  内容: " + str);
                            
                            // 检查是否是 flag 格式
                            if (str.indexOf("flag{") !== -1) {
                                console.log("  🚩 发现 FLAG: " + str);
                            }
                        }
                    } catch (e) {}
                },
                onComplete: function() {}
            });
        } catch (e) {}
    });
    
    console.log("\n[✓] 搜索完成\n");
});

