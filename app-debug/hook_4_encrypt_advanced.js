// 第四关 - 监控加密过程（高级版）
// 适配 Frida 17.x

console.log("[★] 第四关：监控三重加密过程 - 高级版");
console.log("=".repeat(70) + "\n");

Java.perform(function() {
    var Utils = Java.use("cn.binary.frida.Utils");
    var encryptCount = 0;
    
    console.log("[模块 1] Hook simpleEncrypt()");
    console.log("─".repeat(70));
    
    Utils.simpleEncrypt.implementation = function(data, key) {
        encryptCount++;
        
        console.log("\n[simpleEncrypt #" + encryptCount + " 被调用]");
        console.log("─".repeat(70));
        
        // 显示输入
        console.log("输入数据:");
        if (data && data.length > 0) {
            var inputStr = "";
            for (var i = 0; i < Math.min(data.length, 32); i++) {
                inputStr += ("0" + (data[i] & 0xFF).toString(16)).slice(-2) + " ";
            }
            console.log("  Hex: " + inputStr);
            
            try {
                var str = Java.use("java.lang.String").$new(data);
                console.log("  String: " + str);
            } catch (e) {}
        }
        
        console.log("\n密钥数据:");
        if (key && key.length > 0) {
            var keyStr = "";
            for (var i = 0; i < Math.min(key.length, 32); i++) {
                keyStr += ("0" + (key[i] & 0xFF).toString(16)).slice(-2) + " ";
            }
            console.log("  Hex: " + keyStr);
        }
        
        // 执行加密
        var result = this.simpleEncrypt(data, key);
        
        // 显示输出
        console.log("\n输出数据:");
        if (result && result.length > 0) {
            var outputStr = "";
            for (var i = 0; i < Math.min(result.length, 32); i++) {
                outputStr += ("0" + (result[i] & 0xFF).toString(16)).slice(-2) + " ";
            }
            console.log("  Hex: " + outputStr);
        }
        
        console.log("─".repeat(70));
        
        return result;
    };
    
    console.log("\n[模块 2] Hook base64Encode()");
    console.log("─".repeat(70));
    
    Utils.base64Encode.implementation = function(data) {
        console.log("\n[base64Encode 被调用]");
        
        var result = this.base64Encode(data);
        
        console.log("  输入长度: " + (data ? data.length : 0) + " 字节");
        console.log("  输出长度: " + (result ? result.length() : 0) + " 字符");
        console.log("  Base64 结果: " + result);
        console.log("");
        
        return result;
    };
    
    console.log("\n[模块 3] 统计信息");
    console.log("─".repeat(70));
    
    setInterval(function() {
        if (encryptCount > 0) {
            console.log("\n═".repeat(70));
            console.log("📊 [加密统计]");
            console.log("═".repeat(70));
            console.log("  总加密次数: " + encryptCount);
            console.log("  三重加密: " + Math.floor(encryptCount / 3) + " 组");
            console.log("═".repeat(70) + "\n");
        }
    }, 15000);
    
    console.log("\n[✓] 所有 Hook 已设置\n");
});

