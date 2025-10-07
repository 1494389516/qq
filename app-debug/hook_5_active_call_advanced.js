// 第五关 - Frida 主动调用，绕过 flag 过滤（高级版）
// 适配 Frida 17.x

console.log("[★] Frida 主动调用 processSensitiveData - 高级版");
console.log("=".repeat(70) + "\n");

Java.perform(function() {
    console.log("[模块 1] 主动调用 processSensitiveData");
    console.log("─".repeat(70));
    
    // Frida 17.x: Java.choose() 现在返回数组
    var instances = Java.choose("cn.binary.frida.SensitiveDataProcessor");
    
    if (instances.length > 0) {
        console.log("\n[✓] 找到 " + instances.length + " 个 SensitiveDataProcessor 实例\n");
        
        instances.forEach(function(instance, idx) {
            console.log("【实例 #" + (idx + 1) + "】");
            console.log("─".repeat(70));
            
            // 尝试不同的输入
            var testInputs = ["test", "flag", "secret", "data", "admin", "get_flag"];
            
            testInputs.forEach(function(input) {
                try {
                    var result = instance.processSensitiveData(input);
                    
                    console.log("\n输入: \"" + input + "\"");
                    console.log("输出: " + result);
                    console.log("长度: " + result.length());
                    
                    // 检查是否包含 flag
                    if (result.indexOf("flag") !== -1) {
                        console.log(">>> 🚩 发现 FLAG！<<<");
                        console.log(">>> " + result + " <<<");
                    }
                } catch (e) {
                    console.log("\n输入: \"" + input + "\"");
                    console.log("错误: " + e.message);
                }
            });
            
            console.log("\n" + "─".repeat(70) + "\n");
        });
        
        console.log("[✓] 主动调用完成！\n");
    } else {
        console.log("[-] 未找到实例，请先启动应用并打开相应界面\n");
    }
    
    // 同时 Hook 这个方法，监控所有调用
    console.log("[模块 2] Hook processSensitiveData");
    console.log("─".repeat(70) + "\n");
    
    var SensitiveDataProcessor = Java.use("cn.binary.frida.SensitiveDataProcessor");
    
    SensitiveDataProcessor.processSensitiveData.implementation = function(input) {
        console.log("[Hook] processSensitiveData 被调用");
        console.log("  输入: " + input);
        console.log("  输入长度: " + input.length());
        
        var result = this.processSensitiveData(input);
        
        console.log("  输出: " + result);
        console.log("  输出长度: " + result.length());
        console.log("  包含 flag: " + (result.indexOf("flag") !== -1));
        console.log("");
        
        return result;
    };
    
    console.log("[✓] Hook 完成！\n");
});
