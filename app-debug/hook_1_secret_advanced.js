// 第一关 - 获取密钥（高级版）
// 适配 Frida 17.x

console.log("[★] 第一关：获取密钥 - 高级版");
console.log("=".repeat(70) + "\n");

Java.perform(function() {
    console.log("[模块 1] Hook getSecretKey() 方法");
    console.log("─".repeat(70));
    
    var Login = Java.use("cn.binary.frida.Login");
    
    // Hook getSecretKey
    Login.getSecretKey.implementation = function() {
        console.log("\n[getSecretKey 被调用]");
        
        var result = this.getSecretKey();
        
        console.log("  ✓ 密钥获取成功");
        console.log("  密钥内容: " + result);
        console.log("  密钥长度: " + result.length + " 字符");
        console.log("  密钥类型: " + typeof result);
        console.log("");
        
        return result;
    };
    
    // Hook 构造函数，查看密钥初始化
    console.log("[模块 2] Hook Login 构造函数");
    console.log("─".repeat(70) + "\n");
    
    Login.$init.overload().implementation = function() {
        console.log("[Login 对象创建]");
        
        var result = this.$init();
        
        // 访问内部字段
        try {
            var keyField = this.key.value;
            console.log("  内部 key 字段: " + keyField);
        } catch (e) {
            console.log("  无法访问内部字段");
        }
        
        console.log("");
        return result;
    };
    
    // 主动调用获取密钥
    console.log("[模块 3] 主动调用获取密钥");
    console.log("─".repeat(70));
    
    setTimeout(function() {
        try {
            var loginInstance = Login.$new();
            var secretKey = loginInstance.getSecretKey();
            
            console.log("\n🔑 [主动获取成功]");
            console.log("═".repeat(70));
            console.log("  密钥: " + secretKey);
            console.log("═".repeat(70) + "\n");
            
        } catch (e) {
            console.log("[-] 主动调用失败: " + e);
        }
    }, 1000);
    
    console.log("\n[✓] 所有 Hook 已设置完成\n");
});

