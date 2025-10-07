// 第三关 - 修改会员状态（高级版）
// 适配 Frida 17.x

console.log("[★] 第三关：修改会员状态 - 高级版");
console.log("=".repeat(70) + "\n");

Java.perform(function() {
    var Login = Java.use("cn.binary.frida.Login");
    
    console.log("[方法 1] Hook isPremiumUser()");
    console.log("─".repeat(70));
    
    Login.isPremiumUser.implementation = function() {
        console.log("\n[isPremiumUser 被调用]");
        
        // 获取调用栈
        var Thread = Java.use("java.lang.Thread");
        var stack = Thread.currentThread().getStackTrace();
        
        console.log("  调用者:");
        for (var i = 0; i < Math.min(5, stack.length); i++) {
            var className = stack[i].getClassName();
            var methodName = stack[i].getMethodName();
            if (className && methodName) {
                console.log("    [" + i + "] " + className + "." + methodName + "()");
            }
        }
        
        console.log("\n  原返回值: false");
        console.log("  修改为: true");
        console.log("  ✓ 成为高级会员！\n");
        
        return true;
    };
    
    console.log("\n[方法 2] 主动调用验证");
    console.log("─".repeat(70));
    
    setTimeout(function() {
        try {
            var loginInstance = Login.$new();
            var isPremium = loginInstance.isPremiumUser();
            
            console.log("\n[验证结果]");
            console.log("  isPremiumUser(): " + isPremium);
            
            if (isPremium) {
                console.log("  ✓ Hook 成功，已是高级会员！");
            } else {
                console.log("  ✗ Hook 可能失败");
            }
            console.log("");
            
        } catch (e) {
            console.log("[-] 验证失败: " + e);
        }
    }, 1000);
    
    console.log("\n[✓] Hook 设置完成\n");
});

