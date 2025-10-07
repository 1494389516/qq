// 第二关 - 高级版：智能绕过特定黑名单

Java.perform(function() {
    console.log("[★] 智能 Hook canLogin 方法");
    
    var loginClazz = Java.use("cn.binary.frida.Login");
    
    // 需要绕过的黑名单用户名
    var bypassList = ["root", "admin", "administrator"];
    
    loginClazz.canLogin.implementation = function (username) {
        var originalResult = this.canLogin(username);
        
        console.log("\n[检测] 用户名: " + username);
        console.log("[原始] 返回值: " + originalResult);
        
        // 如果原始返回 false，并且在我们的绕过列表中，就改为 true
        if (!originalResult && bypassList.indexOf(username) !== -1) {
            console.log("[修改] " + username + " 在黑名单中，强制返回 true");
            return true;
        }
        
        console.log("[保持] 返回原始值: " + originalResult);
        return originalResult;
    };
    
    console.log("[✓] Hook 完成！绕过列表: " + bypassList.join(", "));
});

