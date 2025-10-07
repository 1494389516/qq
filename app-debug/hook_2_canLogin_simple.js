// 第二关 - 简洁版：强制 canLogin 返回 true

Java.perform(function() {
    console.log("[★] Hook canLogin 方法");
    
    var loginClazz = Java.use("cn.binary.frida.Login");
    
    loginClazz.canLogin.implementation = function (username) {
        console.log("[★] 输入: " + username + " -> 强制返回 true");
        return true;  // 直接返回 true，绕过所有限制
    };
    
    console.log("[✓] Hook 完成！");
});

