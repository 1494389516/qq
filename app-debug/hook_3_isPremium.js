// 第三关 - 修改会员状态为高级会员

Java.perform(function() {
    console.log("[★] Hook isPremiumUser 方法");
    
    var loginClazz = Java.use("cn.binary.frida.Login");
    
    // Hook isPremiumUser 方法，强制返回 true
    loginClazz.isPremiumUser.implementation = function () {
        console.log("[★] isPremiumUser 被调用 -> 返回 true (高级会员)");
        return true;
    };
    
    console.log("[✓] Hook 完成！现在是高级会员了");
});

