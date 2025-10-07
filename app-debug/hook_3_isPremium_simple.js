// 第三关 - 修改会员状态（简洁版）
// 适配 Frida 17.x

console.log("[★] 第三关：修改会员状态\n");

Java.perform(function() {
    var Login = Java.use("cn.binary.frida.Login");
    
    Login.isPremiumUser.implementation = function() {
        console.log("[isPremiumUser 被调用]");
        console.log("  原返回值: false");
        console.log("  修改为: true");
        console.log("  ✓ 成为高级会员！\n");
        return true;
    };
    
    console.log("[✓] Hook 完成\n");
});

