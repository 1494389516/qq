// 小试牛刀 - Hook 获取 loginState.getSecretKey() 的返回值

Java.perform(function() {
    // 在这里编写 Hook 代码
    console.log("[★] Java.perform 初始化完成");
    
    // 获取 loginState 对象
    var loginClazz = Java.use("cn.binary.frida.Login");
    
    loginClazz.getSecretKey.implementation = function () {
        console.log("[★] 调用了 getSecretKey 方法");
        
        // 调用原始方法获取返回值
        var result = this.getSecretKey();
        
        // 打印返回值
        console.log("[★] getSecretKey 返回值: " + result);
        
        // 返回原始值（也可以返回自定义值）
        return result;
    };
});

