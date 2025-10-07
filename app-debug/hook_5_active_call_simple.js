// 第五关 - 主动调用获取敏感数据（简洁版）
// 适配 Frida 17.x

Java.perform(function() {
    console.log("[★] 主动调用 processSensitiveData");
    
    // Frida 17.x: Java.choose() 现在返回数组
    var instances = Java.choose("cn.binary.frida.SensitiveDataProcessor");
    
    if (instances.length > 0) {
        console.log("[✓] 找到 " + instances.length + " 个实例，开始调用...\n");
        
        instances.forEach(function(instance) {
            // 直接调用，不传参数或传特定参数
            var result = instance.processSensitiveData("flag");
            
            console.log("═══════════════════════════════");
            console.log("输入: flag");
            console.log("输出: " + result);
            console.log("═══════════════════════════════\n");
        });
    } else {
        console.log("[-] 未找到实例，请先启动应用并打开相应界面");
    }
});

