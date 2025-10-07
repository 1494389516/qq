// 第九关 - 获取 libfrida.so 基地址（简洁版）

Java.perform(function() {
    console.log("[★] 第九关：获取 libfrida.so 基地址\n");
    
    // 方法 1: Process API
    var modules = Process.enumerateModules();
    for (var i = 0; i < modules.length; i++) {
        if (modules[i].name === "libfrida.so") {
            console.log("=".repeat(50));
            console.log("libfrida.so 基地址: " + modules[i].base);
            console.log("大小: " + modules[i].size + " 字节");
            console.log("路径: " + modules[i].path);
            console.log("=".repeat(50) + "\n");
            break;
        }
    }
    
    // 方法 2: Module API
    var baseAddr = Module.findBaseAddress("libfrida.so");
    if (baseAddr) {
        console.log("验证: " + baseAddr + "\n");
    }
    
    // Hook 验证方法
    var MainActivity = Java.use("cn.binary.frida.MainActivity");
    MainActivity.getLibFridaBaseAddress.implementation = function() {
        var result = this.getLibFridaBaseAddress();
        console.log("[验证] 返回值: 0x" + result.toString(16));
        return result;
    };
    
    console.log("[✓] Hook 完成\n");
});
