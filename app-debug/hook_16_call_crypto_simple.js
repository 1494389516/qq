// 第十六关 - 主动调用加密函数（简洁版）
// 适配 Frida 17.x

console.log("[★] 第十六关：主动调用 Native 加密函数\n");

var moduleName = "libfrida.so";
var base = Module.findBaseAddress(moduleName);

if (!base) {
    console.log("[-] 未找到 " + moduleName);
} else {
    console.log("[✓] " + moduleName + " 基地址: " + base + "\n");
    
    // 查找加密相关函数
    var exports = Module.enumerateExports(moduleName);
    
    var crypto_init = null;
    var crypto_crypt = null;
    
    exports.forEach(function(exp) {
        if (exp.name.indexOf("crypto_init") !== -1) {
            crypto_init = exp.address;
            console.log("[找到] crypto_init: " + exp.address);
        }
        if (exp.name.indexOf("crypto_crypt") !== -1) {
            crypto_crypt = exp.address;
            console.log("[找到] crypto_crypt: " + exp.address);
        }
    });
    
    console.log("");
    
    if (crypto_init && crypto_crypt) {
        console.log("=".repeat(60));
        console.log("[开始主动调用加密]");
        console.log("=".repeat(60) + "\n");
        
        // 分配内存用于存储结果
        var resultBuffer = Memory.alloc(256);
        var keyBuffer = Memory.allocUtf8String("test_key");
        
        // 创建函数调用
        var initFunc = new NativeFunction(crypto_init, 'int64', ['int64', 'pointer', 'uint64']);
        var cryptFunc = new NativeFunction(crypto_crypt, 'int64', ['int64', 'pointer', 'uint64']);
        
        // 测试数据
        var testData = "Hello Frida!";
        var dataBuffer = Memory.allocUtf8String(testData);
        
        console.log("[输入] " + testData);
        console.log("[密钥] test_key\n");
        
        // 调用 crypto_init
        console.log("[步骤 1] 调用 crypto_init");
        var initResult = initFunc(resultBuffer, keyBuffer, 8);
        console.log("  返回值: " + initResult);
        
        // 调用 crypto_crypt
        console.log("\n[步骤 2] 调用 crypto_crypt");
        var cryptResult = cryptFunc(resultBuffer, dataBuffer, testData.length);
        console.log("  返回值: " + cryptResult);
        
        // 读取加密结果
        console.log("\n[加密结果]");
        console.log(hexdump(resultBuffer, {
            length: 64,
            header: true,
            ansi: true
        }));
        
        console.log("\n" + "=".repeat(60));
        console.log("[✓] 加密完成！\n");
        
    } else {
        console.log("[-] 未找到加密函数，尝试搜索所有符号...\n");
        
        console.log("[所有导出符号]");
        exports.slice(0, 30).forEach(function(exp, i) {
            console.log("  [" + i + "] " + exp.name + " → " + exp.address);
        });
    }
}

