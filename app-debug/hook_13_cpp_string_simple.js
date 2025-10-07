// 第十三关 - Hook C++ 函数获取 License（简洁版）

console.log("[★] 第十三关：Hook processLicenseData\n");

var moduleName = "libfrida.so";
var base = Module.findBaseAddress(moduleName);

console.log("模块基地址: " + base + "\n");

// 枚举所有导出符号，查找目标函数
var exports = Module.enumerateExports(moduleName);

console.log("[导出符号搜索]");
console.log("包含 'License' 的符号：\n");

var found = false;
exports.forEach(function(exp) {
    var name = exp.name;
    
    if (name.indexOf("License") !== -1 || 
        name.indexOf("license") !== -1 ||
        name.indexOf("processLicense") !== -1) {
        
        console.log("  " + name);
        console.log("  → " + exp.address);
        console.log("");
        
        // Hook 这个函数
        Interceptor.attach(exp.address, {
            onEnter: function(args) {
                console.log("\n[" + name + " 被调用]");
                console.log("  参数 a1: " + args[0]);
                
                // 保存参数用于 onLeave
                this.arg0 = args[0];
            },
            
            onLeave: function(retval) {
                console.log("  返回值: " + retval);
                
                // 尝试读取返回值（如果是字符串）
                try {
                    // 对于 std::string，通常内部有一个指针
                    // 尝试读取内存
                    if (retval && !retval.isNull()) {
                        var ptr = retval.readPointer();
                        var str = ptr.readCString();
                        if (str && str.length > 0) {
                            console.log("  🔑 许可证数据: " + str);
                        }
                    }
                } catch (e) {
                    // 如果读取失败，尝试其他方式
                    try {
                        var str2 = retval.readCString();
                        if (str2) {
                            console.log("  🔑 许可证数据: " + str2);
                        }
                    } catch (e2) {}
                }
            }
        });
        
        found = true;
    }
});

if (!found) {
    console.log("  未找到相关符号");
    console.log("\n显示前 20 个导出符号：\n");
    
    for (var i = 0; i < 20 && i < exports.length; i++) {
        console.log("  [" + i + "] " + exports[i].name);
    }
}

console.log("\n[✓] Hook 完成\n");
