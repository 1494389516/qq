// 第十二关 - 文件重定向（简洁版）

console.log("[★] 第十二关：文件重定向\n");

// 重定向配置
var redirect = {
    "/proc/self/status": "/data/local/tmp/status.txt"
};

var openPtr = Module.findExportByName("libc.so", "open");

Interceptor.attach(openPtr, {
    onEnter: function(args) {
        var path = args[0].readCString();
        
        console.log("\n[open] " + path);
        
        // 检查重定向
        if (redirect[path]) {
            var newPath = redirect[path];
            console.log("  🔀 重定向到: " + newPath);
            
            // 修改参数
            args[0] = Memory.allocUtf8String(newPath);
            this.redirected = true;
        }
    },
    
    onLeave: function(retval) {
        if (this.redirected) {
            console.log("  ✓ 返回值: " + retval + " (重定向成功)");
        }
    }
});

console.log("[✓] Hook 完成\n");
