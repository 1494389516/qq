// 第十一关 - Hook open 函数（简洁版）

console.log("[★] 第十一关：Hook open() 阻止 'hack' 文件\n");

var openPtr = Module.findExportByName("libc.so", "open");

Interceptor.attach(openPtr, {
    onEnter: function(args) {
        var path = args[0].readCString();
        
        console.log("\n[open] " + path);
        
        if (path.indexOf("hack") !== -1) {
            console.log("  🚫 检测到 'hack'，阻止打开！");
            this.block = true;
        }
    },
    
    onLeave: function(retval) {
        if (this.block) {
            console.log("  返回: -1 (失败)");
            retval.replace(-1);
        }
    }
});

console.log("[✓] Hook 完成\n");
