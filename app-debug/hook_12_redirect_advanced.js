// 第十二关 - 文件重定向（增强版 - 自动创建目标文件）

console.log("[★] 第十二关：文件重定向（增强版）");
console.log("=".repeat(60) + "\n");

// 步骤 1: 创建重定向目标文件
var targetFile = "/data/local/tmp/status.txt";
var targetContent = "Redirect"; // 短内容，触发验证成功

console.log("[步骤 1] 准备重定向目标文件");
console.log("  路径: " + targetFile);

// 使用 Java File API 创建文件
Java.perform(function() {
    var File = Java.use("java.io.File");
    var FileWriter = Java.use("java.io.FileWriter");
    
    try {
        var file = File.$new(targetFile);
        var writer = FileWriter.$new(file);
        writer.write(targetContent);
        writer.close();
        
        console.log("  ✓ 目标文件已创建");
        console.log("  内容: " + targetContent + " (" + targetContent.length + " 字符)");
    } catch (e) {
        console.log("  ⚠️  无法创建文件: " + e);
        console.log("  提示：可能需要手动创建文件");
    }
    console.log("");
});

// 步骤 2: Hook open 函数实现重定向
console.log("[步骤 2] Hook open 函数");

var redirectMap = {
    "/proc/self/status": targetFile
};

var openPtr = Module.findExportByName("libc.so", "open");

if (openPtr) {
    Interceptor.attach(openPtr, {
        onEnter: function(args) {
            var originalPath = args[0].readCString();
            
            // 检查是否需要重定向
            if (redirectMap[originalPath]) {
                var newPath = redirectMap[originalPath];
                
                console.log("\n[open() 重定向]");
                console.log("  原路径: " + originalPath);
                console.log("  新路径: " + newPath);
                
                // 修改参数
                args[0] = Memory.allocUtf8String(newPath);
                this.redirected = true;
            }
        },
        
        onLeave: function(retval) {
            if (this.redirected) {
                var fd = retval.toInt32();
                if (fd >= 0) {
                    console.log("  ✓ 文件描述符: " + fd + " (重定向成功)");
                } else {
                    console.log("  ✗ 打开失败");
                }
            }
        }
    });
    
    console.log("  ✓ Hook 完成");
    console.log("");
    console.log("=".repeat(60));
    console.log("【使用方法】");
    console.log("在 App 中点击 'Test File Redirect' 按钮");
    console.log("应该显示：验证结果：成功！");
    console.log("=".repeat(60) + "\n");
    
} else {
    console.log("  ✗ 未找到 open 函数");
}
