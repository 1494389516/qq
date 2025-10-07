// 第四关 - 监控加密（简洁版）

Java.perform(function() {
    console.log("[★] 监控加密过程");
    
    var Utils = Java.use("cn.binary.frida.Utils");
    var count = 0;
    
    // 字节转十六进制
    function toHex(bytes) {
        var hex = "";
        for (var i = 0; i < bytes.length; i++) {
            var h = (bytes[i] & 0xFF).toString(16);
            hex += (h.length === 1 ? "0" : "") + h + " ";
        }
        return hex.trim();
    }
    
    // Hook 加密方法
    Utils.simpleEncrypt.overload('[B').implementation = function(input) {
        count++;
        var output = this.simpleEncrypt(input);
        
        console.log("\n[加密 " + count + "]");
        console.log("输入: " + toHex(input));
        console.log("输出: " + toHex(output));
        
        return output;
    };
    
    // Hook Base64
    Utils.base64Encode.implementation = function(bytes) {
        var result = this.base64Encode(bytes);
        console.log("\n[Base64] " + result);
        count = 0;
        return result;
    };
    
    console.log("[✓] Hook 完成\n");
});

