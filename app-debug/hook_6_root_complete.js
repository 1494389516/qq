// 第六关完整版 - 解密 root 密码 + 绕过黑名单

Java.perform(function() {
    console.log("[★] 第六关完整解决方案");
    console.log("=".repeat(60) + "\n");
    
    // ========== 任务 1: 解密密码 ==========
    setTimeout(function() {
        console.log("[任务 1] 解密 root 用户密码");
        console.log("-".repeat(60));
        
        try {
            var Login = Java.use("cn.binary.frida.Login");
            var aesKey = Login.$new().getSecretKey();
            
            var Base64 = Java.use("android.util.Base64");
            var Cipher = Java.use("javax.crypto.Cipher");
            var SecretKeySpec = Java.use("javax.crypto.spec.SecretKeySpec");
            var String = Java.use("java.lang.String");
            
            var targetCiphertext = "6FLAoCchJ21hsq2sJzOeTQ==";
            var ciphertextBytes = Base64.decode(targetCiphertext, 2);
            var keyBytes = aesKey.getBytes("UTF-8");
            var secretKey = SecretKeySpec.$new(keyBytes, "AES");
            
            var cipher = Cipher.getInstance("AES/ECB/PKCS7Padding");
            cipher.init(2, secretKey);
            var decryptedBytes = cipher.doFinal(ciphertextBytes);
            var credentials = String.$new(decryptedBytes, "UTF-8");
            
            var parts = credentials.split(":");
            var username = parts[0];
            var password = parts[1];
            
            console.log("✓ 解密成功！");
            console.log("  用户名: " + username);
            console.log("  密码: " + password);
            console.log("-".repeat(60) + "\n");
            
        } catch (e) {
            console.log("✗ 解密失败: " + e + "\n");
        }
    }, 500);
    
    // ========== 任务 2: 绕过黑名单 ==========
    console.log("[任务 2] 绕过 canLogin 黑名单");
    console.log("-".repeat(60));
    
    var Login = Java.use("cn.binary.frida.Login");
    Login.canLogin.implementation = function(username) {
        var originalResult = this.canLogin(username);
        
        if (username === "root") {
            console.log("[Hook] canLogin(root): false → true (已绕过)");
            return true;  // 绕过黑名单
        }
        
        return originalResult;
    };
    console.log("✓ canLogin Hook 完成\n");
    
    // ========== 任务 3: 监控加密验证 ==========
    console.log("[任务 3] 监控加密和登录");
    console.log("-".repeat(60));
    
    var ApiLogin = Java.use("cn.binary.frida.ApiLogin");
    
    ApiLogin.encryptCredentials.implementation = function(username, password) {
        console.log("\n[加密] " + username + ":" + password);
        var result = this.encryptCredentials(username, password);
        console.log("  密文: " + result);
        
        if (result === "6FLAoCchJ21hsq2sJzOeTQ==") {
            console.log("  ✓ 匹配目标密文！");
        }
        
        return result;
    };
    
    ApiLogin.performApiLogin.implementation = function(username, password) {
        console.log("\n[登录验证] " + username);
        var result = this.performApiLogin(username, password);
        
        if (result) {
            console.log("  ✓✓✓ 登录成功！✓✓✓\n");
        } else {
            console.log("  ✗ 登录失败\n");
        }
        
        return result;
    };
    
    console.log("✓ 监控 Hook 完成\n");
    
    console.log("=".repeat(60));
    console.log("【准备就绪】");
    console.log("在 App 中使用解密得到的用户名和密码登录");
    console.log("root 用户已自动绕过黑名单限制");
    console.log("=".repeat(60) + "\n");
});
