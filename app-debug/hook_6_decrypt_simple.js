// 第六关 - 解密密文（简洁版）

Java.perform(function() {
    console.log("[★] 解密目标密文\n");
    
    var targetCiphertext = "6FLAoCchJ21hsq2sJzOeTQ==";
    console.log("[目标] " + targetCiphertext + "\n");
    
    setTimeout(function() {
        try {
            // 获取密钥
            var Login = Java.use("cn.binary.frida.Login");
            var aesKey = Login.$new().getSecretKey();
            console.log("[密钥] " + aesKey + "\n");
            
            // 解密
            var Base64 = Java.use("android.util.Base64");
            var Cipher = Java.use("javax.crypto.Cipher");
            var SecretKeySpec = Java.use("javax.crypto.spec.SecretKeySpec");
            var String = Java.use("java.lang.String");
            
            var ciphertextBytes = Base64.decode(targetCiphertext, 2);
            var keyBytes = aesKey.getBytes("UTF-8");
            var secretKey = SecretKeySpec.$new(keyBytes, "AES");
            
            var cipher = Cipher.getInstance("AES/ECB/PKCS7Padding");
            cipher.init(2, secretKey);
            var decryptedBytes = cipher.doFinal(ciphertextBytes);
            var result = String.$new(decryptedBytes, "UTF-8");
            
            console.log("═══════════════════════════════");
            console.log("解密结果: " + result);
            
            if (result.indexOf(":") !== -1) {
                var parts = result.split(":");
                console.log("\n用户名: " + parts[0]);
                console.log("密码:   " + parts[1]);
            }
            console.log("═══════════════════════════════\n");
            
        } catch (e) {
            console.log("[-] 解密失败: " + e);
        }
    }, 1000);
});
