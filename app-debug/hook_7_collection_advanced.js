// 第七关 - 集合遍历（高级版）
// 适配 Frida 17.x

console.log("[★] 第七关：集合遍历与Flag提取 - 高级版");
console.log("=".repeat(70) + "\n");

Java.perform(function() {
    
    // ═════════════════════════════════════════════════════════════
    // 模块 1：主动解密 Flag
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 1] 主动解密所有 Flag");
    console.log("─".repeat(70));
    
    setTimeout(function() {
        try {
            var Utils = Java.use("cn.binary.frida.Utils");
            var String = Java.use("java.lang.String");
            
            // 已知的加密 Flag
            var encryptedFlags = {
                "List[3]": "MjUyJCVndXlgBh98ejgjMyITKwo4K2Z2dw==",
                "Map[secret]": "MjUyJCUyLTUfAyIjNyofC3Us41oY5Xs=",
                "Array[3]": "MjUyJCVwdzQpOCE5AxUpISsvGQoLFjl3"
            };
            
            // 密钥
            var key = "8848".getBytes();
            
            console.log("\n[解密过程]");
            console.log("密钥: 8848\n");
            
            for (var location in encryptedFlags) {
                var encrypted = encryptedFlags[location];
                
                console.log("[" + location + "]");
                console.log("  加密数据: " + encrypted);
                
                // Base64 解码
                var decoded = Utils.base64Decode(encrypted);
                console.log("  解码长度: " + decoded.length + " 字节");
                
                // 解密
                var decrypted = Utils.simpleDecrypt(decoded, key);
                var flag = String.$new(decrypted);
                
                console.log("  🚩 Flag: " + flag);
                console.log("");
            }
            
            console.log("=".repeat(70) + "\n");
            
        } catch (e) {
            console.log("[-] 解密失败: " + e);
            console.log(e.stack);
        }
    }, 500);
    
    // ═════════════════════════════════════════════════════════════
    // 模块 2：Hook traverseCollections 方法
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 2] Hook traverseCollections()");
    console.log("─".repeat(70) + "\n");
    
    var CollectionTraversal = Java.use("cn.binary.frida.CollectionTraversal");
    
    CollectionTraversal.traverseCollections.implementation = function() {
        console.log("\n[traverseCollections 被调用]");
        console.log("=".repeat(70));
        
        // 访问所有集合
        var list = this.stringList.value;
        var map = this.stringMap.value;
        var array = this.stringArray.value;
        
        console.log("\n[List 集合]");
        console.log("  大小: " + list.size());
        console.log("  类型: " + list.getClass().getName());
        console.log("  内容:");
        
        for (var i = 0; i < list.size(); i++) {
            var item = list.get(i);
            console.log("    [" + i + "] " + item);
            
            // 检查是否是加密的 Flag
            if (item.length() > 20 && item.indexOf("=") !== -1) {
                console.log("        → 疑似加密数据");
            }
        }
        
        console.log("\n[Map 集合]");
        console.log("  大小: " + map.size());
        console.log("  类型: " + map.getClass().getName());
        console.log("  内容:");
        
        var entries = map.entrySet().iterator();
        while (entries.hasNext()) {
            var entry = entries.next();
            var key = entry.getKey();
            var value = entry.getValue();
            console.log("    " + key + " = " + value);
            
            if (value.length() > 20 && value.indexOf("=") !== -1) {
                console.log("        → 疑似加密数据");
            }
        }
        
        console.log("\n[Array 集合]");
        console.log("  长度: " + array.length);
        console.log("  类型: " + array.getClass().getName());
        console.log("  内容:");
        
        for (var j = 0; j < array.length; j++) {
            var item = array[j];
            console.log("    [" + j + "] " + item);
            
            if (item.length() > 20 && item.indexOf("=") !== -1) {
                console.log("        → 疑似加密数据");
            }
        }
        
        console.log("\n=".repeat(70));
        console.log("[注意] 原方法会过滤掉 startsWith(\"flag\") 的项");
        console.log("=".repeat(70) + "\n");
        
        // 调用原方法（会过滤）
        return this.traverseCollections();
    };
    
    // ═════════════════════════════════════════════════════════════
    // 模块 3：主动获取 CollectionTraversal 实例
    // ═════════════════════════════════════════════════════════════
    
    console.log("[模块 3] 主动获取实例并访问集合");
    console.log("─".repeat(70) + "\n");
    
    setTimeout(function() {
        try {
            var instances = Java.choose("cn.binary.frida.CollectionTraversal");
            
            if (instances.length > 0) {
                console.log("[✓] 找到 " + instances.length + " 个实例\n");
                
                instances.forEach(function(instance, idx) {
                    console.log("【实例 #" + (idx + 1) + "】");
                    console.log("─".repeat(70));
                    
                    // 直接访问字段
                    var list = instance.stringList.value;
                    var map = instance.stringMap.value;
                    var array = instance.stringArray.value;
                    
                    console.log("List 大小: " + list.size());
                    console.log("Map 大小: " + map.size());
                    console.log("Array 长度: " + array.length);
                    console.log("");
                });
            } else {
                console.log("[-] 未找到实例\n");
            }
        } catch (e) {
            console.log("[-] 获取实例失败: " + e + "\n");
        }
    }, 2000);
    
    console.log("[✓] 所有 Hook 已设置完成\n");
});

