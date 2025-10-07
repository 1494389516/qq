// 第七关 - 集合遍历（简洁版）

Java.perform(function() {
    console.log("[★] 第七关：获取集合中的隐藏 Flag\n");
    
    setTimeout(function() {
        try {
            var Utils = Java.use("cn.binary.frida.Utils");
            var String = Java.use("java.lang.String");
            var key = "8848".getBytes();
            
            var encrypted = {
                "List Flag  ": "MjUyJCVndXlgBh98ejgjMyITKwo4K2Z2dw==",
                "Map Flag   ": "MjUyJCUyLTUfAyIjNyofC3Us41oY5Xs=",
                "Array Flag ": "MjUyJCVwdzQpOCE5AxUpISsvGQoLFjl3"
            };
            
            console.log("=".repeat(50));
            for (var name in encrypted) {
                var decoded = Utils.base64Decode(encrypted[name]);
                var decrypted = Utils.simpleDecrypt(decoded, key);
                var flag = String.$new(decrypted);
                console.log(name + ": " + flag);
            }
            console.log("=".repeat(50) + "\n");
            
        } catch (e) {
            console.log("[-] 解密失败: " + e);
        }
    }, 500);
    
    // Hook 遍历方法，显示完整内容
    var CollectionTraversal = Java.use("cn.binary.frida.CollectionTraversal");
    
    CollectionTraversal.traverseCollections.implementation = function() {
        console.log("[集合内容]");
        
        var list = this.stringList.value;
        var map = this.stringMap.value;
        var array = this.stringArray.value;
        
        console.log("\nList:");
        for (var i = 0; i < list.size(); i++) {
            var item = list.get(i);
            console.log("  [" + i + "] " + item);
        }
        
        console.log("\nMap:");
        var entries = map.entrySet().iterator();
        while (entries.hasNext()) {
            var entry = entries.next();
            console.log("  " + entry.getKey() + " = " + entry.getValue());
        }
        
        console.log("\nArray:");
        for (var j = 0; j < array.length; j++) {
            console.log("  [" + j + "] " + array[j]);
        }
        console.log("");
        
        return this.traverseCollections();
    };
    
    console.log("[✓] Hook 完成\n");
});
