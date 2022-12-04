# python正则表达式



## 匹配字符串'xxx'第一次匹配的位置

```
re.search(r'xxx','l love xxxxxx')
```

 

## 找出所有在字符串str中‘xx’所在的位置

```
index = -1
while True:
    index = str.find('xx',index+30)
    if index == -1:
        break
```

 

 

![img](./assets/1013528-20170920212227275-1677163860.png)

^：以哪个字符作为开头
$：以哪个字符结尾

 

其实正则表达式用的比较多的就是贪心算法，关于贪心算法的实例。

### 实例1：把尖括号里面的内容提取出来

```
str = "<www.hao123.com>,<www.baidu.com>"、
a = re.findall('<(.*?)>',str)
print a
```

 

### 实例2：查询一个IP的经纬度



```
with open("D:\getPoint\getPoint\spiders\IPs.txt") as IPs:
    for IP in IPs:
        url = "https://www.shodan.io/host/"+IP
        req = requests.get(url=url)
        # print req.content
        point = re.search("setView\(\[(.*?)\]",req.content,re.S).group(1)
        print IP[:-1]+"坐标 ===> "+point
```

