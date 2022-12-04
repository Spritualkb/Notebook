浏览器

作用：发送http请求，接收回传的数据，渲染网页

服务器

服务器具有告诉的CPU运算能力、长时间的可靠运行能力

# http协议

## 一次浏览器的请求过程



-  浏览器通过 DNS 把域名解析成对应的IP地址；
-  根据这个 IP 地址在互联网上找到对应的服务器，建立连接；
-  客户端向服务器发送HTTP协议请求包，请求服务器里的资源文档；
-  在服务器端，实际上还有复杂的业务逻辑比如服务器可能有多台，到底指定哪台服务器处理请求。都需要一个负载均衡设备来平均分配所有用户的请求，还有请求的数据是存储在分布式缓存里还是一个静态文件中，或是在数据库里，完成以上操作之后，服务器将相应的数据资源返回给浏览器
-  客户端与服务器断开。由客户端解析HTML文档，在客户端屏幕上渲染图形结果

![img](photo/clip_image002.jpg) 

###  纯文本和超文本区别

纯文本只能包含文字内容，不能保存样式等，且不能传输样式等

超文本：最常用的Word文档.doc，.ppt

# HTML骨架

## DTD

```html
<!DOCTYPE html>
<html lang="en" dir="ltr">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>Document</title>
  </head>
  <body>
    <div>文字</div>

  </body>
</html>
```

DTD(Document Type Definition,文本类型定义)，必须出现在第一行。让浏览器知道什么格式的文件

```html
<!DOCTYPE html>
```

<!>，表示警示标签。

DOCTYPE表示文档类型，html就是HTML超文本标记语言

## 关于html标签

整个网页必须被<html></html>包裹，它里面有<head></head>和<body></body>两部分

- <head></head>:网页的配置

- <body></body>:网页的正式内容，浏览器可是区域

标签有一个属性lang，是英文language的意思，表示整个网页的主体语言。

en表示英文。中文的表示有三种方法。zh、cn、zh-CN。

需要注意的是，无论哪种语言，都使用英文开发

```html
<html lang="en">
<html lang="zh-CN">
```

## 字符集

在head标签对中，是一个个文件的位置。几乎所有的位置都是写在meta标签中的。

meta就是“元”的意思，表示基本配置。

首先映入眼帘的是配置字符集；

```html
<meta charset="UTF=8">
<meta charset="gb2312">
<meta charset="gbk">
```

charset是英文charset set文字集合的意思

| **字符集** | **字库是否全面**                                             | **优缺点**                                                   |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **UTF-8**  | 这个字库涵盖了地球上所有国家、民族的语言文字。非常全，每年更新，它是一个国际化的字库 | 每个汉字占3个字节。所以如果你想网页快一点打开，不能使用UTF-8。  新华网的阿拉伯语频道、日语频道等都要使用UTF-8. |
| **gb2312** | gb是国标的意思，只有汉族的文字和少量其他符号。               | 每个汉字占2个字节。  几乎所有的门户网站，都是gb2312。        |
| **gbk**    | gbk是gb2312的略微增强版，文字稍微多了点， gbk也是只有汉语，只不过多了点怪异汉语字，比如“喆”。 | 每个汉字占2个字节。  几乎所有的门户网站，都是gbk。           |

如果网页使用场景是面向群体是国际化的，使用utf-8，比如中华网；如果面向群体主要是国内，使用gbk比如腾讯网

## title标签

title就是在浏览器选项卡的区域显示的文字：

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <title>我是网页标签</title>
</head>
<body>
    
    </body>
</html>
```

## keyword关键字

SEO(search engine optimization,搜索引擎优化)

最基本的SEO技术就是把keyword写好。keyword就是网页关键字。

name属性一定要设置为keywords，content就是关键字的内容，中间用逗号隔开

```html
<meta name="keywords" content="前端，HTML，JavaScript">
```

## description页面描述

页面描述就是搜索引擎收到你之后，展示的文字。

```html
<meta name="Description" content="网页的描述" />
```

以腾讯网为例，看到源码，keywords和description

![img](photo/clip_image002-16559113094381.jpg)

![img](photo/clip_image004-16559113094392.jpg)

 

 

## HTML的基本语法

### 标签

1. 标签名必须书写在一对尖括号<>内部。

2. 标签分为单标签和双标签，双标签必须成对出现，有开始标签和结束标签。

3. 结束标签必须有关闭符号/。

4. 根据标签内部存放的内容不同，将不同的标签划分为两个级别

##### 视口标签

<meta name="viewport" content="width=device-width, initial-scale=1.0">

##### 浏览器私有设置


edge是win10中的IE升级版浏览器，这句话的意思表示设置兼容性为让edge和ie渲染方式一样。

<meta http-equiv="X-UA-Compatible" content="ie=edge">

类似的还有一些“双核浏览器”比如360浏览器、QQ浏览器、搜狗高速浏览器、百度浏览器、猎豹浏览器等，都可以加上这句话，表示尽可能的用高级核打开页面：

<meta name="renderer" content="webkit">

##### h标签

h1一般是logo

h系列标签，又称为标题标签，主要作用是给页面文本添加标题语义

##### p标签

p标签是段落

##### a标签

作用是设置文本的超级链接和锚点

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <h2>刘亦菲</h2>
    <p style="text-indent: 2em;">#2em是单位
        1987年8月25日出生于湖北省武汉市，华语影视女演员、歌手，毕业于北京电影学院2002级表演系本科。
    </p>

    <p>
        2002年因出演电视剧《金粉世家》中白秀珠一角踏入演艺圈 [1]  。2003年因出演武侠剧《天龙八部》中王语嫣一角崭露头角 [2]  。2004年凭借仙侠剧《仙剑奇侠传》赵灵儿一角获得了高人气与关注度 [3]  。
    </p>
    <p>
        2005年因在武侠剧《神雕侠侣》中饰演小龙女受到广泛关注 [4-5]  。
        2006年发行首张国语专辑《刘亦菲》和日语专辑《All My Words》 [6-7]  ；
        同年成为金鹰节历史上首位金鹰女神 [8]  。
        2008年起转战大银幕，并凭借好莱坞电影《功夫之王》成为首位荣登IMDB电影新人排行榜榜首的亚洲女星 [9]  。2009年，获封“四小花旦”之一 [10]  。2012年，获得第24届香港专业电影摄影师学会最具魅力女演员奖 [11]  。2014年，凭借古装片《铜雀台》中灵雎一角获得第5届澳门国际电影节最佳女主角奖 [12]  。

    </p>
    
</body>
</html>
```

<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

##### img标签

英文:imge(图片)

单标签，文本级标签

作用：在指定位置插入一张图片

##### img标签的属性

##### src：作用是引入图片的路径

##### alt：图片加载不出来时候的替换文本

##### width:设置图片的宽度

##### height：设置图片的高度

一般单独设置一项就可以了，因为它会等比例缩放，不然会容易变形（width，height）

##### title:设置鼠标移到图片上的悬停文本

##### border:作用是给图片添加边框

图片的border属性了解就可以，真正加边框是用css实现，因为边框不可能只有黑色

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <img src="C:\Users\lkb\Documents\Tencent Files\3058886310\Image\Group2\$5\80\$580_CG6RRN}C}LGS27GXRG.jpg" alt="表情包" title="悬停文本" width="220px" height="20px" border="10">
</body>
</html>
```



#### 锚点

英文ancher（锚）

双标签，文本级标签

作用：在指定位置添加一个超级链接，从而实现相应的跳转

a标签有几个属性，是给超级链接添加相应的作用

href:英文hypertext reference(超文本引用)

```html
<a href="http://www.baidu.com">跳转到百度</a>
#绝对路径
<a href="1_跳转到该网址.html">跳转到1文件</a>
#相对路径
#上面的都是在当前页面加载
```



target:作用时是否在新标签打开链接，值一定是"_blank"

```html
<a href="1_跳转到该网址.html" target="_blank">跳转到1文件</a>
#加下划线是避免其他框架如iframe框架的冲突
```



















