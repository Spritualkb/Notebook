# 时间的表达

![image-20220413104046771](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184235118-475600563.png)

import time

time.time()

b=int(time.time())

# 绘制折现图

import turtle

import.math

#定义多个点的坐标

x1,y1 = 100,100

x2,y2 = 100,-100

x3,y3 = -100,-100

x4,y4 = -100,100



#绘制折线

turtle.penup()

turtle.goto(x1,y1)

turtle.penup()

turtle.goto(x2,y2)

turtle.goto(x3,y3)

turtle.goto(x4,y4)



#计算起始点和终点的距离



distance = math.sqrt((x1-x4)* * 2+(y1=y4)* * 2)

turtle.write(distance)



####  \  行连接符



# 对象

**python中，一切皆对象**。每个对象由标识(identity)、类型(type)、value(值)组成

**对象的本质就是：一个内存条，拥有特定的值，支持特定类型的相关操作。**

![image-20220413210353123](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184253445-377164960.png)



a就像是洗衣机的标签标签，而3就像洗衣机



# 引用

在python中，变量也成为：对象的引用。因为，变量存储的就是对象的地址。

变量通过地址引用了"对象"。



变量位于：栈内存(压栈出栈等)

对象位于：栈内存。

![image-20220413211042272](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184254729-826491358.png)

![image-20220413211201217](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184255445-677893838.png)



# 图形化程序设计

```python
import turtle   #导入turtle模块

turtle.showturle()      #显示箭头

turtle.write("高琪")    #写字符串

turtle.forward(300)   #前进300像素

turtle.color("red")    #画笔颜色改为red

turtle.left(90)    #箭头左转90度

turtle.goto(0,50)  #去坐标(0,50)

turtle.penup()        #抬笔。这样，路径不会画出来

turtle.pendown()        #下笔。这样，路径会画出来

turtle.circle(100)      #画圆，半径为100


```



# 字符串

## 字符串的编码

python3会直接支持Unicode,可以表示世界上任何书面语言的字符。python3的字符默认就是16位Unicode编码，ASCII码是Unicode编码的自己。

使用内置函数ord()可以把字符转换成对应的Unicode码；

使用内置函数chr()可以把十进制数字转换成对应的字符。

![image-20220413212859364](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184255881-1552050303.png)

![image-20220413213210859](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184257692-173080471.png)

字符串的切割

split()分割和join()合并

![image-20220413213636895](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184258867-579142095.png)

![image-20220413213659628](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184259327-1481847245.png)

# 字符串主流机制和字符串比较

字符串驻留：仅保存一份相同且不可变字符串的方法，不同的值被存放在字符串驻留池中。python支持字符串驻留机制，对于符合标识符规则的字符串(仅包含下划线(_)、字母和数字)会启用字符串逐鹿机制驻留机制。

![image-20220413215931395](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184259907-7405161.png)

is #比较的是两个对象的id

==   #比较的是两个对象的value



## 常用查找方法

len(a)   #字符串长度

a.startswith("我是谁")  #以指定字符串开头      结果：True

a.endswith("me")    #以指定字符串结尾       结果：Ture

a.find('高')        #第一次出现指定字符串的位置     结果:2

a.rfind("高")       #最后一次出现指定字符串的位置     结果：29

a.count("编程")   #指定字符串出现了几次   结果：3

a.isalnum()       #所有字符串全是字母或数字    结果：False

## 去除首尾信息

![image-20220413220941975](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184300796-1095965092.png)

## 大小写转换

![image-20220413221115175](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184302862-548930605.png)

## 格式排版

![image-20220413221154926](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184303696-1024066263.png)

## another‘

![image-20220413221310040](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184305535-2132119092.png)





| **转义字符**            | **描述**   |
| ----------------------- | ---------- |
| \ (在行尾时) | 续行符     |
| \ \                 | 反斜杠符号 |
| \ '                 | 单引号     |
| \ "    | 双引号                                   |
| **\a**     | 警告声                                   |
| **\b**     | 退格                                     |
| **\e**     | 转义                                     |
| **\000**   | 空                                       |
| **\n**     | 换行                                     |
| **\v**     | 纵向制表符                               |
| **\t**     | 横向制表符                               |
| **\r**     | 回车                                     |
| **\t**     | 水平制表符                               |
| **\f**     | 换页                                     |
| **\oyy**   | 八进制数yy代表的字符，例如：\o12代表换行 |
| **\xyy**   | 十进制数yy代表的字符，例如：\x0a代表换行 |
| **\other** | 其它的字符以普通格式输出                 |





| **符号**  | **作用**                                                     |
| --------- | ------------------------------------------------------------ |
| *****     | 定义宽度或者小数点精度                                       |
| **-**     | 用做左对齐                                                   |
| **+**     | 在正数前面显示加号( + )                                      |
| (空格键)  | 在正数前面显示空格                                           |
| **#**     | 在八进制数前面显示零(‘0’)，在十六进制前面显示’0x’或者’0X’(取决于用的是’x’还是’X’) |
| **0**     | 显示的数字前面填充‘0’而不是默认的空格                        |
| **%**     | ‘%%’输出一个单一的’%’                                        |
| **(var)** | 映射变量(字典参数)                                           |
| **m.n**   | m 是显示的最小总宽度,n是小数点后的位数(如果可用的话)         |

其中比较有用的是m.n，它们可以控制输出浮点数和整数的总宽度以及浮点数的小数精度



```python
【例5.4】数字字符转换。
print("%x" % 123)
'7b'
print("%X" % 123)
'7B’
print("%#X" % 123)
'0X7B'
print("%#x" % 123)
'0x7b'
print('%f' % 1234.567890 )
'1234.567890' 
print('%.2f' % 1234.567890 )
'1234.57'  
print('%E' % 1234.567890 )
'1.234568E+03' 
print('%e' % 1234.567890 )
'1.234568e+03' 
print('%g' % 1234.567890 )
'1234.57'
print('%G' % 1234.567890 )
'1234.57' 
print("%e" % (1111111111111111111111L) )
'1.111111e+21'
print("%22.10e" % (1111111111111111111111))
1.1111111111e+21

```



| **符号**  | **作用**                                                     |
| --------- | ------------------------------------------------------------ |
| *****     | 定义宽度或者小数点精度                                       |
| **-**     | 用做左对齐                                                   |
| **+**     | 在正数前面显示加号( + )                                      |
| (空格键)  | 在正数前面显示空格                                           |
| **#**     | 在八进制数前面显示零(‘0’)，在十六进制前面显示’0x’或者’0X’(取决于用的是’x’还是’X’) |
| **0**     | 显示的数字前面填充‘0’而不是默认的空格                        |
| **%**     | ‘%%’输出一个单一的’%’                                        |
| **(var)** | 映射变量(字典参数)                                           |
| **m.n**   | m 是显示的最小总宽度,n是小数点后的位数(如果可用的话)         |





##  数字字符转换方式表

| **格式化字符**  | **转换方式**                                 |
| --------------- | -------------------------------------------- |
| **%c**          | 转换成字符(ASCII 码值，或者长度为一的字符串) |
| **%r**          | 优先用 repr()函数进行字符串转换              |
| **%s**          | 优先用 str()函数进行字符串转换               |
| %d / %i | 转成有符号十进制数                           |
| **%u**          | 转成无符号十进制数                           |
| **%o**    | **转成无符号八进制数**                              |
| **%x/%X** | (Unsigned)转成无符号十六进制数(x/X代表转换后的十六进制字符的大小写) |
| **%e/%E** | 转成科学计数法(e/E 控制输出 e/E)                          |
| **%f/%F** | 转成浮点数(小数部分自然截断)                               |
| **%g/%G** | 转为浮点数，根据值的大小采用%e或%f格式                       |
| **%%**    | 输出%                                                  |

## 时间格式化

日期和时间格式化表：

| **格式化字符** | **转换方式**                           |
| -------------- | -------------------------------------- |
| **%a**         | 星期几的简写                           |
| **%A**         | 星期几的全称                           |
| **%b**         | 月分的简写                             |
| **%B**         | 月份的全称                             |
| **%c**         | 标准的日期的时间串                     |
| **%C**         | 年份的后两位数字                       |
| **%d**         | 十进制表示的每月的第几天               |
| **%D**         | 月/天/年                               |
| **%e**         | 在两字符域中，十进制表示的每月的第几天 |
| **%F**         | 年-月-日                               |
| **%g**         | 年份的后两位数字，使用基于周的年       |
| **%G**         | 年分，使用基于周的年                   |
| **%h**         | 简写的月份名                           |
| **%H**         | 24小时制的小时                         |
| **%I**         | 12小时制的小时                         |
| **%j**         | 十进制表示的每年的第几天               |

```python
print("__________________字符串定义和输出___________________")
s1='hello,my name is"毛小花"'
s2="hello，我叫'alen'"
s3='''我们做好朋友吧\
你同意吗？\
我同意'''
print(s1)
print(s2)
print(s3)
print("__________________字符串长度获取___________________")
x=len(s1)
print("S1的长度为%d"%x)
print("__________________字符串统计字符个数___________________")
s="abcddbacdfr"
print('"d"出现的次数为%d'%s.count('d'))
print("__________________字符串去重复___________________")
s="abcddbacdfr"
qu=set(s)
print("去重后的字符串：",str(qu))
print("__________________转义字符串输出___________________")
print('c:\windows\newfolder')
print(r'c:\windows\newfolder')
print("__________________字符串连接与复制___________________")
print('hello' + 'world')
print(2 * 'hello' + 'world')
print("__________________数字字符格式输出___________________")
print("%x" % 123)
print("%X" % 123)
print("%#X" % 123)
print("%#x" % 123)
print('%f' % 1234.567890 )
print('%.2f' % 1234.567890 )
print('%E' % 1234.567890 )
print('%e' % 1234.567890 )
print('%g' % 1234.567890 )
print('%G' % 1234.567890 )
print("%e" % (1111111111111111111111))
print("%22.10e" % (1111111111111111111111))
print("__________________日期格式化输出___________________")
import time
time1=time.time()
print("当前时间为：",time1)
# 格式化成2016-03-20 11:45:39形式
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# 格式化成Sat Mar 28 22:24:24 2016形式
print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
# 将格式字符串转换为时间戳
a = "Sat Mar 28 22:24:24 2016"
print(time.mktime(time.strptime(a, "%a %b %d %H:%M:%S %Y")))
# 日历的输出
import calendar
cal = calendar.month(2022, 4)
print("2022年4月日历:")
print(cal)
print("__________________字符串切片___________________")
s4='2000-12-20'
print(s4[0:4],"年",s4[5:7],"月",s4[8:],"日")
print("__________________字符串分割后存为列表___________________")
s5="10001 洗衣服 日用品 12"
newli=s5.split(" ")
print(newli)
while "" in newli: #移除列表中的空数据
    newli.remove("")
print(newli)
print("__________________常见字符串操作___________________")
word = 'hello,world'
print(len(word))  #获取长度
print(word.endswith('world'))  #是否以某串结尾
print(word.endswith('world.'))
print(word.startswith('hello'))  #是否以某串开始
print(word.startswith('Hello'))
print(word.capitalize())  #把字符串第一个字符大写
print(word.find('ello'))  #在字符串中匹配子串，匹配到返回匹配到的索引位置，匹配不到返回-1
print(word.find('abcd'))
print(word.find('hello'))
print('ello' in word) # 字符串中子串的判断
print('abcd' in word)
print(word.index("h")) #同find一样，只是找不到的情况下会报异常
print('—————————————————————————————————————')
word = 'hello,world'
print(word.isalpha()) #判断是不是全字母
word = 'helloworld'
print( word.isalpha())
number = '1234.5678'
print( number.isdigit()) #判断是不是全数字
number = '12345678'
print( number.isdigit())
print('—————————————————————————————————————————————————  ')
l = ['1', '2', '3']
print(l)
print(','.join(l)) #以，分割合为一个串
word = 'hello'
print( word.replace('llo', 'llo world')) #用后面的串替换前面的串
l = '1, 2, 3'
print(l.split(','))  #以，分割获取每一个元素，最后得到一个列表
word = '   hello world   '
print(word.strip()) #去掉空字符

```

## 字符串函数

| **方法**                                                     | **描述**                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [string.capitalize()](https://www.runoob.com/python/att-string-capitalize.html) | 把字符串的第一个字符大写                                     |
| [string.center(width)](https://www.runoob.com/python/att-string-center.html) | 返回一个原字符串居中,并使用空格填充至长度 width 的新字符串   |
| **[string.count(str, beg=0, end=len(string))](https://www.runoob.com/python/att-string-count.html)** | 返回 str 在  string 里面出现的次数，如果 beg 或者 end 指定则返回指定范围内 str 出现的次数 |
| [string.decode(encoding='UTF-8',   errors='strict')](https://www.runoob.com/python/att-string-decode.html) | 以 encoding 指定的编码格式解码 string，如果出错默认报一个 ValueError 的 异 常 ，  除非 errors 指 定 的 是 'ignore' 或 者'replace' |
| [string.encode(encoding='UTF-8',   errors='strict')](https://www.runoob.com/python/att-string-encode.html) | 以 encoding 指定的编码格式编码 string，如果出错默认报一个ValueError 的异常，除非 errors 指定的是'ignore'或者'replace' |
| **[string.endswith(obj, beg=0, end=len(string))](https://www.runoob.com/python/att-string-endswith.html)** | 检查字符串是否以 obj 结束，如果beg 或者 end 指定则检查指定的范围内是否以 obj 结束，如果是，返回 True,否则返回 False. |
| [string.expandtabs(tabsize=8)](https://www.runoob.com/python/att-string-expandtabs.html) | 把字符串 string 中的 tab 符号转为空格，tab 符号默认的空格数是 8。 |
| **[string.find(str, beg=0, end=len(string))](https://www.runoob.com/python/att-string-find.html)** | 检测 str 是否包含在 string 中，如果 beg 和  end 指定范围，则检查是否包含在指定范围内，如果是返回开始的索引值，否则返回-1 |
| **[string.format()](https://www.runoob.com/python/att-string-format.html)** | 格式化字符串                                                 |
| **[string.index(str, beg=0, end=len(string))](https://www.runoob.com/python/att-string-index.html)** | 跟find()方法一样，只不过如果str不在 string中会报一个异常.    |
| [string.isalnum()](https://www.runoob.com/python/att-string-isalnum.html) | 如果 string 至少有一个字符并且所有字符都是字母或数字则返回 True,否则返回 False |
| [string.isalpha()](https://www.runoob.com/python/att-string-isalpha.html) | 如果 string 至少有一个字符并且所有字符都是字母则返回 True,否则返回 False |
| [string.isdecimal()](https://www.runoob.com/python/att-string-isdecimal.html) | 如果 string 只包含十进制数字则返回 True 否则返回 False.      |
| [string.isdigit()](https://www.runoob.com/python/att-string-isdigit.html) | 如果 string 只包含数字则返回 True 否则返回 False.            |
| [string.islower()](https://www.runoob.com/python/att-string-islower.html) | 如果 string 中包含至少一个区分大小写的字符，并且所有这些(区分大小写的)字符都是小写，则返回  True，否则返回 False |
| [string.isnumeric()](https://www.runoob.com/python/att-string-isnumeric.html) | 如果 string 中只包含数字字符，则返回 True，否则返回 False    |
| [string.isspace()](https://www.runoob.com/python/att-string-isspace.html) | 如果 string 中只包含空格，则返回 True，否则返回 False.       |
| [string.istitle()](https://www.runoob.com/python/att-string-istitle.html) | 如果 string 是标题化的(见 title())则返回  True，否则返回 False |
| [string.isupper()](https://www.runoob.com/python/att-string-isupper.html) | 如果 string 中包含至少一个区分大小写的字符，并且所有这些(区分大小写的)字符都是大写，则返回  True，否则返回 False |
| **[string.join(seq)](https://www.runoob.com/python/att-string-join.html)** | 以 string 作为分隔符，将 seq 中所有的元素(的字符串表示)合并为一个新的字符串 |
| [string.ljust(width)](https://www.runoob.com/python/att-string-ljust.html) | 返回一个原字符串左对齐,并使用空格填充至长度 width 的新字符串 |
| [string.lower()](https://www.runoob.com/python/att-string-lower.html) | 转换 string 中所有大写字符为小写.                            |
| [string.lstrip()](https://www.runoob.com/python/att-string-lstrip.html) | 截掉 string 左边的空格                                       |
| [string.maketrans(intab,   outtab)](https://www.runoob.com/python/att-string-maketrans.html) | maketrans() 方法用于创建字符映射的转换表，对于接受两个参数的最简单的调用方式，第一个参数是字符串，表示需要转换的字符，第二个参数也是字符串表示转换的目标。 |
| [max(str)](https://www.runoob.com/python/att-string-max.html) | 返回字符串 *str* 中最大的字母。                              |
| [min(str)](https://www.runoob.com/python/att-string-min.html) | 返回字符串 *str* 中最小的字母。                              |
| **[string.partition(str)](https://www.runoob.com/python/att-string-partition.html)** | 有点像 find()和  split()的结合体,从 str 出现的第一个位置起,把 字 符 串 string 分 成 一 个 3 元 素 的 元 组 (string_pre_str,str,string_post_str),如果 string 中不包含str 则  string_pre_str == string. |
| **[string.replace(str1, str2,  num=string.count(str1))](https://www.runoob.com/python/att-string-replace.html)** | 把 string 中的  str1 替换成 str2,如果 num 指定，则替换不超过 num 次. |
| [string.rfind(str,   beg=0,end=len(string) )](https://www.runoob.com/python/att-string-rfind.html) | 类似于 find() 函数，返回字符串最后一次出现的位置，如果没有匹配项则返回 -1。 |
| [string.rindex(   str, beg=0,end=len(string))](https://www.runoob.com/python/att-string-rindex.html) | 类似于 index()，不过是返回最后一个匹配到的子字符串的索引号。 |
| [string.rjust(width)](https://www.runoob.com/python/att-string-rjust.html) | 返回一个原字符串右对齐,并使用空格填充至长度 width 的新字符串 |
| [string.rpartition(str)](https://www.runoob.com/python/att-string-rpartition.html) | 类似于 partition()函数,不过是从右边开始查找                  |
| [string.rstrip()](https://www.runoob.com/python/att-string-rstrip.html) | 删除 string 字符串末尾的空格.                                |
| **[string.split(str="", num=string.count(str))](https://www.runoob.com/python/att-string-split.html)** | 以 str 为分隔符切片 string，如果 num 有指定值，则仅分隔 **num+1** 个子字符串 |
| [string.splitlines([keepends\])](https://www.runoob.com/python/att-string-splitlines.html) | 按照行('\r', '\r\n', '\n')分隔，返回一个包含各行作为元素的列表，如果参数 keepends 为 False，不包含换行符，如果为 True，则保留换行符。 |
| [string.startswith(obj,   beg=0,end=len(string))](https://www.runoob.com/python/att-string-startswith.html) | 检查字符串是否是以 obj 开头，是则返回 True，否则返回 False。如果beg 和 end 指定值，则在指定范围内检查. |
| **[string.strip([obj\])](https://www.runoob.com/python/att-string-strip.html)** | 在 string 上执行 lstrip()和 rstrip()                         |
| [string.swapcase()](https://www.runoob.com/python/att-string-swapcase.html) | 翻转 string 中的大小写                                       |
| [string.title()](https://www.runoob.com/python/att-string-title.html) | 返回"标题化"的 string,就是说所有单词都是以大写开始，其余字母均为小写(见 istitle()) |
| **[string.translate(str, del="")](https://www.runoob.com/python/att-string-translate.html)** | 根据 str 给出的表(包含 256 个字符)转换 string  的字符,  要过滤掉的字符放到 del 参数中 |
| [string.upper()](https://www.runoob.com/python/att-string-upper.html) | 转换 string 中的小写字母为大写                               |
| [string.zfill(width)](https://www.runoob.com/python/att-string-zfill.html) | 返回长度为 width 的字符串，原字符串 string 右对齐，前面填充0 |

##  正则表达式

一般对于邮箱，网址之类的要做正则处理

```python
print("__________________正则表达式___________________")
print("方法一：先将正则表达式模式编译成一个正则表达式对象，再匹配的正则表达式，返回匹配到的对象，只匹配一个")
import re
string = "123456789@email.com"
pattern = r'\d+@email.com'
prog = re.compile(pattern)
result = prog.match(string)
print(result)
print("方法二：先将正则表达式模式编译成一个正则表达式对象，再搜索查找与正则表达式匹配字符串，匹配到第一个位置")
import re
string = """
hello's email:123456789@qq.com
world's email:987654321@email.com
"""
pattern =r'\d+@email.com'
prog = re.compile(pattern)
result = prog.search(string)
print(result)
print("方法三：先将正则表达式模式编译成一个正则表达式对象，再用findall匹配到正则表达式的非重复子字符串，并返回列表")
import re
string = """
hello's email:123456789@email.com
world's email:987654321@email.com
"""
pattern = r'\d+@email.com'
prog = re.compile(pattern)
result = prog.findall(string)
print(result)
print("方法四：先将正则表达式模式编译成一个正则表达式对象，再用finditer匹配到正则表达式的非重复子字符串，并返回迭代器")
import re
string = """
hello's email:123456789@email.com
world's email:987654321@email.com
"""
pattern = r'\d+@email.com'
result = prog.finditer(string)
print(result)
print("方法五：正则表达式替换，re.sub(pattern, repl, string, count=0, flags=0)使用repl替换所有在string中与pattern相匹配的子字符串")
import re
string = """
hello's email:123456789@email.com
world's email:987654321@email.com
"""
pattern = r'\d+@email.com'
prog = re.compile(pattern)
string = re.sub(prog, "none", string)
print(string)

print("__________________利用正则表达式对输入的邮箱地址进行校验___________________")
import re
text = input("please input your email address：\n")
if re.match(r'^[0-9a-za-z_]{0,19}@[0-9a-za-z]{1,13}\.[com,cn,net]{1,3}$',text):
  print('email address is right!')
else:
  print('please reset your right email address!')



```

## 面向对象

```python
'''
class Student: #Student为类的名称（类名）由一个或多个单词组成，每个单词的首字母大写，其余小写
    pass

#Python中一切皆对象Student也是对象
print(id(Student))  #1967336464112
print(type(Student)) #<class 'type'>
print(Student)  #<class '__main__.Student'>
'''
class Student:  #Student为类的名称（类名）由一个或多个单词组成，每一个或多个单词组成，每个单词的首字母大写，其余小写
    
    native_pace='江门'  #直接写在类里的变量，称为类属性
    def __init__(self,name,age):
        self.name=name  #self.name 称为实例属性，进行了一个赋值的操作，将局部变量的name的值赋值给实体属性
        self.age=age

    #实例方法
    def eat(self):
        print('学生在吃饭。。。')

    #静态方法
    @staticmethod
    def method():
        print('我使用了staticmethod进行修饰，所以我是静态方法')
        
        #类方法
    @classmethod
    def cm(cls):
        print('我是类方法，因为我使用了classmethod进行修饰')

#在类之外定义的称为函数，在类之内定义的称为方法
def drink():
    print('喝水')
#创建Student类的对象
stu1=Student('张三',20)
print(id(stu1))
print(type(stu1))
print(stu1)
print('-'*20)
print(id(Student))
print(type(Student))
print(Student)
```

![image-20220531220452554](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184306240-1205216996.png)

- 对象的创建又称为类的实例化

- 语法：  实例名=类名()

- 例子：

- ```python
  #创建Student类的实例对象
  stu=Student('Jack',20)
  print(stu,name)  #实例属性
  print(stu,age) #实例属性
  stu.info()
  ```

- 意义：有了实例对象，就可以调用类中的内容

- Python 类方法和实例方法相似，它最少也要包含一个参数，只不过类方法中通常将其命名为 cls，Python 会自动将类本身绑定给 cls 参数（注意，绑定的不是类对象）。也就是说，我们在调用类方法时，无需显式为 cls 参数传参。类方法推荐使用类名直接调用，当然也可以使用实例对象来调用（不推荐）。
- 静态方法，其实就是我们学过的函数，和函数唯一的区别是，静态方法定义在类这个空间（类命名空间）中，而函数则定义在程序所在的空间（全局命名空间）中。
- 静态方法没有类似 self、cls 这样的特殊参数，因此 Python 解释器不会对它包含的参数做任何类或对象的绑定。也正因为如此，类的静态方法中无法调用任何类属性和类方法。
- 静态方法需要使用`＠staticmethod`修饰。静态方法的调用，既可以使用类名，也可以使用类对象
- 在实际编程中，几乎不会用到类方法和静态方法，因为我们完全可以使用函数代替它们实现想要的功能，但在一些特殊的场景中（例如工厂模式中），使用类方法和静态方法也是很不错的选择。

### 类属性 _ 类方法_  静态方法的使用方式

- 类属性：类中方法外的变量称为类属性，被该类的所有对象所共享
- 类方法：使用@classmethod修饰的方法，使用类名直接访问的方法
- 静态方法：使用@staticmethod修饰的主法，使用类名直接访问的方法

![image-20220531230422539](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184306750-1855939270.png)

1. 这两个实例对象，会有一个类指针指向Student这个类对象，直接调取那个类属性也就是江门，也就是说这个值是被他们两个实例对象所共享的。

   ![image-20220531231917318](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184307251-916329708.png)

### 排序

```python
student_new.sort(key=lambda x:int(x['english']),reverse=asc_or_desc_bool)
#x（可以随便写）是一个参数，这个参数是一个字典（或者啥都行），在这个字典当中根据这个键获取它的值，然后进行int类型转换，最后将结果赋值给key
#匿名函数用的是lambda 就是把x这个列表传进来
```

```python
def sort():
    show()
    if os.path.exists(filename):
        with open(filename,'r',encoding='utf-8') as rfile:
            student_list=rfile.readlines()
        student_new=[]
        for item in student_list:
            d=dict(eval(item))#把学生信息遍历然后把字符串转为字典当中去
            student_new.append(d)#把字典添加到学生列表当中


    else:
        return
    asc_or_desc=input('请选择（0.升序 1.降序）:')
    if asc_or_desc=='0':
        asc_or_desc_bool=False
    elif asc_or_desc=='1':
        asc_or_desc_bool=True
    else:
        print('您的输入有误，请重新输入')
        sort()
    mode=input('请选择排序方式（1.按英语成绩排序 2.按Python成绩排序 3.按数学成绩排序 0.按总成绩排序）:')
    if mode=='1':
        student_new.sort(key=lambda x:int(x['english']),reverse=asc_or_desc_bool)
    elif mode=='2':
        student_new.sort(key=lambda x:int(x['python']),reverse=asc_or_desc_bool)
    elif mode=='3':
        student_new.sort(key=lambda x:int(x['math']),reverse=asc_or_desc_bool)
    elif mode=='0':
        student_new.sort(key=lambda x:int(x['english'])+int(x['python'])+int(x['math']),reverse=asc_or_desc_bool)
    else:
        print('您的输入有误，请重新输入！！！')
        sort()
    show_student(student_new)
```

