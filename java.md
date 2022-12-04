# java



# 面向对象编程概念

如果您以前从未使用面向对象的编程语言，则需要先学习一些基本概念，然后才能开始编写任何代码。本课将向您介绍对象，类，继承，接口和包。每个讨论集中在这些概念如何与现实世界的关联，同时提供对 Java 编程语言语法的介绍。

- [什么是对象？](https://zq99299.github.io/java-tutorial/java/concepts/obgect.html)

  对象是相关状态和行为的软件包。软件对象通常用于对在日常生活中发现的真实世界对象进行建模。本课讲解了一个对象中的状态和行为是如何表现的，介绍了数据封装的概念，并以这种方式解释了设计软件的好处。

- [什么是类？](https://zq99299.github.io/java-tutorial/java/concepts/class.html)

  类是创建对象的蓝图或原型。本节定义了一个类，用于对现实世界对象的状态和行为进行建模。它有意地集中在基础上，显示一个简单的类甚至可以干净地模拟状态和行为。

- [什么是继承？](https://zq99299.github.io/java-tutorial/java/concepts/inheritance.html)

  继承为组织和构建软件提供了强大而自然的机制。本节介绍了类如何从其超类继承状态和行为，并解释了如何使用Java编程语言提供的简单语法从一个类派生另一个类。

- [什么是接口？](https://zq99299.github.io/java-tutorial/java/concepts/interface.html)

  接口是一个类和外部世界之间的合同。当一个类实现一个接口时，它承诺提供该接口发布的行为。本节定义了一个简单的接口，并解释了实现它的任何类的必要更改。

- [什么是包？](https://zq99299.github.io/java-tutorial/java/concepts/package.html)

  包是用于以逻辑方式组织类和接口的命名空间。将代码放入软件包可使大型软件项目更易于管理。本节介绍了为什么这是有用的，并将介绍给 Java 平台提供的应用程序编程接口（API）。

- [问题与练习：面向对象编程概念](https://zq99299.github.io/java-tutorial/java/concepts/qande.html)

  使用本节中提出的问题和练习来测试对对象，类，继承，接口和包的理解。































#### JDK LTS长期支持

![image-20220916174645562](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183631604-1814074230.png)

##  java跨平台原理

#### 高级语言的编译方式

- c/c++

![image-20220831120539443](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183633100-799321550.png)

整个文件一起翻译

-  python

![image-20220831120728012](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183634515-1135599408.png)

按行翻译

-  java

![image-20220831120947054](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183636427-650024614.png)

![image-20220831121132159](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183637498-361748648.png)

## JDK和JRE

JRE：java的运行环境(不需要编写)

JDK：java的开发工具包

***

#### 	JDK组成

- JVM虚拟机：Java程序运行的地方
- 核心类库：Java已经写好的东西，我们就可以直接用
- 开发工具：javac、java、jdb、jhat......

javac编译工具

java运行工具

jdb调试工具

jhat内存分析工具

- 先通过javac编译工具进行翻译，然后再通过java执行工具执行

![image-20220916200533874](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183638559-815048453.png)



#### 	JRE组成

JVM、核心类库、运行工具

- JDK包含了JRE
- JRE包含了JVM

## IDEA的使用

### IDEA项目结构介绍

- project(项目)
- moudel(模块)
- package(包)
- class(类)

![image-20220916203555771](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183639534-1339119554.png)



![image-20220916222029377](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183641270-1568733368.png)

先创建空project-->moudel-->package-->class-->编写代码-->编译运行

### IDEA的结构

- project-module-package-class
- project中可以创建多个moudel
- moudel中可以创建多个package
- package中可以创建多个class



### IDEA常用快捷键

| 快捷键               | 功能效果               |
| -------------------- | ---------------------- |
| main/psvm、sout、... | 快捷键入相关代码       |
| ctrl+D               | 复制当前行数据到下一行 |
| ctrl+X               | 删除所在行             |
| ctrl +ALT+L          | 格式化代码             |
| ALT +SHIT +上/下     | 上下移动当前代码       |
| ctrl+/ ,ctrl+shift+/ | 对代码进行注释         |

### IDEA导入模块



![image-20220916231540031](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183641768-1531085729.png)

![image-20220916231615386](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183642765-1853235870.png)

1. 选中iml导入
2. 自己建模块然后复制src里的代码粘贴到新建的模块





### 打开工程

![image-20220916232436915](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183643304-1138841977.png)





## 注释

1. 单行注释

   ```java
   // 。。。。
   ```

   

2. 多行注释

   ```java
   /*  。。。。*/
   ```

   

3. 文档注释 

   ```java
   /** 。。。。*/
   ```



## 基础语法



### 循环

#### 计算阶乘

##### **1.使用递归方式实现**

```java
public static int recursion(int num){//利用递归计算阶乘

        int sum=1;

        if(num < 0)

            throw new IllegalArgumentException("必须为正整数!");//抛出不合理参数异常

        if(num==1){
            return 1;//根据条件,跳出循环

        }else{
            sum=num * recursion(num-1);//运用递归计算

            return sum;

        }

    }
```

##### **2.使用循环方式实现**

```java
public class TextFactorial {//操作计算阶乘的类

    public static int simpleCircle(int num){//简单的循环计算的阶乘

        int sum=1;

        if(num<0){//判断传入数是否为负数

            throw new IllegalArgumentException("必须为正整数!");//抛出不合理参数异常

        }

        for(int i=1;i<=num;i++){//循环num

            sum *= i;//每循环一次进行乘法运算

        }

        return sum;//返回阶乘的值

    }
```

##### **3.利用数组添加计算**

```java
public static long addArray(int num){//数组添加计算阶乘

        long[]arr=new long[21];//创建数组

        arr[0]=1;

         

        int last=0;

        if(num>=arr.length){
            throw new IllegalArgumentException("传入的值太大");//抛出传入的数太大异常

        }

        if(num < 0)

            throw new IllegalArgumentException("必须为正整数!");//抛出不合理参数异常

        while(last<num){//建立满足小于传入数的while循环

            arr[last+1]=arr[last]*(last+1);//进行运算

            last++;//last先进行运算，再将last的值加1

        }

        return  arr[num];

    }
```

##### **4.利用BigInteger类计算**

```java
public static synchronized BigInteger bigNumber(int num){//利用BigInteger类计算阶乘

 

            ArrayList list = new ArrayList();//创建集合数组

            list.add(BigInteger.valueOf(1));//往数组里添加一个数值

            for (int i = list.size(); i <= num; i++) {
                BigInteger lastfact = (BigInteger) list.get(i - 1);//获得第一个元素

                BigInteger nextfact = lastfact.multiply(BigInteger.valueOf(i));//获得下一个数组

                list.add(nextfact);

            }

            return (BigInteger) list.get(num);//返回数组中的下标为num的值

 

    }
```

##### 主函数入口：

```java
public static void main(String []args){//java程序的主入口处

        int num=5;

        int num1=23;

        System.out.println("简单的循环计算"+num+"的阶乘为"//调用simpleCircle

                +simpleCircle(num));

        System.out.println("利用递归计算"+num+"的阶乘为"//调用recursion

                +recursion(num));

        System.out.println("数组添加计算"+num+"的阶乘为"//调用addArray

                +addArray(num));

        System.out.println("利用BigInteger类计算"+num1+"的阶乘为"//调用bigNumber

                +bigNumber(num1));

         

    }

}
```



##### 阶乘求和

```c
#include<stdio.h>

int main()
{
	int i,j;
	long sum=0; //阶乘求和 
	long a=1; //阶乘 
	for(i=1;i<=15;i++)
	{
		a=1; //阶乘初始化为1
		for(j=1;j<=i;j++)
		{
			a=a*j;
		}
		sum=sum+a; //每轮阶乘求和
	}
	printf("1!+2!+3!+...+15!=%d",sum);
	
	return 0;
}
```



### 数组内存

#### 栈

![image-20221203215412283](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183643901-1055071363.png)

#### 堆



#### 方法区

![image-20221203215351063](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183644405-237441641.png)







## 杂项



### 1、java中的＜＜符号是什么意思

这是一种移位运算符 在二进制下进行移位

a<<i 其中a前为要移动的数，i为要移动的位数

例如：

3<<1

是将3先转化为24位的二进制

0000 0000 0000 0000 0000 0000 0000 0011

然后再左移一位 最后结果为 0000 0000 0000 0000 0000 0000 0000 0110

再转化为十进制结果为6

如果左移过程中超过了32位 高位就会舍弃 低位补零

```
>>是右移 和 <<具有同样的道理
```

只不过右移过程中溢出时，低位会舍弃，高位补零

**用最有效率的方法算出2乘以8等于几？**

```java
package javaStudy.array;

public class test {
    public static void main(String[] args) {
        int age=2;
        int age2=3;
        int a =age<<age2;
        System.out.println(a);
    }
}
```





