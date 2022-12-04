# MySQL  账号/密码root 端口：3306

## 数据库基本概念

### 【1】数据库基本概念

#### （1）数据

所谓数据（Data）是指对客观事物进行描述并可以鉴别的符号，这些符号是可识别的、抽象的。它不仅仅指狭义上的数字，而是有多种表现形式：字母、文字、文本、图形、音频、视频等。现在计算机存储和处理的数据范围十分广泛，而描述这些数据的符号也变得越来越复杂了。

#### （2）数据库

数据库（Database，DB）指的是以一定格式存放、能够实现多个用户共享、与应用程序彼此独立的数据集合。

#### （3）数据库管理系统

数据库管理系统（Database Management System，DBMS）是用来定义和管理数据的软件。如何科学的组织和存储数据，如何高效的获取和维护数据，如何保证数据的安全性和完整性，这些都需要靠数据库管理系统完成。目前，比较流行的数据库管理系统有：Oracle、MySQL、SQL Server、DB2等。

#### （4）数据库应用程序

数据库应用程序（Database Application System，DBAS）是在数据库管理系统基础上，使用数据库管理系统的语法，开发的直接面对最终用户的应用程序，如学生管理系统、人事管理系统、图书管理系统等。

#### （5）数据库管理员

数据库管理员（Database Administrator，DBA）是指对数据库管理系统进行操作的人员，其主要负责数据库的运营和维护。

#### （6）最终用户

最终用户（User）指的是数据库应用程序的使用者。用户面向的是数据库应用程序（通过应用程序操作数据），并不会直接与数据库打交道。

#### （7） 数据库系统

数据库系统（Database System，DBS）一般是由数据库、数据库管理系统、数据库应用程序、数据库管理员和最终用户构成。其中DBMS是数据库系统的基础和核心。

## 数据库类型和常见的关系型数据库

### 【1】数据库类型

数据库经过几十年的发展，出现了多种类型。根据数据的组织结构不同，主要分为网状数据库、层次数据库、关系型数据库和非关系型数据库四种。目前最常见的数据库模型主要是：关系型数据库和非关系型数据库。

#### 关系型数据库

关系型数据库模型是将复杂的数据结构用较为简单的二元关系（二维表）来表示，如图1-4所示。在该类型数据库中，对数据的操作基本上都建立在一个或多个表格上，我们可以采用结构化查询语言（SQL）对数据库进行操作。关系型数据库是目前主流的数据库技术，其中具有代表性的数据库管理系统有：Oracle、DB2、SQL Server、MySQL等。


PS：关系=二维表

#### 非关系型数据库NOSQL

NOSQL（Not Only SQL）泛指非关系型数据库。关系型数据库在超大规模和高并发的web2.0纯动态网站已经显得力不从心，暴露了很多难以克服的问题。NOSQL数据库的产生就是为了解决大规模数据集合多重数据种类带来的挑战，尤其是大数据应用难题。常见的非关系型数据库管理系统有Memcached、MongoDB，redis，HBase等。 

### 【2】常见的关系型数据库

虽然非关系型数据库的优点很多，但是由于其并不提供SQL支持、学习和使用成本较高并且无事务处理，所以本书的重点是关系型数据库。下面我们将介绍一下常用的关系型数据库管理系统。

#### Oracle

Oracle数据库是由美国的甲骨文（Oracle）公司开发的世界上第一款支持SQL语言的关系型数据库。经过多年的完善与发展，Oracle数据库已经成为世界上最流行的数据库，也是甲骨文公司的核心产品。
Oracle数据库具有很好的开放性，能在所有的主流平台上运行，并且性能高、安全性高、风险低；但是其对硬件的要求很高、管理维护和操作比较复杂而且价格昂贵，所以一般用在满足对银行、金融、保险等行业大型数据库的需求上。

#### 1、DB2

DB2是IBM公司著名的关系型数据库产品。DB2无论稳定性，安全性，恢复性等等都无可挑剔，而且从小规模到大规模的应用都可以使用，但是用起来非常繁琐，比较适合大型的分布式应用系统。

#### 2、SQL Server

SQL Server是由Microsoft开发和推广的关系型数据库，SQL Server的功能比较全面、效率高，可以作为中型企业或单位的数据库平台。SQL Server可以与Windows操作系统紧密继承，无论是应用程序开发速度还是系统事务处理运行速度，都能得到大幅度提升。但是，SQL Server只能在Windows系统下运行，毫无开放性可言。

#### 3、MySQL

MySQL是一种开放源代码的轻量级关系型数据库，MySQL数据库使用最常用的结构化查询语言（SQL）对数据库进行管理。由于MySQL是开放源代码的，因此任何人都可以在General Public License的许可下下载并根据个人需要对其缺陷进行修改。
由于MySQL数据库体积小、速度快、成本低、开放源码等优点，现已被广泛应用于互联网上的中小型网站中，并且大型网站也开始使用MySQL数据库，如网易、新浪等。  

## MySQl介绍

MySQL数据库最初是由瑞典MySQL AB公司开发，2008年1月16号被Sun公司收购。2009年，SUN又被Oracle收购。MySQL是目前IT行业最流行的开放源代码的数据库管理系统，同时它也是一个支持多线程高并发多用户的关系型数据库管理系统。MySQL之所以受到业界人士的青睐，主要是因为其具有以下几方面优点：
1. 开放源代码
    MySQL最强大的优势之一在于它是一个开放源代码的数据库管理系统。开源的特点是给予了用户根据自己需要修改DBMS的自由。MySQL采用了General Public License，这意味着授予用户阅读、修改和优化源代码的权利，这样即使是免费版的MySQL的功能也足够强大，这也是为什么MySQL越来越受欢迎的主要原因。
2. 跨平台
MySQL可以在不同的操作系统下运行，简单地说，MySQL可以支持Windows系统、UNIX系统、Linux系统等多种操作系统平台。这意味着在一个操作系统中实现的应用程序可以很方便地移植到其他的操作系统下。
3. 轻量级
MySQL的核心程序完全采用多线程编程，这些线程都是轻量级的进程，它在灵活地为用户提供服务的同时，又不会占用过多的系统资源。因此MySQL能够更快速、高效的处理数据。
4. 成本低
MySQL分为社区版和企业版，社区版是完全免费的，而企业版是收费的。即使在开发中需要用到一些付费的附加功能，价格相对于昂贵的Oracle、DB2等也是有很大优势的。其实免费的社区版也支持多种数据类型和正规的SQL查询语言，能够对数据进行各种查询、增加、删除、修改等操作，所以一般情况下社区版就可以满足开发需求了，而对数据库可靠性要求比较高的企业可以选择企业版。
另外，PHP中提供了一整套的MySQL函数，对MySQL进行了全方位的强力支持。 
总体来说，MySQL是一款开源的、免费的、轻量级的关系型数据库，其具有体积小、速度快、成本低、开放源码等优点，其发展前景是无可限量的。 

PS：社区版与企业版主要的区别是：
1. 社区版包含所有MySQL的最新功能，而企业版只包含稳定之后的功能。换句话说，社区版可以理解为是企业版的测试版。 
2.MySQL官方的支持服务只是针对企业版，如果用户在使用社区版时出现了问题，MySQL官方是不负责任的。

1)安装了Windows Service：MySQL80，并且已经启动。

![image-20220422163935451](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183110333-499848736.png)

2)安装了MySQL软件。安装位置为：C:\Program Files\MySQL。

![image-20220422163954699](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183110802-248408937.png)

**（MySQL文件下放的是软件的内容）**
3)安装了MySQL数据文件夹，用来存放MySQL基础数据和以后新增的数据。安装位置为C:\ProgramData\MySQL\MySQL Server 8.0。

**（ProgramData文件夹可能是隐藏的，显示出来即可）**
**（MySQL文件下的内容才是真正的MySQL中数据）**
4)在MySQL数据文件夹中有MySQL的配置文件：my.ini。它是MySQL数据库中使用的配置文件，修改这个文件可以达到更新配置的目的。以下几个配置项需要大家特别理解。
port=3306：监听端口是3306

1. basedir="C:/Program Files/MySQL/MySQL Server 8.0/"：软件安装位置

2. datadir=C:/ProgramData/MySQL/MySQL Server 8.0/Data：数据文件夹位置

3. default_authentication_plugin=caching_sha2_password：默认验证插件

4. default-storage-engine=INNODB：默认存储引擎


（这些内容在Linux下可能会手动更改）

## MySQL登录，访问，退出操作

### 【1】登录：

访问MySQL服务器对应的命令：mysql.exe ,位置：C:\Program Files\MySQL\MySQL Server 8.0\bin

![image-20220422164512787](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183111125-568962557.png)

（mysql.exe需要带参数执行，所以直接在图形界面下执行该命令会自动结束）

打开控制命令台：win+r:

![image-20220422164452084](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183111452-374857647.png)

执行mysql.exe命令的时候出现错误：

![image-20220422164522051](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183111760-488157870.png)

需要配置环境变量path:

![image-20220422164533066](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183112091-421392469.png)

注意：控制命令台必须重启才会生效：

登录的命令：mysql  -hlocalhost -uroot –p

1. mysql：bin目录下的文件mysql.exe。mysql是MySQL的命令行工具，是一个客户端软件，可以对任何主机的mysql服务（即后台运行的mysqld）发起连接。

2. -h：host主机名。后面跟要访问的数据库服务器的地址；如果是登录本机，可以省略

3. -u：user 用户名。后面跟登录数据的用户名，第一次安装后以root用户来登录，是MySQL的管理员用户

4. -p:   password 密码。一般不直接输入，而是回车后以保密方式输入。 

![image-20220422164741751](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183112597-2126417239.png)

### 【2】访问数据库

显示MySQL中的数据库列表：show databases; 默认有四个自带的数据库，每个数据库中可以有多个数据库表、视图等对象。

- 切换当前数据库的命令：**use mysql**;
  MySQL下可以有多个数据库，如果要访问哪个数据库，需要将其置为当前数据库。
- 该命令的作用就是将数据库mysql（默认提供的四个数据库之一的名字）置为当前数据库


显示当前数据库的所有数据库表：**show tables**;

MySQL 层次：不同项目对应不同的数据库组成 - 每个数据库中有很多表  - 每个表中有很多数据

 select * from user;查看有多少数据

### 【3】退出数据库

退出数据库可以使用quit或者exit命令完成，也可以用\q;  完成退出操作

## 数据库的卸载

### 【1】卸载数据库

1. 停止MySQL服务：在命令行模式下执行net stop mysql或者在Windows服务窗口下停止服务

![image-20220422234449477](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183113076-1651975659.png)

2. 在控制面板中删除MySQL软件

![image-20220422234559482](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183113625-1654131162.png)

3. 删除软件文件夹：直接删除安装文件夹C:\Program Files\MySQL，其实此时该文件夹已经被删除或者剩下一个空文件夹。
   删除数据文件夹：直接删除文件夹C:\ProgramData\MySQL。此步不要忘记，否则会影响MySQL的再次安装。
   （ProgramData文件夹可能是隐藏的，显示出来即可）
   （MySQL文件下的内容才是真正的MySQL中数据）

4. 删除path环境变量中关于MySQL安装路径的配置 

## 使用图形客户端navicat12连接MySQL

![image-20220422234725594](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183114028-1959551315.png)

### 【1】认识Navicat

Navicat是一套快速、可靠并价格相当便宜的数据库管理工具，专为简化数据库的管理及降低系统管理成本而设。它的设计符合数据库管理员、开发人员及中小企业的需要。Navicat 是以直觉化的图形用户界面而建的，让你可以以安全并且简单的方式创建、组织、访问并共用信息。

![image-20220422234742077](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183114634-963286248.png)

Navicat Premium 是一套数据库开发工具，让你从单一应用程序中同时连接 MySQL、MariaDB、MongoDB、SQL Server、Oracle、PostgreSQL 和 SQLite 数据库。它与 Amazon RDS、Amazon Aurora、Amazon Redshift、Microsoft Azure、Oracle Cloud、MongoDB Atlas、阿里云、腾讯云和华为云等云数据库兼容。你可以快速轻松地创建、管理和维护数据库。

![image-20220422234752693](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183114996-954130523.png)

### 【2】安装navicat

直接解压安装包，拷贝到你定义的目录下，双击其中的navicat.exe，即可开始运行。打开后选择 连接工具按钮----连接，输入四个连接连接参数，并进行测试，结果提示连接失败，报2059异常。

 ![image-20220422234802506](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183115428-1779652613.png)

该错误的原因是在MySQL8之前版本中加密规则mysql_native_password，而在MySQL8以后的加密规则为caching_sha2_password。解决此问题有两种方法，一种是更新navicat驱动来解决此问题，一种是将mysql用户登录的加密规则修改为mysql_native_password。此处采用第二种方式。

![image-20220422234812624](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183115937-739043902.png)

设置密码永不过期
alter user 'root'@'localhost' identified by 'root' password expire never;
设置加密规则为mysql_native_password 
alter user 'root'@'localhost' identified with mysql_native_password by 'root';
重新访问navicat，提示连接成功。



可以看到，和在cmd下执行show databases，use mysql，show tables做的任务其实是一样的，但是提供了图形化的更方便的操作页面。

![image-20220422234824604](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183116366-145712849.png)

![image-20220422234835565](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183116782-1244630688.png)

## SQL语言入门

### 【1】SQL语言入门

​      我们都知道，数据库管理人员（DBA）通过数据库管理系统（DBMS）可以对数据库（DB）中的数据进行操作，但具体是如何操作的呢？这就涉及到我们本节要讲的SQL语言。
SQL（Structured Query Language）是结构化查询语言的简称，它是一种数据库查询和程序设计语言，同时也是目前使用最广泛的关系型数据库操作语言。在数据库管理系统中，使用SQL语言来实现数据的存取、查询、更新等功能。SQL是一种非过程化语言，只需提出“做什么”，而不需要指明“怎么做”。
​      SQL是由IBM公司在1974~1979年之间根据E.J.Codd发表的关系数据库理论为基础开发的，其前身是“SEQUEL”，后更名为SQL。由于SQL语言具有集数据查询、数据操纵、数据定义和数据控制功能于一体，类似自然语言、简单易用以及非过程化等特点，得到了快速的发展，并于1986年10月，被美国国家标准协会（American National Standards Institute，ANSI）采用为关系数据库管理系统的标准语言，后为国际标准化组织（International Organization for Standardization，ISO）采纳为国际标准。

![image-20220422234935561](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183117179-1078541426.png)

### 【2】SQL语言分为五个部分：

- 数据查询语言（Data Query Language，DQL）：**DQL主要用于数据的查询**，其基本结构是使用SELECT子句，FROM子句和WHERE子句的组合来查询一条或多条数据。

- 数据操作语言（Data Manipulation Language，DML）：DML主要用于对数据库中的**数据进行增加、修改和删除**的操作，其主要包括：

  1、INSERT：增加数据
  2、UPDATE：修改数据
  3、DELETE：删除数据

- 数据定义语言（Data Definition Language，DDL）：DDL主要用针对是**数据库对象**（数据库、表、索引、视图、触发器、存储过程、函数）进行创建、修改和删除操作。其主要包括：

  1、CREATE：创建数据库对象
  2、ALTER：修改数据库对象
  3、DROP：删除数据库对象

- 数据控制语言（Data Control Language，DCL）：DCL用来授予或回收**访问 数据库的权限**，其主要包括：

  1、GRANT：授予用户某种权限
  2、REVOKE：回收授予的某种权限

- 事务控制语言（Transaction Control Language，TCL）：TCL用于数据库的**事务管理**。其主要包括：

  1、GRANT：授予用户某种权限
  2、REVOKE：回收授予的某种权限

## DDL_DML_创建数据库表

### 【1】认识数据库表

表（Table）是数据库中数据存储最常见和最简单的一种形式，数据库可以将复杂的数据结构用较为简单的二维表来表示。二维表是由行和列组成的，分别都包含着数据，如表所示。

每个表都是由若干行和列组成的，在数据库中表中的行被称为记录，表中的列被称为是这些记录的字段。

记录也被称为一行数据，是表里的一行。在关系型数据库的表里，一行数据是指一条完整的记录。

字段是表里的一列，用于保存每条记录的特定信息。如上表所示的学生信息表中的字段包括“学号”、“姓名”、“性别”和“年龄”。数据表的一列包含了某个特定字段的全部信息。 

### 【2】创建数据库表 t_student

建立一张用来存储学生信息的表
字段包含学号、姓名、性别，年龄、入学日期、班级，email等信息
学号是主键 = 不能为空 +  唯一
姓名不能为空
性别默认值是男
Email唯一
（1）创建数据库：

（2）新建查询：


（3）创建数据库表：

```mysql
##这是一个单行注释
/*
多行注释
多行注释
多行注释
*/
/*
建立一张用来存储学生信息的表
字段包含学号、姓名、性别，年龄、入学日期、班级，email等信息
*/
-- 创建数据库表：
create table t_student(
        sno int(6), -- 6显示长度 
        sname varchar(5), -- 5个字符
        sex char(1),
        age int(3),
        enterdate date,
        classname varchar(10),
        email varchar(15)
);
-- 查看表的结构：展示表的字段详细信息
desc t_student;
-- 查看表中数据：
select * from t_student;
-- 查看建表语句：
show create table t_student;
/*
CREATE TABLE `t_student` (
  `sno` int DEFAULT NULL,
  `sname` varchar(5) DEFAULT NULL,
  `sex` char(1) DEFAULT NULL,
  `age` int DEFAULT NULL,
  `enterdate` date DEFAULT NULL,
  `classname` varchar(10) DEFAULT NULL,
  `email` varchar(15) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
*/

```

## 数据库表列类型

### 1.整数类型

![image-20220423164510467](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183117519-875021487.png)

MySQL支持选择在该类型关键字后面的括号内指定整数值的显示宽度(例如，INT(4))。显示宽度并不限制可以在列内保存的值的范围，也不限制超过列的指定宽度的值的显示
主键自增：不使用序列，通过auto_increment，要求是整数类型

### 2.浮点数类型

![image-20220423164521406](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183117819-2147142420.png)

需要注意的是与整数类型不一样的是，浮点数类型的宽度不会自动扩充。 score double(4,1)
 score double(4,1)--小数部分为1位，总宽度4位，并且不会自动扩充。
3.字符串类型

![image-20220423164530128](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183118178-827951850.png)

CHAR和VARCHAR类型相似，均用于存于较短的字符串，主要的不同之处在于存储方式。**CHAR类型长度固定，VARCHAR类型的长度可变**。
因为VARCHAR类型能够根据字符串的实际长度来动态改变所占字节的大小，所以在不能明确该字段具体需要多少字符时推荐使用VARCHAR类型，这样可以大大地节约磁盘空间、提高存储效率。
CHAR和VARCHAR表示的是字符的个数，而不是字节的个数

### 4.日期和时间类型

![image-20220423164606321](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183118542-1528582640.png)

TIMESTEMP类型的数据指定方式与DATETIME基本相同，两者的不同之处在于以下几点：
(1) 数据的取值范围不同，TIMESTEMP类型的取值范围更小。
(2) 如果我们对TIMESTAMP类型的字段没有明确赋值，或是被赋与了NULL值，MySQL会自动将该字段赋值为系统当前的日期与时间。
(3) TIMESTEMP类型还可以使用CURRENT_TIMESTAMP来获取系统当前时间。
(4) TIMESTEMP类型有一个很大的特点，那就是时间是根据时区来显示的。例如，在东八区插入的TIMESTEMP数据为2017-07-11 16:43:25，在东七区显示时，时间部分就变成了15:43:25，在东九区显示时，时间部分就变成了17:43:25。  

## DML_添加数据

注意事项
int  宽度是显示宽度，如果超过，可以自动增大宽度 int底层都是4个字节
时间的方式多样  '1256-12-23'  "1256/12/23"  "1256.12.23"
字符串不区分单引号和双引号
如何写入当前的时间  now() , sysdate() , CURRENT_DATE()
char varchar 是字符的个数，不是字节的个数，可以使用binary，varbinary表示定长和不定长的字节个数。
如果不是全字段插入数据的话，需要加入字段的名字

```mysql
-- 查看表记录：
select * from t_student;
-- 在t_student数据库表中插入数据：
insert into t_student values (1,'张三','男',18,'2022-5-8','软件1班','123@126.com');
insert into t_student values (10010010,'张三','男',18,'2022-5-8','软件1班','123@126.com');
insert into t_student values (2,'张三','男',18,'2022.5.8','软件1班','123@126.com');
insert into t_student values (2,"张三",'男',18,'2022.5.8','软件1班','123@126.com');
insert into t_student values (7,"张三",'男',18,now(),'软件1班','123@126.com');
insert into t_student values (9,"易烊千玺",'男',18,now(),'软件1班','123@126.com');
insert into t_student (sno,sname,enterdate) values (10,'李四','2023-7-5');
```

## DML_修改，删除数据

注意事项
1.关键字，表名，字段名不区分大小写
2.默认情况下，内容不区分大小写
3.删除操作from关键字不可缺少
4.修改，删除数据别忘记加限制条件

```mysql
-- 修改表中数据
update t_student set sex = '女' ;
update t_student set sex = '男' where sno = 10 ;
UPDATE T_STUDENT SET AGE = 21 WHERE SNO = 10;
update t_student set CLASSNAME = 'java01' where sno = 10 ;
update t_student set CLASSNAME = 'JAVA01' where sno = 9 ;
update t_student set age = 29 where classname = 'java01';
-- 删除操作：
delete from t_student where sno = 2;


```

## DML_修改，删除数据库表

```mysql
-- 查看数据：
select * from t_student;
-- 修改表的结构：
-- 增加一列：
alter table t_student add score double(5,2) ; -- 5:总位数  2：小数位数 
update t_student set score = 123.5678 where sno = 1 ;
-- 增加一列（放在最前面）
alter table t_student add score double(5,2) first;
-- 增加一列（放在sex列的后面）
alter table t_student add score double(5,2) after sex;
-- 删除一列：
alter table t_student drop score;
-- 修改一列：
alter table t_student modify score float(4,1); -- modify修改是列的类型的定义，但是不会改变列的名字
alter table t_student change score score1 double(5,1); -- change修改列名和列的类型的定义
-- 删除表：
drop table t_student;
```

## 表的完整性约束

### 非外键约束

#### 【1】代码演示非外键约束：

```mysql
/*
建立一张用来存储学生信息的表
字段包含学号、姓名、性别，年龄、入学日期、班级，email等信息
约束：
建立一张用来存储学生信息的表
字段包含学号、姓名、性别，年龄、入学日期、班级，email等信息
【1】学号是主键 = 不能为空 +  唯一 ，主键的作用：可以通过主键查到唯一的一条记录【2】如果主键是整数类型，那么需要自增
【3】姓名不能为空
【4】Email唯一
【5】性别默认值是男
【6】性别只能是男女
【7】年龄只能在18-50之间
*/
-- 创建数据库表：
create table t_student(
        sno int(6) primary key auto_increment, 
        sname varchar(5) not null, 
        sex char(1) default '男' check(sex='男' || sex='女'),
        age int(3) check(age>=18 and age<=50),
        enterdate date,
        classname varchar(10),
        email varchar(15) unique
);
-- 添加数据：
--  1048 - Column 'sname' cannot be null 不能为null
-- 3819 - Check constraint 't_student_chk_1' is violated. 违反检查约束
insert into t_student values (1,'张三','男',21,'2023-9-1','java01班','zs@126.com');
-- 1062 - Duplicate entry '1' for key 't_student.PRIMARY' 主键重复
-- > 1062 - Duplicate entry 'ls@126.com' for key 't_student.email' 违反唯一约束
insert into t_student values (2,'李四','男',21,'2023-9-1','java01班','ls@126.com');
insert into t_student values (3,'露露','男',21,'2023-9-1','java01班','ls@126.com');
-- 如果主键没有设定值，或者用null.default都可以完成主键自增的效果
insert into t_student (sname,enterdate) values ('菲菲','2029-4-5');
insert into t_student values (null,'小明','男',21,'2023-9-1','java01班','xm@126.com');
insert into t_student values (default,'小刚','男',21,'2023-9-1','java01班','xg@126.com');
-- 如果sql报错，可能主键就浪费了，后续插入的主键是不连号的，我们主键也不要求连号的
insert into t_student values (null,'小明','男',21,'2023-9-1','java01班','oo@126.com');
-- 查看数据：
select * from t_student;
```

#### 【2】约束从作用上可以分为两类：

(1)   表级约束：可以约束表中任意一个或多个字段。与列定义相互独立，不包含在列定义中；与定义用‘，’分隔；必须指出要约束的列的名称；

(2)   列级约束：包含在列定义中，直接跟在该列的其它定义之后 ，用空格分隔；不必指定列名；

```mysql
-- 删除表：
drop table t_student;
-- 创建数据库表：
create table t_student(
        sno int(6) auto_increment, 
        sname varchar(5) not null, 
        sex char(1) default '男',
        age int(3),
        enterdate date,
        classname varchar(10),
        email varchar(15),
        constraint pk_stu primary key (sno),  -- pk_stu 主键约束的名字
        constraint ck_stu_sex check (sex = '男' || sex = '女'),
        constraint ck_stu_age check (age >= 18 and age <= 50),
        constraint uq_stu_email unique (email)
);
-- 添加数据：
insert into t_student values (1,'张三','男',21,'2023-9-1','java01班','zs@126.com');
-- > 3819 - Check constraint 'ck_stu_sex' is violated.
-- > 3819 - Check constraint 'ck_stu_age' is violated.
-- > 1062 - Duplicate entry 'zs@126.com' for key 't_student.uq_stu_email'
insert into t_student values (3,'李四','男',21,'2023-9-1','java01班','zs@126.com');
-- 查看数据：
select * from t_student;
```

#### 【3】在创建表以后添加约束：

```mysql
-- 删除表：
drop table t_student;
-- 创建数据库表：
create table t_student(
        sno int(6), 
        sname varchar(5) not null, 
        sex char(1) default '男',
        age int(3),
        enterdate date,
        classname varchar(10),
        email varchar(15)
);
-- > 1075 - Incorrect table definition; there can be only one auto column and it must be defined as a key
-- 错误的解决办法：就是auto_increment去掉
-- 在创建表以后添加约束：
alter table t_student add constraint pk_stu primary key (sno) ; -- 主键约束
alter table t_student modify sno int(6) auto_increment; -- 修改自增条件
alter table t_student add constraint ck_stu_sex check (sex = '男' || sex = '女');
alter table t_student add constraint ck_stu_age check (age >= 18 and age <= 50);
alter table t_student add constraint uq_stu_email unique (email);
-- 查看表结构：
desc t_student;
```

验证约束添加成功：查看表结构：



![image-20220423222622959](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183118863-1139172289.png)



#### 【4】总结：

##### 1.主键约束 

主键约束（PRIMARY KEY，缩写PK），是数据库中最重要的一种约束，其作用是约束表中的某个字段可以唯一标识一条记录。因此，使用主键约束可以快速查找表中的记录。就像人的身份证、学生的学号等等，设置为主键的字段取值不能重复（唯一），也不能为空（非空），否则无法唯一标识一条记录。

主键可以是单个字段，也可以是多个字段组合。对于单字段主键的添加可使用表级约束，也可以使用列级约束；而对于多字段主键的添加只能使用表级约束。

##### 2.非空约束 

非空约束（NOT NULL，缩写NK）规定了一张表中指定的某个字段的值不能为空（NULL）。设置了非空约束的字段，在插入的数据为NULL时，数据库会提示错误，导致数据无法插入。

无论是单个字段还是多个字段非空约束的添加只能使用列级约束（非空约束无表级约束）


为已存在表中的字段添加非空约束 

```mysql
 alter   table student8 modify stu_sex varchar(1) not null;
```

 

使用ALTER TABLE语句删除非空约束 

 

```mysql
alter  table student8 modify stu_sex varchar(1) null;
```



3. ##### 唯一约束


唯一约束（UNIQUE，缩写UK）比较简单，它规定了一张表中指定的某个字段的值不能重复，即这一字段的每个值都是唯一的。如果想要某个字段的值不重复，那么就可以为该字段添加为唯一约束。

无论单个字段还是多个字段唯一约束的添加均可使用列级约束和表级约束

4. ##### 检查约束

检查约束（CHECK）用来限制某个字段的取值范围，可以定义为列级约束，也可以定义为表级约束。MySQL8开始支持检查约束。 

5. ##### 默认值约束 

默认值约束（DEFAULT）用来规定字段的默认值。如果某个被设置为DEFAULT约束的字段没插入具体值，那么该字段的值将会被默认值填充。

 默认值约束的设置与非空约束一样，也只能使用列级约束。

6. ##### 字段值自动增加约束

自增约束（AUTO_INCREMENT）可以使表中某个字段的值自动增加。一张表中只能有一个自增长字段，并且该字段必须定义了约束（该约束可以是主键约束、唯一约束以及外键约束），如果自增字段没有定义约束，数据库则会提示“Incorrect table definition; there can be only one auto column and it must be defined as a key”错误。

由于自增约束会自动生成唯一的ID，所以自增约束通常会配合主键使用，并且只适用于整数类型。一般情况下，设置为自增约束字段的值会从1开始，每增加一条记录，该字段的值加1。

为已存在表中的字段添加自增约束 

```mysql
/*创建表student11*/

 create   table student11 (

       stu_id int(10) primary key,
    
       stu_name varchar(3),
    
       stu_sex varchar (1)

);

/*为student11表中的主键字段添加自增约束*/

alter   table student11 modify stu_id int(10) auto_increment;
```

 

使用ALTER TABLE语句删除自增约束 

```mysql
alter   table studen11 modify stu_id int(10);
```

 























