# Linux

## 操作系统

Centos和RedHat具有相同的内核

Linux是开放源码的自由软件

Centos使用非常多，尤其在中国

![image-20220321102524982.png](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183702741-1373869202.gif)

## Linux的文件结构

![image-20220322191943189.png](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183703281-478936387.gif)

![image-20220322191725629.png](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183703625-477990064.gif)

蓝色代表普通目录

青色代表系统目录或者可执行目录

绿色代表临时文件

输入ls

| 目录      | 说明                           | 备注                                                         |
| --------- | ------------------------------ | ------------------------------------------------------------ |
| bin       | 存放普通用户可执行的指令       | 即使在单用户模式下也能够执行处理                             |
| boot      | 开机引导目录                   | 包括Linux内核文件与开机所需要的文件                          |
| dev       | 设备目录                       | 所有的硬件设备及周边均放置在这个设备目录中                   |
| etc       | 各种配置文件目录               | 大部分配置属性均存放在这里                                   |
| lib/lib64 | 开机时常用的动态链接库         | bin及sbin指令也会调用对应的lib库                             |
| media     | 可移除设备挂载目录             | 类似软盘 U盘  光盘等临时挂放目录                             |
| mnt       | 用户临时挂载其他的文件系统     | 额外的设备可挂载在这里,相对临时而言                          |
| opt       | 第三方软件安装目录             | 现在习惯性的放置在/usr/local中                               |
| proc      | 虚拟文件系统                   | 通常是内存中的映射,特别注意在误删除数据文件后，比如DB，只要系统不重启,还是有很大几率能将数据找回来 |
| root      | 系统管理员主目录               | 除root之外,其他用户均放置在/home目录下                       |
| run       | 系统运行是所需文件             | 以前防止在/var/run中,后来拆分成独立的/run目录。重启后重新生成对应的目录数据 |
| sbin      | 只有root才能运行的管理指令     | 跟bin类似,但只属于root管理员                                 |
| srv       | 服务启动后需要访问的数据目录   |                                                              |
| sys       | 跟proc一样虚拟文件系统         | 记录核心系统硬件信息                                         |
| tmp       | 存放临时文件目录               | 所有用户对该目录均可读写                                     |
| usr       | 应用程序放置目录               |                                                              |
| var       | 存放系统执行过程经常改变的文件 |                                                              |

**/etc**： 这个是系统中的配置文件

 **/bin, /sbin, /usr/bin, /usr/sbin**: 这是系统预设的执行文件的放置目录，比如 ls 就是在/bin/ls 目录下的。

 值得提出的是，/bin, /usr/bin 是给系统用户使用的指令（除root外的通用户），而/sbin, /usr/sbin 则是给root使用的指令。

 **/var**： 这是一个非常重要的目录，系统上跑了很多程序，那么每个程序都会有相应的日志产生，而这些日志就被记录到这个目录 下，具体在/var/log 目录下，另外mail的预设放置也是在这里。

#### 基本概念

**用户目录：位于/home/user,称之为用户工作目录或家目录**

\# 在home有一个user 这里就是之前创建的msb123用户[root@localhost ~]# cd /home[root@localhost home]# lsmsb123# 使用~回到root目录，使用/是回到根目录下[root@localhost msb123]# cd ~[root@localhost ~]# 

 **登录信息**

[root@localhost /]#Linux的bash解析器终端用来显示主机名和当前用户的标识；# root表示bai当前用户叫root（系统管理员账户）# localhost表示当前使用的主机名叫localhost（没有设置系bai统名字的时候默认名称是localhost）# / 表示你当前所处的目录位置 (这里的'/'表示你当前在根目录下)

 **相对路径和绝对路径**

 **绝对路径**

 从/目录开始描述的路径为绝对路径，如：

[root@localhost /]# cd /home/msb123[root@localhost /]# ls /usr

 **相对路径**

 从当前位置开始描述的路径为相对路径，如：

[root@localhost /]# cd ../../[root@localhost /]# ls abc/def

 **.** **和 ..**

 每个目录下都有**.**和**..**

. 表示当前目录.. 表示上一级目录，即父目录

root有最高权限（管理员权限）

3个为一组

第一组为（所有者） 所有者就是创建文件的用户

第二组为 用户组 由用户组合成的用户组（由所有者所规定和允许）

第三组为 其他组 系统内的其他所有用户就是other用户类

l rwx rwx rwx [ l ] 表示为链接文档(link file)

d r-x r-x r-x [ d ] 表示目录

首先第一个字母 在Linux中第一个字符代表这个文件是目录、文件或链接文件等等。

**r 代表浏览权限** 代表可读(read)

**w 代表新建，修改，删除文件的权限** 代表可写(write)

**x 代表执行文件的权限；对目录来说用户具有进入目录的权限。 ** 代表可执行(execute)

 

![image-20220322212400801.png](photo/clip_image008-16493199746403.gif)

uid用户的id

Gid用户组的id

#### 3、基本命令信息

 熟悉一些入门的命令

### 0.stat

查看目标文件详情

  \### 1、ls

ls 命令作用：Linux ls命令用于显示指定工作目录下之内容（列出目前工作目录所含之文件及子目录)。语法： ls  [-alrtAFR](选项)  [name...](参数)参数：-a 显示所有文件及目录 (ls内定将文件名或目录名称开头为"."的视为隐藏档，不会列出) 示例如下：[root@localhost ~]# ls -a. .. anaconda-ks.cfg .bash_history .bash_logout .bash_profile .bashrc .cshrc .tcshrc-l 除文件名称外，亦将文件型态、权限、拥有者、文件大小等资讯详细列出 示例如下：[root@localhost ~]# ls -l总用量 4-rw-------. 1 root root 1437 8月 31 15:54 anaconda-ks.cfg-r 将文件以相反次序显示(原定依英文字母次序) 示例如下：[root@localhost ~]# ls -ra.tcshrc .cshrc .bashrc .bash_profile .bash_logout .bash_history anaconda-ks.cfg .. .-t 将文件依建立时间之先后次序列出 示例如下：[root@localhost ~]# ls -lt总用量 4-rw-------. 1 root root 1437 8月 31 15:54 anaconda-ks.cfg-A 同 -a ，但不列出 "." (目前目录) 及 ".." (父目录)  示例如下：[root@localhost ~]# ls -Aanaconda-ks.cfg .bash_history .bash_logout .bash_profile .bashrc .cshrc .tcshrc-F 在列出的文件名称后加一符号；例如可执行档则加 "*", 目录则加 "/" 示例如下：[root@localhost ~]# ls -F /homemsb123/-R 若目录下有文件，则以下之文件亦皆依序列出 示例如下：[root@localhost ~]# ls -R /home/home:msb123/home/msb123:

常用组合[1]查看文件详情：ls -l 或 ll[2]增强对文件大小易读性，以人类可读的形式显示文件大小： ls -lh[3]对文件或者目录进行从大到小的排序： ls -lhs[4]查看当前目录下的所有文件或者目录，包括隐藏文件： ls -la[5]只查看当前目录下的目录文件： ls -d .[6]按照时间顺序查看，从上到倒下时间越来越近： ls -ltr[7]查看文件在对应的inode信息：ls -li

### 2、cd

cd 命令作用：变换当前目录到dir。默认目录为home，可以使用绝对路径、或相对路径。语法：cd [dir](路径)

 跳到用户目录下[root@localhost ~]# cd /home/msb123[root@localhost msb123]#  # 回到home目录[root@localhost msb123]# cd ~[root@localhost ~]# # 跳到上次所在目录[root@localhost ~]# cd -/home/msb123[root@localhost msb123]## 跳到父目录(也可以直接使用 cd ..)[root@localhost msb123]# cd ./..[root@localhost home]# # 再次跳到上次所在目录[root@localhost home]# cd -/home/msb123[root@localhost msb123]# # 跳到当前目录的上两层[root@localhost msb123]# cd ../..[root@localhost /]## 把上个命令的最后参数作为dir这里我们先将文件夹cd 到python2.7路径[root@localhost /]# cd /usr/include/python2.7/[root@localhost python2.7]## 这里使用cd ./..参数作为引子[root@localhost python2.7]# cd ./..# 这里我们使用命令，重复最后一个命令参数，直到回到了根目录[root@localhost include]# cd !$cd ./..[root@localhost usr]# cd ./..[root@localhost /]#

### 3、pwd

pwd 命令作用：可立刻得知目前所在的工作目录的绝对路径名称语法：pwd [--help][--version]参数说明:--help 在线帮助。--version 显示版本信息。查看当前所在目录：[root@localhost /]# cd /home[root@localhost home]# pwd/home[root@localhost home]# 

#### 4、mkdir 命令

**mkdir**

 **作用：**命令用来创建指定的名称的目录，要求创建目录的用户在当前目录中具有写权限，并且指定的目录名不能是当前目录中已有的目录

 **语法：**mkdir [选项] 目录

 **命令功能：**通过 mkdir 命令可以实现在指定位置创建以 DirName(指定的文件名)命名的文件夹或目录。要创建文件夹或目录的用户必须对所创建的文件夹的父文件夹具有写权限。并且，所创建的文件夹(目录)不能与其父目录(即父文件夹)中的文件名重名，即同一个目录下不能有同名的(区分大小写)

 **命令参数：**

| 选项参数 | 完整参数                           | 功能描述                                                     |
| -------- | ---------------------------------- | ------------------------------------------------------------ |
| -m       | --mode=模式                        | 设定权限<模式> (类似 chmod)，而不是 rwxrwxrwx 减 umask       |
| -p       | --parents                          | 可以是一个路径名称。     此时若路径中的某些目录尚不存在,加上此选项后,     系统将自动建立好那些尚不存在的目录,即一次可以建立多个目录; |
| -v       | --verbose     --help     --version | --verbose 每次创建新目录都显示信息     --help显示此帮助信息并退出     --version输出版本信息并退出 |

#### 5、touch 命令

**touch**

 **作用：**用于修改文件或者目录的时间属性，包括存取时间和更改时间。若文件不存在，系统会建立一个新的文件。

 ls -l 可以显示档案的时间记录。

 **语法：**touch [-acfm] [-d<日期时间>] [-r<参考文件或目录>] [-t<日期时间>] [--help] [--version] [文件或目录…]

 **命令参数：**

| 参数        | 参数描述                                                  |
| ----------- | --------------------------------------------------------- |
| -a          | 只更新访问时间，不改变修改时间                            |
| -m          | 改变修改时间记录                                          |
| -c          | 不创建不存在的文件                                        |
| -f          | 不使用，是为了与其他  unix 系统的相容性而保留。           |
| -m          | 只更新修改时间，不改变访问时间                            |
| -r file     | 使用文件file的时间更新文件的时间                          |
| -t          | 将时间修改为参数指定的日期,如：07081556代表7月8号15点56分 |
| --no-create | 不会建立文件                                              |
| --help      | 列出指令格式                                              |
| --version   | 列出版本讯息                                              |

### 6.mv: 改名字

rm 和mv之类的命令应在后面加上 -i

### 7.cp: 用于复制文件或目录

常用的一些有以下参数

-i 提示

-r 复制目录及目录内所有项目

-a 复制的文件与原文件时间一样

（4）移动当前文件夹下的所有文件到上一级目录

[root@localhost test2]# mv * ../[root@localhost test2]# ls ../test1 test2 text1.txt text2.log

####  

### Tab键自动补充与提示

#### 8.cat

 **作用：**用于连接文件并打印到标准输出设备上

 **语法：**cat [-AbeEnstTuv] [--help] [--version] fileName

 **命令参数：**

|        |                        |                                                    |
| ------ | ---------------------- | -------------------------------------------------- |
| **-n** | **--number**           | 由 1 开始对所有输出的行数编号                      |
| **-b** | **--number-nonblank**  | 和 -n 相似，只不过对于空白行不编号                 |
| **-s** | **--squeeze-blank**    | 当遇到有连续两行以上的空白行，就代换为一行的空白行 |
| **-v** | **--show-nonprinting** | 使用 ^ 和 M- 符号，除了 LFD 和  TAB 之外           |
| **-E** | **--show-ends**        | 在每行结束处显示 $                                 |
| **-T** | **--show-tabs**        | 将 TAB 字符显示为 ^I                               |
| **-A** | **--show-all**         | 等价于 -vET                                        |
| **-e** |                        | 等价于"-vE"选项                                    |
| **-t** |                        | 等价于"-vT"选项                                    |

#### 9、more 命令

more

 **作用：**类似 cat ，不过会以一页一页的形式显示，更方便使用者逐页阅读，而最基本的指令就是按空白键（space）就往下一页显示，按 b 键就会往回（back）一页显示，而且还有搜寻字串的功能（与 vi 相似），使用中的说明文件，请按 h

 **语法：**more [-dlfpcsu] [-num] [+/pattern] [+linenum] [fileNames..]

 **命令参数：**

| 参数      | 参数功能描述                                                 |
| --------- | ------------------------------------------------------------ |
| -num      | 一次显示的行数                                               |
| -d        | 提示使用者，在画面下方显示 [Press space to continue, 'q' to quit.] ，     如果使用者按错键，则会显示 [Press 'h' for instructions.] 而不是 '哔' 声 |
| -l        | 取消遇见特殊字元  ^L（送纸字元）时会暂停的功能               |
| -f        | 计算行数时，以实际上的行数，而非自动换行过后的行数（有些单行字数太长的会被扩展为两行或两行以上） |
| -p        | 不以卷动的方式显示每一页，而是先清除萤幕后再显示内容         |
| -c        | 跟 -p 相似，不同的是先显示内容再清除其他旧资料               |
| -s        | 当遇到有连续两行以上的空白行，就代换为一行的空白行           |
| -u        | 不显示下引号 （根据环境变数 TERM 指定的 terminal 而有所不同） |
| +/pattern | 在每个文档显示前搜寻该字串（pattern），然后从该字串之后开始显示 |
| +num      | 从第 num 行开始显示                                          |
| fileNames | 需要显示内容的文档，可为复数个数                             |

 常用的操作命令

| 按键   | 按键功能描述                     |
| ------ | -------------------------------- |
| Enter  | 向下 n 行，需要定义。默认为 1 行 |
| Ctrl+F | 向下滚动一屏                     |
| 空格键 | 向下滚动一屏                     |
| Ctrl+B | 返回上一屏                       |
| =      | 输出当前行的行号                 |
| :f     | 输出文件名和当前行的行号         |
| V      | 调用vi编辑器                     |
| !命令  | 调用Shell，并执行命令            |
| q      | 退出more                         |

#### 9、less 命令

less

 **作用：**less 与 more 类似，但使用 less 可以随意浏览文件，而 more 仅能向前移动，却不能向后移动，而且 less 在查看之前不会加载整个文件。

 **语法：**less [参数] 文件

 **命令参数：**

| 参数       | 参数功能描述                                   |
| ---------- | ---------------------------------------------- |
| -i         | 忽略搜索时的大小写                             |
| -N         | 显示每行的行号                                 |
| -o         | <文件名> 将less 输出的内容在指定文件中保存起来 |
| -s         | 显示连续空行为一行                             |
| /字符串：  | 向下搜索“字符串”的功能                         |
| ?字符串：  | 向上搜索“字符串”的功能                         |
| n          | 重复前一个搜索（与  / 或 ? 有关）              |
| N          | 反向重复前一个搜索（与 / 或 ? 有关）           |
| -x <数字>  | 将“tab”键显示为规定的数字空格                  |
| b          | 向后翻一页                                     |
| d          | 向后翻半页                                     |
| h          | 显示帮助界面                                   |
| Q          | 退出less 命令                                  |
| u          | 向前滚动半页                                   |
| y          | 向前滚动一行                                   |
| 空格键     | 滚动一行                                       |
| 回车键     | 滚动一页                                       |
| [pagedown] | 向下翻动一页                                   |
| [pageup]   | 向上翻动一页                                   |

## 第五章 Linux基本命令二

#### 1、head 命令

head

 **作用：**用于查看文件的开头部分的内容，有一个常用的参数 **-n** 用于显示行数，默认为 10，即显示 10 行的内容

 **语法：**head [参数] [文件]

 **命令参数：**

| 参数     | 参数描述     |
| -------- | ------------ |
| -q       | 隐藏文件名   |
| -v       | 显示文件名   |
| -c<数目> | 显示的字节数 |
| -n<行数> | 显示的行数   |

 （1）显示 1.log 文件中前 20 行

[root@localhost ~]# head 1.log -n 20

 （2）显示 1.log 文件前 20 字节

[root@localhost ~]# head -c 20 log2014.log

 （3）显示 t.log最后 10 行

[root@localhost ~]# head -n -10 t.log

扩展：tail 命令，查看文件的末尾

**tail -f xxx：实时观看后面内容的输入**

#### 2、which 命令

which

 在 linux 要查找某个命令或者文件，但不知道放在哪里了，可以使用下面的一些命令来搜索

which   查看可执行文件的位置。whereis  查看文件的位置。locate  配合数据库查看文件位置。find   实际搜寻硬盘查询文件名称。

 **作用：**用于查找文件（which指令会在环境变量$PATH设置的目录里查找符合条件的文件。）

 **语法：**which [文件...]

 **命令参数：**

| 参数           | 参数描述                                                     |
| -------------- | ------------------------------------------------------------ |
| -n<文件名长度> | 指定文件名长度，指定的长度必须大于或等于所有文件中最长的文件名 |
| -p<文件名长度> | 与-n参数相同，但此处的<文件名长度>包括了文件的路径           |
| -w             | 指定输出时栏位的宽度                                         |
| -V             | 显示版本信息                                                 |

### su user :改变用户

#### 3、whereis命令

whereis

whereis 命令只能用于程序名的搜索，而且只搜索二进制文件（参数-b）、man说明文件（参数-m）和源代码文件（参数-s）。如果省略参数，则返回所有信息。whereis 及 locate 都是基于系统内建的数据库进行搜索，因此效率很高，而find则是遍历硬盘查找文件

 **作用：**用于查找文件

 **语法：**whereis [-bfmsu][-B <目录>...]-M <目录>...][-S <目录>...][文件...]

 **命令参数：**

| 参数     | 参数描述                                                     |
| -------- | ------------------------------------------------------------ |
| -b       | 定位可执行文件                                               |
| -B<目录> | 只在设置的目录下查找可执行文件                               |
| -f       | 不显示文件名前的路径名称                                     |
| -m       | 定位帮助文件                                                 |
| -M<目录> | 只在设置的目录下查找说帮助文件                               |
| -s       | 定位源代码文件                                               |
| -S<目录> | 只在设置的目录下查找源代码文件                               |
| -u       | 搜索默认路径下除可执行文件、源代码文件、帮助文件以外的其它文件 |

 \### 4、locate命令

从数据库中索引

[root@localhost ~]# yum install mlocate...省略...[root@localhost ~]# updatedb

 **作用：**用于查找符合条件的文档，他会去保存文档和目录名称的数据库内，查找合乎范本样式条件的文档或目录

 **语法：**locate [-d ][--help][--version][范本样式...]

 **命令参数：**

| 参数    | 参数描述                                                     |
| ------- | ------------------------------------------------------------ |
| -b      | 仅匹配路径名的基本名称                                       |
| -c      | 只输出找到的数量                                             |
| -d      | 使用 DBPATH 指定的数据库，而不是默认数据库  /var/lib/mlocate/mlocate.db |
| -e      | 仅打印当前现有文件的条目                                     |
| -1      | 如果 是 1．则启动安全模式。在安全模式下，使用者不会看到权限无法看到  的档案。     这会始速度减慢，因为  locate 必须至实际的档案系统中取得档案的 权限资料 |
| -0      | 在输出上带有NUL的单独条目                                    |
| -S      | 不搜索条目，打印有关每个数据库的统计信息                     |
| -q      | 安静模式，不会显示任何错误讯息                               |
| -P      | 检查文件存在时不要遵循尾随的符号链接                         |
| -l      | 将输出（或计数）限制为LIMIT个条目                            |
| -n      | 至多显示 n个输出                                             |
| -m      | 被忽略，为了向后兼容                                         |
| -r      | REGEXP -- 使用基本正则表达式                                 |
| --regex | 使用扩展正则表达式                                           |
| -o      | 指定资料库存的名称                                           |
| -h      | 显示帮助                                                     |
| -i      | 忽略大小写                                                   |
| -V      | 显示版本信息                                                 |

### 5、find命令

从磁盘中查找

 **作用：**用于在文件树中查找文件，并作出相应的处理

 **语法：**

find [-H] [-L] [-P] [-Olevel] [-D help|tree|search|stat|rates|opt|exec] [path...] [expression]

 **命令参数：**

| 参数     | 参数描述                                                     |
| -------- | ------------------------------------------------------------ |
| pathname | find命令所查找的目录路径。例如用.来表示当前目录，用/来表示系统根目录 |
| -print   | find命令将匹配的文件输出到标准输出                           |
| -exec    | find命令对匹配的文件执行该参数所给出的shell命令。相应命令的形式为'command' { } ;，注意{ }和\；之间的空格 |
| -ok      | 和-exec的作用相同，只不过以一种更为安全的模式来执行该参数所给出的shell命令，在执行每一个命令之前，都会给出提示，让用户来确定是否执行 |

 **命令选项：**

| 选项          | 选项描述                                                     |
| ------------- | ------------------------------------------------------------ |
| -name         | 按照文件名查找文件                                           |
| -perm         | 按文件权限查找文件                                           |
| -user         | 按文件属主查找文件                                           |
| -group        | 按照文件所属的组来查找文件                                   |
| -type         | 查找某一类型的文件，诸如：     b - 块设备文件     d - 目录     c - 字符设备文件     l - 符号链接文件     p - 管道文件     f - 普通文件 |
| -size n  :[c] | 查找文件长度为n块文件，带有c时表文件字节大小                 |
| -amin n       | 查找系统中最后N分钟访问的文件                                |
| -atime n      | 查找系统中最后n*24小时访问的文件                             |
| -cmin n       | 查找系统中最后N分钟被改变文件状态的文件                      |
| -ctime n      | 查找系统中最后n*24小时被改变文件状态的文件                   |
| -mmin n       | 查找系统中最后N分钟被改变文件数据的文件                      |
| -mtime n      | 查找系统中最后n*24小时被改变文件数据的文件                   |
| -maxdepth  n  | 最大查找目录深度                                             |
| -prune        | 选项来指出需要忽略的目录。在使用-prune选项时要当心，     因为如果你同时使用了-depth选项，那么-prune选项就会被find命令忽略 |
| -newer        | 如果希望查找更改时间比某个文件新但比另一个文件旧的所有文件，可以使用-newer选项 |

### 6、chmod命令

 Linux/Unix 的文件调用权限分为三级 : 文件拥有者、群组、其他。

用于改变 linux 系统文件或目录的访问权限。用它控制文件或目录的访问权限。该命令有两种用法。一种是包含字母和操作符表达式的文字设定法；另一种是包含数字的数字设定法。每一文件或目录的访问权限都有三组，每组用三位表示，分别为文件属主的读、写和执行权限；与属主同组的用户的读、写和执行权限；系统中其他用户的读、写和执行权限。可使用 ls -l test.txt 查找

[root@localhost ~]# ll总用量 20-rw-------. 1 root root 1437 8月 31 15:54 anaconda-ks.cfg-rw-r--r--. 1 root root  0 9月  8 18:29 file1lrwxrwxrwx. 1 root root  15 9月  7 16:31 link_text2 -> mydir/text2.logdrwxr-xr-x. 4 root root  92 9月  7 18:08 mydir-rw-r--r--. 1 root root  13 9月  8 16:35 myfile-rw-r--r--. 1 root root  36 9月  9 13:16 test3-rw-r--r--. 1 root root  36 9月  8 18:36 test.log-rwxr-xr-x. 1 root root  67 9月  8 18:36 test.sh

 这里使用test.log作为例子

-rw-r--r--. 1 root root  36 9月  8 18:36 test.log第一列共有 10 个位置，第一个字符指定了文件类型。在通常意义上，一个目录也是一个文件。如果第一个字符是横线，表示是一个非目录的文件。如果是 d，表示是一个目录。从第二个字符开始到第十个 9 个字符，3 个字符一组，分别表示了 3 组用户对文件或者目录的权限。权限字符用横线代表空许可，r 代表只读，w 代表写，x 代表可执行

 **语法：**

chmod [-cfvR] [--help] [--version] mode file...

 **常用参数：**

| 参数 | 参数描述                           |
| ---- | ---------------------------------- |
| -c   | 当发生改变时，报告处理信息         |
| -R   | 处理指定目录以及其子目录下所有文件 |

 **权限范围：**

u ：目录或者文件的当前的用户g ：目录或者文件的当前的群组o ：除了目录或者文件的当前用户或群组之外的用户或者群组a ：所有的用户及群组

 **权限代号：**

| 代号 | 代号权限              |
| ---- | --------------------- |
| r    | 读权限，用数字4表示   |
| w    | 写权限，用数字2表示   |
| x    | 执行权限，用数字1表示 |
| -    | 删除权限，用数字0表示 |
| s    | 特殊权限              |

环境：-rw-r--r--. 1 root root  36 9月  8 18:36 test.log

 （1）增加文件 t.log 所有用户可执行权限

[root@localhost ~]# ls -n test.log-rwxr-xr-x. 1 0 0 36 9月  8 18:36 test.log

 （2）撤销原来所有的权限，然后使拥有者具有可读权限,并输出处理信息

[root@localhost ~]# chmod u=r test.log -cmode of "test.log" changed from 0755 (rwxr-xr-x) to 0455 (r--r-xr-x)[root@localhost ~]# ls -n test.log-r--r-xr-x. 1 0 0 36 9月  8 18:36 test.log

 （3）给 file 的属主分配读、写、执行(7)的权限，给file的所在组分配读、执行(5)的权限，给其他用户分配执行(1)的权限

[root@localhost ~]# chmod 751 test.log -c或者[root@localhost ~]# chmod u=rwx,g=rx,o=x t.log -c

 （4）将mydir 目录及其子目录所有文件添加可读权限

[root@localhost ~]# chmod u+r,g+r,o+r -R text/ -c

### 7、chown命令

####  

chown 将指定文件的拥有者改为指定的用户或组，用户可以是用户名或者用户 ID；组可以是组名或者组 ID；文件是以空格分开的要改变权限的文件列表，支持通配符

 **常用参数：**

| 参数      | 参数描述                             |
| --------- | ------------------------------------ |
| user      | 新的文件拥有者的使用者 ID            |
| group     | 新的文件拥有者的使用者组(group)      |
| -c        | 显示更改的部分的信息                 |
| -f        | 忽略错误信息                         |
| -h        | 修复符号链接                         |
| -v        | 显示详细的处理信息                   |
| -R        | 处理指定目录以及其子目录下的所有文件 |
| --help    | 显示辅助说明                         |
| --version | 显示版本                             |

 （1）改变拥有者和群组 并显示改变信息

[root@localhost ~]# chown -c mail:mail test.logchanged ownership of "test.log" from root:root to mail:mail-r--r-xr-x. 1 mail mail    36 9月  8 18:36 test.log

 （2）改变文件群

[root@localhost ~]# chown -c :mail test.sh changed ownership of "test.sh" from root:root to :mail

 （3）改变文件夹及子文件目录属主及属组为 mail

[root@localhost ~]# chown -cR mail: mydirchanged ownership of "mydir/test1/text1.txt" from root:root to mail:mailchanged ownership of "mydir/test1" from root:root to mail:mail...省略...

#### 8、tar 命令

用来压缩和解压文件。tar 本身不具有压缩功能，只具有打包功能，有关压缩及解压是调用其它的功能来完成。弄清两个概念：打包和压缩。打包是指将一大堆文件或目录变成一个总的文件；压缩则是将一个大的文件通过一些压缩算法变成一个小文件

 **命令参数：**

| 参数 | 参数描述                       |
| ---- | ------------------------------ |
| -c   | 建立新的压缩文件               |
| -f   | 定压缩文件                     |
| -r   | 添加文件到已经压缩文件包中     |
| -u   | 添加改了和现有的文件到压缩包中 |
| -x   | 从压缩包中抽取文件             |
| -t   | 显示压缩文件中的内容           |
| -z   | 支持gzip压缩                   |
| -j   | 支持bzip2压缩                  |
| -Z   | 支持compress解压文件           |
| -v   | 显示操作过程                   |

 有关 gzip 及 bzip2 压缩:

gzip 实例：压缩 gzip fileName .tar.gz 和.tgz 解压：gunzip filename.gz 或 gzip -d filename.gz 对应：tar zcvf filename.tar.gz   tar zxvf filename.tar.gz

bz2实例：压缩 bzip2 -z filename .tar.bz2 解压：bunzip filename.bz2或bzip -d filename.bz2 对应：tar jcvf filename.tar.gz     解压：tar jxvf filename.tar.bz2

 

 

image-20220323192630755

 （1）将test.log test.sh全部打包成 tar 包

[root@localhost ~]# [root@localhost ~]# tar -cvf log.tar test.log test.shtest.logtest.sh

 （2）将 /etc 下的所有文件及目录打包到指定目录或当前目录，并使用 gz 压缩

[root@localhost ~]# tar -zcvf ./etc.tar.gz /etc

 （3）查看刚打包的文件内容（一定加z，因为是使用 gzip 压缩的）

[root@localhost ~]# tar -ztvf ./etc.tar.gz...省略...

 （4）要压缩打包 /home, /etc ，但不要 /home/mashibing ，只能针对文件，不能针对目录

[root@localhost ~]# tar --exclude /home/mshibing -zcvf myfile.tar.gz /home/* /etc

#### 9、date命令

 **作用：**用来显示或设定系统的日期与时间

 常见参数

|           |                                           |
| --------- | ----------------------------------------- |
| -d        | 显示 datestr  中所设定的时间 (非系统时间) |
| --help    | 显示辅助讯息                              |
| -s        | 将系统时间设为  datestr 中所设定的时间    |
| -u        | 显示目前的格林威治时间                    |
| --version | 显示版本编号                              |

### 10、cal 命令

 **作用：**用户显示公历（阳历）日历

 **语法：**cal [选项] [[[日] 月] 年

 （1）显示指定年月日期

[root@localhost ~]# cal 9 2020

 （2）显示2020年每个月日历

[root@localhost ~]# cal -y 2020

 （3）将星期一做为第一列,显示前中后三月

[root@localhost ~]# cal -3m

### 11、grep命令

**作用：**用于查找文件里符合条件的字符串

 **常用参数：**

| 参数 | 参数描述                               |
| ---- | -------------------------------------- |
| -A n | 显示匹配字符后n行                      |
| -B n | 显示匹配字符前n行                      |
| -C n | 显示匹配字符前后n行                    |
| -c   | 计算符合样式的列数                     |
| -i   | 忽略大小写                             |
| -l   | 只列出文件内容符合指定的样式的文件名称 |
| -f   | 从文件中读取关键词                     |
| -n   | 显示匹配内容的所在文件中行数           |
| -R   | 递归查找文件夹                         |

（1）查找指定进程

[root@localhost ~]# ps -ef | grep svnroot    6771  9267 0 15:17 pts/0  00:00:00 grep --color=auto svn

 （2）查找指定进程个数

[root@localhost ~]# ps -ef | grep svn -c1

 （3）从文件中读取关键词

[root@localhost ~]# cat test.log | grep -f test.log马士兵教育：www.mashibing.com

 （4）从文件夹中递归查找以.sh结尾的行，并只列出文件

[root@localhost ~]# grep -lR '.sh$'.bash_historytest.sh.viminfolog.tar

 （5）查找非x开关的行内容

[root@localhost ~]# grep '^[^x]' test.log马士兵教育：www.mashibing.com

 （6）显示包含 ed 或者 at 字符的内容行

[root@localhost ~]# grep -E 'ed|at' test.log

#### 12、ps命令

ps

**作用：**用于显示当前进程 (process) 的状态

ps 工具标识进程的5种状态码:

D 不可中断 uninterruptible sleep (usually IO)R 运行 runnable (on run queue)S 中断 sleepingT 停止 traced or stoppedZ 僵死 a defunct (”zombie”) process



 （1）显示当前所有进程环境变量及进程间关系

[root@localhost ~]# ps -ef

 （2）显示当前所有进程

[root@localhost ~]# ps -A

 （3）与grep联用查找某进程

[root@localhost ~]# ps -aux | grep apacheroot   20112 0.0 0.0 112824  980 pts/0  S+  15:30  0:00 grep --color=auto apache

 （4）找出与 cron 与 syslog 这两个服务有关的 PID 号码

[root@localhost ~]# ps aux | grep '(cron|syslog)'root   20454 0.0 0.0 112824  984 pts/0  S+  15:30  0:00 grep --color=auto (cron|syslog)

#### 13、kill命令

kill

kill 命令用于删除执行中的程序或工作

 **常用参数：**

| 参数 | 参数描述                                                     |
| ---- | ------------------------------------------------------------ |
| -l   | 信号，若果不加信号的编号参数，则使用“-l”参数会列出全部的信号名称 |
| -a   | 当处理当前进程时，不限制命令名和进程号的对应关系             |
| -p   | 指定kill 命令只打印相关进程的进程号，而不发送任何信号        |
| -s   | 指定发送信号                                                 |
| -u   | 指定用户                                                     |

 （1）先使用ps查找进程pro1，然后用kill杀掉

[root@localhost ~]# kill -9 $(ps -ef | grep pro1)-bash: kill: root: 参数必须是进程或任务 ID-bash: kill: (27319) - 没有那个进程-bash: kill: (27317) - 没有那个进程

#  
