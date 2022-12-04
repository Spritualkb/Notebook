# 命令

① 改变盘符命令： <磁盘号>： 

例：c:、d:、e:、……

② 改变当前目录命令：cd <文件路径名>

③ 创建新文件夹命令：md <文件夹名>

④ 删除文件夹命令：rd <文件夹名>

⑤ 删除文件命令：del <文件名> 或者使用命令：erase <文件名>

⑥ 更改文件名命令：rename <原文件名> <新文件名>

⑦ 查看文件命令：dir 

复制文件

copy "目标文件夹路径"

.\   当前路径

..\    上一个路径

配置环境变量

path=%要配置的变量名(列如path)%;"这里填写要添加的路径"  #都去掉双引号

********

# 汇编语言

汇编语言保存在.asm后缀的文件里

;My first 32-bit assembly program.分号后是注释

.386  ;指出程序使用的指令集，还可以是486、586、686等

.model flat,stdcall ;指出存储类型，flat是平展模型，我们本学期就使用这个编程模型，

​        ; stdcall是指出API调用时参数的格式，本学期我们固定不变。

include  \masm6\io32.inc ;定义包含文件，注意路径，以io32.inc文件所在的实际路径为准

.data ;定义数据段

msg byte ‘My first 32-bit assembly program.’,0dh,0ah,0

.code ;定义代码段

start: mov eax,offset msg ;第一条汇编语句前需要一个标号，本例设为start，

​           ;offset msg 是字符串首字符的偏移地址

   call dispmsg ; 调用显示子程序

   exit 0 ;返回操作系统命令，汇编语言必须要有这个命令

   end start ;汇编结束命令，这里的标号必须与第一条指令的标号相对应

d:\masm6\bin\ml ?

***********************

用命令

```
ml /c /coff firstDemo.asm
```

产生汇编目标程序.obj，也可以产生列表文件等其它信息的文件。我们使用masm6\bin文件夹中的ml.exe汇编文件，这是微软件提供的汇编程序

参数说明：/c，表示仅使用ml完成源程序的汇编工作。

​     /coff,表示生成COFF目标模块文件，COFF是32位windows和UNIX操作系统使用的目标文件格式。

## 1、 连接程序，本过程产生执行文件。

使用bin文件夹中的link32.exe文件，将目标文件与库文件等连接起来，产生exe执行文件。

命令格式： 

```
link32  /subsystem:console firstDemo.obj
```

参数说明：/subsystem:conscole，表示生成windows控制台环境的可执行文件。如果要生成图形窗口的可执行文件，则是/subsystem:windows。

连接过程结束后，用dir命令查看结果文件firstDemo.exe。

## 2、 运行程序，本过程是查看执行文件的执行过程。

在命令提示符窗口，运行上一步生成的可执行文件。

命令格式： firstdemo

批处理文件.bat

里面写上命令然后改后缀名即可制作完成

```
#查看文件内容
type \io32.ini
```

 

```
.data
len byte 5
var word 12,45,365.778,999
max.word ?
.code
start:nope
	  mov ax,[var]
	  mov [max],ax
	  mov ecx,[len]
	  mov ebx,offset var;addr
	s:mov ax,[ebx]
	  cmp ax,[max]
	  jna ok
	  mov [max],ax
	  
	  ok:add ebx,2
	  loop s
	  ret 
	  end start
	  
	  
```

