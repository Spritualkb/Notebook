# SHODAN

##  通过python调用Shodan API获得搜索结果



 shodan文档：http://shodan.readthedocs.io/en/latest/tutorial.html

shodan：https://www.shodan.io/





### 常用 Shodan 库函数

- `shodan.Shodan(key)` ：初始化连接API
- `Shodan.count(query, facets=None)`：返回查询结果数量
- `Shodan.host(ip, history=False)`：返回一个IP的详细信息
- `Shodan.ports()`：返回Shodan可查询的端口号
- `Shodan.protocols()`：返回Shodan可查询的协议
- `Shodan.services()`：返回Shodan可查询的服务
- `Shodan.queries(page=1, sort='timestamp', order='desc')`：查询其他用户分享的查询规则
- `Shodan.scan(ips, force=False)`：使用Shodan进行扫描，ips可以为字符或字典类型
- `Shodan.search(query, page=1, limit=None, offset=None, facets=None, minify=True)`： 查询Shodan数据







### (1) 搜索指定内容（apache）的信息

```py
import shodan
SHODAN_API_KEY = "SkVS0RAbiTQpzzEsahqnq2Hv6SwjUfs3"
api = shodan.Shodan(SHODAN_API_KEY)
try:
    results = api.search('Apache')
    print 'Results found: %s' % results['total']
    for result in results['matches']:         
            print ("%s:%s|%s|%s"%(result['ip_str'],result['port'],result['location']['country_name'],result['hostnames']))
except shodan.APIError, e:
    print 'Error: %s' % e
```

还可以使用付费功能例如`Apache country:"US"`



### (2) 搜索指定内容，将获得的IP写入文件

```py
import shodan
SHODAN_API_KEY = "SkVS0RAbiTQpzzEsahqnq2Hv6SwjUfs3"
api = shodan.Shodan(SHODAN_API_KEY)
file_object = open('ip.txt', 'w')
try:
    results = api.search('Apache')
    print 'Results found: %s' % results['total']
    for result in results['matches']:         
#            print result['ip_str']
            file_object.writelines(result['ip_str']+'\n')
except shodan.APIError, e:
    print 'Error: %s' % e
file_object.close()
```



### (3) 通过命令行参数指定搜索条件，将搜索到的IP写入文件

```py
import shodan
import sys
SHODAN_API_KEY = "SkVS0RAbiTQpzzEsahqnq2Hv6SwjUfs3"
api = shodan.Shodan(SHODAN_API_KEY)
if len(sys.argv)<2:
    print '[!]Wrong parameter'
    sys.exit(0)
print '[*]Search string: %s' % sys.argv[1]
file_object = open('ip.txt', 'w')
try:
    results = api.search(sys.argv[1])
    print '[+]Results found: %s' % results['total']
    for result in results['matches']:         
#            print result['ip_str']
            file_object.writelines(result['ip_str']+'\n')
except shodan.APIError, e:
    print 'Error: %s' % e
file_object.close()
```



### (4) 读取文件中的IP列表，反查IP信息

```py
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')  
SHODAN_API_KEY = "SkVS0RAbiTQpzzEsahqnq2Hv6SwjUfs3"
api = shodan.Shodan(SHODAN_API_KEY)
def searchip( str ):
    try:
        host = api.host(str)
    except shodan.exception.APIError:
        print "[!]No information available"
        print "---------------------------------------------"
        return
    else:
        # Print general info
        try:
            print "IP: %s\r\nOrganization: %s\r\nOperating System: %s" % (host['ip_str'], host.get('org', 'n/a'), host.get('os', 'n/a'))
        except UnicodeEncodeError:
            print "[!]UnicodeEncode Error\r\n"     
        else:
            # Print all banners
            for item in host['data']:
                print "Port: %s\r\nBanner: %s" % (item['port'], item['data'])
        print "---------------------------------------------"   
        return
file_object = open('ip.txt', 'r')
for line in file_object:
    searchip(line)
```





## 通过Shodan官网下载搜索结果

查询一次消耗一个export credit，无论结果有多少个，最多为10000个

导出格式选择为json

### (1) 从下载的json结果文件中提取IP

```py
import json
file_object = open("shodan_data.json", 'r')
for line in file_object:
    data = json.loads(line)
    print data["ip_str"]   
file_object.close()
```



### (2) 从下载的json结果文件中提取指定国家的IP和端口

国家代号在二级元素中，对应结构：`data["location"]["country_code"]`

```py
import json
import sys
import re
def search(country):
    file_object = open("shodan_data.json", 'r')
    file_object2 = open(country+".txt", 'w')
    for line in file_object:
        data = json.loads(line)  
        if re.search(data["location"]["country_code"], country, re.IGNORECASE):
            str1 = "%s:%s" % (data["ip_str"],data["port"])
            print str1
            file_object2.writelines(str1+'\n')
    file_object.close()
    file_object2.close()
if __name__ == "__main__":
    if len(sys.argv)<2:
        print ('[!]Wrong parameter')
        sys.exit(0)
    else:
        print ('[*]Search country code: %s' % sys.argv[1])
        search(sys.argv[1])
        print ("[+]Done")
```



命令行参数:

```
search.py US
```

生成文件US.txt，保存IP和对应的端口







```py
import optparse
import shodan
import requests
def main():
    usage='[usage: -j Type what you want]' \
          '        [-i IP to search]' \
          '        [-s Todays camera equipment]'
    parser=optparse.OptionParser(usage)
    parser.add_option('-j',dest='jost',help='Type what you want')
    parser.add_option('-i',dest='host',help='IP to search')
    parser.add_option('-s',action='store_true',dest='query',help='Todays camera equipment')
    (options,args)=parser.parse_args()
    if options.jost:
        jost=options.jost
        Jost(jost)
    elif options.host:
        host=options.host
        Host(host)
    elif options.query:
        query()
    else:
        parser.print_help()
        exit()
 
def Jost(jost):
    SHODAN_API_KEY='ZmgQ9FZf1rnRuR0MLhT5SXw0xBE8LDLc'
    api=shodan.Shodan(SHODAN_API_KEY)
    try:
        result=api.search('{}'.format(jost))
        print('[*]Results found:{}'.format(result['total']))
        for x in result['matches']:
            print('IP{}'.format(x['ip_str']))
            print(x['data'])
            with open('shodan.txt','a') as p:
                p.write(x['ip_str']+'\n')
                p.write(x['data']+'\n')
    except shodan.APIError as e:
        print('[-]Error:',e)
 
def Host(host):
    SHODAN_API_KEY='ZmgQ9FZf1rnRuR0MLhT5SXw0xBE8LDLc'
    try:
      api=shodan.Shodan(SHODAN_API_KEY)
      hx=api.host('{}'.format(host))
      print("IP:{}".format(hx['ip_str']))
      print('Organization:{}'.format(hx.get('org','n/a')))
      print('Operating System:{}'.format(hx.get('os','n/a')))
      for item in hx['data']:
          print("Port:{}".format(hx['port']))
          print('Banner:{}'.format(hx['data']))
    except shodan.APIError as g:
        print('[-]Error:',g)
 
def query():
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
    url = "https://api.shodan.io/shodan/query?key=ZmgQ9FZf1rnRuR0MLhT5SXw0xBE8LDLc"
    r = requests.get(url, headers=header)
    sd = r.json()
    sg = sd['matches'][0:]
    for b in sg:
        print("描述:", b['description'])
        print('标签:', b['tags'])
        print('时间戳:', b['timestamp'])
        print('标题:', b['title'])
        print('服务:', b['query'])
        print('---------------------------------')
 
if __name__ == '__main__':
    main()
```







搜索execption
nginx 500 这些 可以看到漏洞的页面



shodan.io 是可以按照漏洞关键词来搜



## shodan api官网

[正段开发商 (shodan.io)](https://developer.shodan.io/api)