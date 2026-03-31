# Issues

## 1. Can not decode content-encoding: br
code/C4/02_text_to_metadata_filter.py
```
加载BiliBili视频失败: 400, message:
  Can not decode content-encoding: br
没有成功加载任何视频，程序退出
```

### 解决方案
错误原因：aiohttp 不支持 B 站的 br 压缩编码
修复方法：代码里强制禁用 br 解码
但是未生效，先跳过