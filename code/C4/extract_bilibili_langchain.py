#!/usr/bin/env python3
"""
B站视频元数据提取工具
使用requests直接访问B站API提取视频信息
"""

import json
from datetime import datetime
import requests
from langchain_core.documents import Document

def extract_bilibili_metadata(urls):
    """
    提取B站视频元数据
    
    Args:
        urls: B站视频URL列表
    
    Returns:
        包含所有视频元数据的列表
    """
    all_metadata = []
    
    for url in urls:
        try:
            print(f"正在提取: {url}")
            
            # 从URL中提取BV号
            bv_id = url.split('/video/')[-1].split('?')[0]
            
            # 直接使用requests访问B站API
            api_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bv_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': f'https://www.bilibili.com/video/{bv_id}',
            }
            
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('code') != 0:
                raise Exception(f"B站API返回错误: {data.get('message', '未知错误')}")
            
            info = data.get('data', {})
            
            # 转换时间戳为可读格式
            publish_time = datetime.fromtimestamp(info.get("pubdate", 0)).strftime('%Y-%m-%d %H:%M:%S')
            
            # 转换时长为分钟:秒格式
            duration_seconds = info.get("duration", 0)
            minutes = duration_seconds // 60
            seconds = duration_seconds % 60
            duration_str = f"{minutes}分{seconds}秒"
            
            metadata = {
                "url": url,
                "bvid": bv_id,
                "title": info.get("title", "未知标题"),
                "description": info.get("desc", ""),
                "author": info.get("owner", {}).get("name", "未知作者"),
                "view_count": info.get("stat", {}).get("view", 0),
                "like_count": info.get("stat", {}).get("like", 0),
                "comment_count": info.get("stat", {}).get("reply", 0),
                "danmaku_count": info.get("stat", {}).get("danmaku", 0),
                "coin_count": info.get("stat", {}).get("coin", 0),
                "share_count": info.get("stat", {}).get("share", 0),
                "favorite_count": info.get("stat", {}).get("favorite", 0),
                "publish_date": publish_time,
                "duration": duration_str,
                "duration_seconds": duration_seconds,
                "tname": info.get("tname", ""),
                "thumbnail": info.get("pic", ""),
                "tid": info.get("tid", 0),
                "transcript": "",  # 需要登录才能获取字幕

                # 'title': info.get('title', '未知标题'),
                # 'author': info.get('owner', {}).get('name', '未知作者'),
                'source': info.get('bvid', '未知ID'),
                # 'view_count': info.get('stat', {}).get('view', 0),
                'length': info.get('duration', 0),
            }
            doc = Document(page_content="", metadata=metadata)
            all_metadata.append(doc)
            print(f"✓ 成功提取: {metadata['title']}")
                
        except Exception as e:
            print(f"✗ 提取失败: {str(e)}")
            all_metadata.append({"url": url, "error": str(e)})
    
    return all_metadata


def main():
    """主函数"""
    # B站视频URL列表
    bilibili_urls = [
        "https://www.bilibili.com/video/BV1Bo4y1A7FU",
        "https://www.bilibili.com/video/BV1ug4y157xA",
        "https://www.bilibili.com/video/BV1yh411V7ge",
    ]
    
    print("=" * 60)
    print("B站视频元数据提取工具")
    print("=" * 60)
    print(f"待提取视频数量: {len(bilibili_urls)}")
    print("=" * 60)
    
    # 提取元数据
    results = extract_bilibili_metadata(bilibili_urls)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("提取结果")
    print("=" * 60)
    
    video_list = []
    for i, doc in enumerate(results, 1):
        metadata = doc.metadata
        video_list.append(metadata)
        print(f"\n视频 {i}:")
        print("-" * 60)
        if "error" in metadata:
            print(f"  错误: {metadata['error']}")
        else:
            print(f"  标题: {metadata['title']}")
            print(f"  作者: {metadata['author']}")
            print(f"  BV号: {metadata['bvid']}")
            print(f"  播放量: {metadata['view_count']:,}")
            print(f"  点赞数: {metadata['like_count']:,}")
            print(f"  评论数: {metadata['comment_count']:,}")
            print(f"  弹幕数: {metadata['danmaku_count']:,}")
            print(f"  投币数: {metadata['coin_count']:,}")
            print(f"  收藏数: {metadata['favorite_count']:,}")
            print(f"  分享数: {metadata['share_count']:,}")
            print(f"  发布日期: {metadata['publish_date']}")
            print(f"  时长: {metadata['duration']}")
            print(f"  分区: {metadata['tname']}")
            desc = metadata['description']
            print(f"  描述: {desc[:100]}..." if len(desc) > 100 else f"  描述: {desc}")
            print(f"  封面: {metadata['thumbnail']}")
    
    # 保存到JSON文件
    output_file = "bilibili_metadata.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(video_list, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"结果已保存到: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
