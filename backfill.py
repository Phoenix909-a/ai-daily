#!/usr/bin/env python3
"""
一次性补数据脚本 — 为缺失的历史日期生成并添加 data.json 记录。
运行完毕后可安全删除本文件。

使用方法（在 GitHub Actions 中）：
    python backfill.py
"""

import subprocess
import sys
import json
import os

MISSING_DATES = [
    "2026-06-01",
    "2026-06-04",
    "2026-06-05",
    "2026-06-06",
    "2026-06-07",
]

def main():
    for date_str in MISSING_DATES:
        print(f"\n{'='*60}")
        print(f"  正在生成 {date_str} 的数据 ...")
        print(f"{'='*60}")
        result = subprocess.run(
            [sys.executable, "ai_daily.py",
             "--date", date_str,
             "--update-json", "--max-news", "35",
             "--no-reddit",         # Reddit 在国内连不上
             "--no-hackernews",     # 历史数据不需要
             "--no-github"],        # 历史数据不需要
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr[-2000:])
        if result.returncode != 0:
            print(f"  ⚠️  {date_str} 运行出错 (rc={result.returncode})，继续下一个")
        else:
            print(f"  ✅ {date_str} 完成")

    # 验证最终 data.json
    print(f"\n{'='*60}")
    print("  所有日期处理完毕，最终 data.json 内容：")
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        print(f"    {d['date']} ({d['weekday']}) - {len(d['news'])} 条新闻")
    print(f"  共 {len(data)} 天数据")

if __name__ == "__main__":
    main()
