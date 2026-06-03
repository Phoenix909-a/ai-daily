#!/usr/bin/env python3
"""
AI Daily Briefing Generator — 每日 AI 进展自动简报生成器

聚合 arXiv、HuggingFace、Reddit、NewsAPI 等多个信源，调用 DeepSeek API
生成结构化中文简报并输出 Markdown 文件。

Usage:
    python ai_daily.py                                    # 今天
    python ai_daily.py --date 2026-06-01                  # 指定日期
    python ai_daily.py --date yesterday                    # 昨天
    python ai_daily.py --date -2                           # 前天
    python ai_daily.py -o ./briefings                     # 自定义输出目录
    python ai_daily.py --skip-summary                     # 跳过 AI 总结
    python ai_daily.py --proxy http://127.0.0.1:7890      # 指定代理
    python ai_daily.py --verbose                          # 详细日志

环境变量:
    DEEPSEEK_API_KEY    必需   DeepSeek API Key
    NEWSAPI_KEY         可选   NewsAPI Key (https://newsapi.org/register)
    HTTP_PROXY          可选   HTTP 代理地址
    HTTPS_PROXY         可选   HTTPS 代理地址 (requests 自动读取)
"""

import os
import re
import sys
import json
import time
import logging
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from typing import Optional
from email.utils import parsedate_to_datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
CONNECT_TIMEOUT = 10        # 连接超时（海外站点很快失效）
READ_TIMEOUT = 40            # 读取超时
LONG_TIMEOUT = 150           # DeepSeek 长超时
MAX_RETRIES = 2              # 快速失败后换源
GENTLE_DELAY = 1.0           # 请求间隔（秒）
USER_AGENT = "AIDailyBriefing/1.0"

# arXiv RSS feeds (比 API 更稳定，中国大陆可访问)
ARXIV_RSS = {
    "cs.AI": "https://rss.arxiv.org/rss/cs.AI",
    "cs.CL": "https://rss.arxiv.org/rss/cs.CL",
    "cs.CV": "https://rss.arxiv.org/rss/cs.CV",
    "cs.LG": "https://rss.arxiv.org/rss/cs.LG",
}

# HuggingFace 国内镜像
HF_ENDPOINTS = [
    "https://hf-mirror.com/api/daily_papers",       # 国内镜像，优先尝试
    "https://huggingface.co/api/daily_papers",       # 官方源（需代理）
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

logger = logging.getLogger("ai_daily")


# ═══════════════════════════════════════════════════════════════════════════════
#  Proxy support
# ═══════════════════════════════════════════════════════════════════════════════

def _has_proxy() -> bool:
    """Check if any proxy is configured (env var or already set)."""
    return bool(os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY"))


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_session() -> requests.Session:
    """Session with retry strategy."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _rget(url: str, **kwargs) -> Optional[requests.Response]:
    """Safe HTTP GET. Returns None on any failure (fast fail)."""
    kwargs.setdefault("timeout", (CONNECT_TIMEOUT, READ_TIMEOUT))
    kwargs.setdefault("headers", {}).setdefault("User-Agent", USER_AGENT)
    try:
        resp = _make_session().get(url, **kwargs)
        if resp.status_code >= 400:
            logger.debug("  HTTP %d for %s", resp.status_code, url[:60])
            return None
        return resp
    except requests.RequestException as e:
        logger.debug("  %s %s", type(e).__name__, url[:50])
        return None


def _rpost(url: str, **kwargs) -> Optional[requests.Response]:
    """Safe HTTP POST. Returns None on failure."""
    kwargs.setdefault("timeout", (CONNECT_TIMEOUT, LONG_TIMEOUT))
    kwargs.setdefault("headers", {}).setdefault("User-Agent", USER_AGENT)
    try:
        resp = _make_session().post(url, **kwargs)
        if resp.status_code >= 400:
            logger.debug("  HTTP %d for %s", resp.status_code, url[:50])
            return None
        return resp
    except requests.RequestException as e:
        logger.debug("  %s %s", type(e).__name__, url[:50])
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Source 1: arXiv
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_arxiv(target_date: date) -> list[dict]:
    """Fetch papers for target date. RSS for today, Semantic Scholar for past."""
    logger.info("[arXiv]  Fetching papers for %s …", target_date)

    # Strategy 1: RSS feeds (daily, only works for today)
    papers = _fetch_arxiv_rss(target_date)
    if papers:
        return papers

    # Strategy 2: Semantic Scholar (works for any date)
    logger.info("  RSS gave 0 (past date), trying Semantic Scholar …")
    papers = _fetch_semantic_scholar(target_date)
    if papers:
        return papers

    logger.warning("  All strategies gave 0 papers (expected for past dates)")
    return []


def _fetch_arxiv_rss(target_date: date) -> list[dict]:
    """Fetch from arXiv RSS feeds. Only contains recent papers."""
    headers = {"User-Agent": USER_AGENT + " (mailto:user@example.com)"}
    papers = []
    seen_ids = set()

    for cat_name, rss_url in ARXIV_RSS.items():
        logger.debug("  RSS %s …", cat_name)
        resp = _rget(rss_url, headers=headers)
        if not resp:
            continue

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            continue

        for item in root.findall(".//item"):
            title = item.findtext("title", "Untitled").strip()
            link = item.findtext("link", "").strip()
            desc = item.findtext("description", "")
            pub_str = item.findtext("pubDate", "")
            arxiv_id = item.findtext("{http://arxiv.org/schemas/atom}id", "")

            dedup_key = arxiv_id or link
            if dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)

            # Filter by date
            if pub_str:
                try:
                    pub_date = parsedate_to_datetime(pub_str).date()
                    if pub_date != target_date:
                        continue
                except Exception:
                    continue

            clean_desc = re.sub(r"<[^>]+>", "", desc).strip()
            clean_desc = re.sub(r"\s+", " ", clean_desc)[:1000]

            arxiv_match = re.search(r"abs/(\d+\.\d+)", link)
            paper_link = f"https://arxiv.org/abs/{arxiv_match.group(1)}" if arxiv_match else link

            papers.append({
                "title": title,
                "summary": clean_desc,
                "link": paper_link,
                "authors": [],
                "categories": [cat_name],
            })

        time.sleep(0.3)

    papers = [p for p in papers if p["link"] and "arxiv.org" in p["link"]]
    if papers:
        logger.info("  -> %d papers (from RSS)", len(papers))
    return papers


def _fetch_semantic_scholar(target_date: date) -> list[dict]:
    """Fetch AI papers from Semantic Scholar API."""
    logger.debug("  Semantic Scholar …")

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": "artificial intelligence machine learning deep learning large language model",
        "year": str(target_date.year),
        "limit": 20,
        "fields": "title,url,publicationDate,openAccessPdf",
    }

    resp = _rget(url, params=params)
    if not resp:
        return []

    try:
        body = resp.json()
    except json.JSONDecodeError:
        return []

    papers = []
    target_s = target_date.strftime("%Y-%m-%d")
    for p in body.get("data", []):
        pub_date = p.get("publicationDate", "")
        if not pub_date or pub_date[:10] != target_s:
            continue
        pdf = p.get("openAccessPdf") or {}
        link = pdf.get("url") or p.get("url", "")
        papers.append({
            "title": p.get("title", "Untitled"),
            "summary": "",
            "link": link,
            "authors": [],
            "categories": [],
        })

    if papers:
        logger.info("  -> %d papers (from Semantic Scholar)", len(papers))
    return papers


# ═══════════════════════════════════════════════════════════════════════════════
#  Source 2: Hugging Face Daily Papers (with Chinese mirror)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_huggingface(target_date: date) -> list[dict]:
    """Fetch daily papers from HuggingFace via mirror. Then Semantic Scholar fallback."""
    logger.info("[HuggingFace]  Fetching …")

    for endpoint in HF_ENDPOINTS:
        logger.debug("  Trying %s …", endpoint)
        resp = _rget(endpoint)
        if not resp:
            continue
        try:
            data = resp.json()
        except (json.JSONDecodeError, TypeError):
            continue
        return _parse_hf_response(data, target_date)

    logger.warning("  All HF endpoints failed")
    return []


def _parse_hf_response(data: list, target_date: date) -> list[dict]:
    """Parse HuggingFace API response and filter by date."""
    target_s = target_date.strftime("%Y-%m-%d")
    papers = []

    for item in data[:60]:
        created = item.get("createdAt", "") or item.get("updatedAt", "")
        paper = item.get("paper", item)
        pub_date = paper.get("publicationDate", "")

        if created and not created.startswith(target_s):
            if pub_date and not pub_date.startswith(target_s):
                continue

        title = paper.get("title", item.get("title", "Untitled"))
        summary = paper.get("summary", item.get("summary", ""))
        paper_id = paper.get("id", "")
        paper_url = f"https://arxiv.org/abs/{paper_id}" if paper_id else paper.get("url", "")

        papers.append({
            "title": title.strip(),
            "summary": re.sub(r"\s+", " ", summary)[:1000] if summary else "",
            "link": paper_url,
            "source": "huggingface",
        })

    logger.info("  -> %d papers (from HF)", len(papers))
    return papers


# ═══════════════════════════════════════════════════════════════════════════════
#  Source 3: Reddit r/MachineLearning (needs proxy in China)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_reddit(target_date: date) -> list[dict]:
    """Fetch top posts from r/MachineLearning. May need proxy in China."""
    if not _has_proxy():
        logger.info("[Reddit]  Skipped (no proxy, blocked in China)")
        return []

    logger.info("[Reddit]  Fetching …")

    dt_start = datetime(target_date.year, target_date.month, target_date.day)
    dt_end = dt_start + timedelta(days=1)

    url = "https://www.reddit.com/r/MachineLearning/search.json"
    params = {
        "q": f"timestamp:{int(dt_start.timestamp())}..{int(dt_end.timestamp())}",
        "restrict_sr": "on", "sort": "top", "syntax": "cloudsearch", "limit": 100,
    }
    headers = {"User-Agent": f"{USER_AGENT} (by /u/ai_briefing_bot)"}

    resp = _rget(url, params=params, headers=headers)
    if not resp:
        return _fetch_reddit_hot(target_date)

    try:
        body = resp.json()
    except json.JSONDecodeError:
        return _fetch_reddit_hot(target_date)

    posts = []
    for child in body.get("data", {}).get("children", []):
        d = child.get("data", {})
        ts = d.get("created_utc", 0)
        if datetime.utcfromtimestamp(ts).date() != target_date:
            continue
        permalink = d.get("permalink", "")
        posts.append({
            "title": d.get("title", "Untitled"),
            "selftext": (d.get("selftext") or "")[:500],
            "url": f"https://www.reddit.com{permalink}",
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
        })

    posts.sort(key=lambda x: x["score"], reverse=True)
    logger.info("  -> %d posts", len(posts))
    return posts[:15]


def _fetch_reddit_hot(target_date: date) -> list[dict]:
    """Fallback: paginate hot listing."""
    logger.debug("  [Fallback] Hot listing …")

    url = "https://www.reddit.com/r/MachineLearning/hot.json"
    headers = {"User-Agent": f"{USER_AGENT} (by /u/ai_briefing_bot)"}
    posts = []
    after = None

    for _ in range(3):
        params = {"limit": 100}
        if after:
            params["after"] = after

        resp = _rget(url, params=params, headers=headers)
        if not resp:
            break

        try:
            body = resp.json()
        except json.JSONDecodeError:
            break

        children = body.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            d = child.get("data", {})
            ts = d.get("created_utc", 0)
            pd = datetime.utcfromtimestamp(ts).date()

            if pd < target_date - timedelta(days=2):
                after = None
                break
            if pd != target_date:
                continue

            permalink = d.get("permalink", "")
            posts.append({
                "title": d.get("title", "Untitled"),
                "selftext": (d.get("selftext") or "")[:500],
                "url": f"https://www.reddit.com{permalink}",
                "score": d.get("score", 0),
                "num_comments": d.get("num_comments", 0),
            })

        after = body.get("data", {}).get("after")
        if not after:
            break
        time.sleep(1.0)

    posts.sort(key=lambda x: x["score"], reverse=True)
    logger.debug("  -> %d posts", len(posts))
    return posts[:15]


# ═══════════════════════════════════════════════════════════════════════════════
#  Source 4: NewsAPI (needs NEWSAPI_KEY + proxy in China)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_newsapi(target_date: date) -> list[dict]:
    """Fetch AI-related news from NewsAPI."""
    api_key = os.environ.get("NEWSAPI_KEY", "")
    if not api_key:
        logger.info("[NewsAPI]  No NEWSAPI_KEY set, skipped")
        return []

    if not _has_proxy():
        logger.info("[NewsAPI]  Skipped (no proxy)")
        return []

    logger.info("[NewsAPI]  Fetching …")

    date_s = target_date.strftime("%Y-%m-%d")
    params = {
        "q": '"artificial intelligence" OR "AI" OR "large language model" OR "machine learning"',
        "from": date_s, "to": date_s, "language": "en",
        "sortBy": "popularity", "pageSize": 30, "apiKey": api_key,
    }

    resp = _rget("https://newsapi.org/v2/everything", params=params)
    if not resp:
        return []

    try:
        body = resp.json()
    except json.JSONDecodeError:
        return []

    if body.get("status") != "ok":
        logger.warning("  API error: %s", body.get("message", "unknown"))
        return []

    articles = []
    for art in body.get("articles", []):
        title = (art.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue
        articles.append({
            "title": title,
            "description": (art.get("description") or "")[:500],
            "url": art.get("url", ""),
            "source": art.get("source", {}).get("name", ""),
        })

    logger.info("  -> %d articles", len(articles))
    return articles[:20]


# ═══════════════════════════════════════════════════════════════════════════════
#  Source 5: RSS Feeds (Chinese tech media — 国内直连)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_rss_feeds(target_date: date) -> list[dict]:
    """Fetch from Chinese AI media: 量子位, 雷锋网, 36氪, IT之家."""
    logger.info("[RSS]  Fetching Chinese tech media …")

    feeds = [
        ("量子位", "https://www.qbitai.com/feed"),
        ("雷锋网", "https://www.leiphone.com/feed"),
        ("36氪", "https://36kr.com/feed"),
        ("IT之家", "https://www.ithome.com/rss/"),
    ]

    items = []
    for name, feed_url in feeds:
        logger.debug("  Fetching %s …", name)
        resp = _rget(feed_url)
        if not resp:
            logger.warning("  %s unavailable", name)
            continue

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as e:
            logger.warning("  %s XML error: %s", name, e)
            continue

        # RSS 2.0
        for entry in root.findall(".//item"):
            title = entry.findtext("title", "Untitled")
            link = entry.findtext("link", "")
            desc = entry.findtext("description", "")
            pub_str = entry.findtext("pubDate", "")

            if pub_str:
                try:
                    pd = parsedate_to_datetime(pub_str).date()
                    if pd != target_date:
                        continue
                except Exception:
                    pass

            items.append({
                "title": title.strip(),
                "description": re.sub(r"<[^>]+>", "", desc)[:500],
                "link": link.strip(),
                "source": name,
            })

        # Atom format
        atom = "{http://www.w3.org/2005/Atom}"
        for entry in root.findall(f".//{atom}entry"):
            el = entry.find(f"{atom}title")
            title = el.text.strip() if el is not None else "Untitled"

            el = entry.find(f"{atom}updated")
            pub_str = el.text if el is not None else ""
            if pub_str:
                try:
                    pd = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).date()
                    if pd != target_date:
                        continue
                except Exception:
                    pass

            el = entry.find(f"{atom}link")
            link = el.attrib.get("href", "") if el is not None else ""

            el = entry.find(f"{atom}summary")
            desc = re.sub(r"<[^>]+>", "", el.text or "")[:500]

            items.append({
                "title": title.strip(),
                "description": desc,
                "link": link.strip(),
                "source": name,
            })

        time.sleep(GENTLE_DELAY)

    # Deduplicate by URL
    seen = set()
    unique = []
    for item in items:
        if item["link"] not in seen:
            seen.add(item["link"])
            unique.append(item)

    logger.info("  -> %d items", len(unique))
    return unique


# ═══════════════════════════════════════════════════════════════════════════════
#  DeepSeek API
# ═══════════════════════════════════════════════════════════════════════════════

def call_deepseek(system_prompt: str, user_prompt: str,
                  temperature: float = 0.3, max_tokens: int = 4096) -> Optional[str]:
    """Call DeepSeek chat/completions. Returns response text or None."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        logger.error("[DeepSeek]  DEEPSEEK_API_KEY not set!")
        return None

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    logger.info("[DeepSeek]  Generating summary (%d chars input) …", len(user_prompt))

    resp = _rpost(url, json=payload, headers=headers)
    if not resp:
        return None

    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        logger.info("  -> %d chars output", len(text))
        return text
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error("  Parse error: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Briefing Generation
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个专业的 AI 领域新闻简报编辑。你的任务是根据提供的原始素材，生成一份结构清晰、信息密度高的每日 AI 简报。

请严格按照以下格式输出（Markdown 格式）：

# AI 每日简报 — YYYY-MM-DD

## 今日摘要
3-5 句话概括今日最重要的 AI 进展。

## 重磅论文
每项包含：标题、一句话中文解读（30字以内）、arXiv 链接。只列出真正重要的论文。

## 重要新闻
每项包含：标题、简要内容（一两句话）、原文链接。

## 社区热议
每项包含：帖子标题、关键观点摘录（一句话）、Reddit 链接。

## 中文科技媒体
（如果素材中有的话）

注意事项：
- 所有解读和摘要使用中文
- 突出重点，避免罗列
- 保持客观，不添加素材中没有的信息
- 每条不超过 100 字"""


def _format_prompt(papers: list, news: list, reddit: list, rss: list) -> str:
    """Format raw data into a prompt for the LLM."""
    parts = []

    # Papers
    parts.append("## 今日论文\n")
    for i, p in enumerate(papers[:25], 1):
        parts.append(f"{i}. [{p['title']}]({p['link']})")
        if p.get("summary"):
            parts.append(f"   {p['summary'][:300]}")
        parts.append("")

    # News
    if news:
        parts.append("\n## 科技新闻\n")
        for i, art in enumerate(news[:15], 1):
            parts.append(f"{i}. [{art['title']}]({art['url']})")
            if art.get("description"):
                parts.append(f"   {art['description'][:300]}")
            parts.append("")

    # Reddit
    if reddit:
        parts.append("\n## Reddit 社区讨论\n")
        for i, post in enumerate(reddit[:10], 1):
            parts.append(
                f"{i}. [{post['title']}]({post['url']})  "
                f"[+{post['score']} comments:{post['num_comments']}]"
            )
            if post.get("selftext"):
                parts.append(f"   {post['selftext'][:300]}")
            parts.append("")

    # RSS
    if rss:
        parts.append("\n## 中文科技媒体\n")
        for i, item in enumerate(rss[:10], 1):
            parts.append(f"{i}. [{item['title']}]({item['link']})  [{item.get('source', '')}]")
            if item.get("description"):
                parts.append(f"   {item['description'][:300]}")
            parts.append("")

    return "\n".join(parts)


def _raw_briefing(papers: list, news: list, reddit: list, rss: list,
                  target_date: date) -> str:
    """Generate a simple markdown briefing without AI (fallback)."""
    date_s = target_date.strftime("%Y-%m-%d")
    lines = [
        f"# AI Daily Briefing -- {date_s}",
        "",
        "> Note: Generated from raw data (DeepSeek summary unavailable)",
        "",
    ]

    if papers:
        lines += ["## Papers", ""]
        for p in papers[:20]:
            lines.append(f"- [{p['title']}]({p['link']})")
            if p.get("authors"):
                lines.append(f"  *Authors: {', '.join(p['authors'][:5])}*")
            lines.append("")

    if news:
        lines += ["## News", ""]
        for art in news[:15]:
            lines.append(f"- [{art['title']}]({art['url']})")
            if art.get("description"):
                lines.append(f"  {art['description'][:200]}")
            lines.append("")

    if reddit:
        lines += ["## Reddit", ""]
        for post in reddit[:10]:
            lines.append(f"- [{post['title']}]({post['url']})  (+{post['score']})")
            lines.append("")

    if rss:
        lines += ["## Chinese Media", ""]
        for item in rss[:10]:
            lines.append(f"- [{item['title']}]({item['link']})")
            lines.append("")

    return "\n".join(lines)


def generate_briefing(papers: list, news: list, reddit: list, rss: list,
                      target_date: date, skip_summary: bool = False) -> str:
    """Generate the briefing via DeepSeek or raw fallback."""
    if skip_summary:
        logger.info("[Main]  Skip-summary mode")
        return _raw_briefing(papers, news, reddit, rss, target_date)

    user_prompt = _format_prompt(papers, news, reddit, rss)

    if len(user_prompt) > 120_000:
        logger.warning("  Truncating prompt from %d to 120k chars", len(user_prompt))
        user_prompt = user_prompt[:120_000]

    result = call_deepseek(SYSTEM_PROMPT, user_prompt)
    if result:
        return result

    logger.warning("[Main]  DeepSeek failed, using raw briefing")
    return _raw_briefing(papers, news, reddit, rss, target_date)


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON output (data.json for web page)
# ═══════════════════════════════════════════════════════════════════════════════

DATA_JSON_FILENAME = "data.json"
WEEKDAY_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _load_data_json(filepath: str) -> list:
    """Load existing data.json. Returns empty list if missing/invalid."""
    if not os.path.exists(filepath):
        logger.info("  data.json not found, will create new")
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning("  data.json root is not a list, overwriting")
        return []
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("  data.json read failed: %s, will overwrite", e)
        return []


def _save_data_json(data: list, filepath: str) -> bool:
    """Save data.json with pretty formatting."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("  data.json updated (%d days)", len(data))
        return True
    except IOError as e:
        logger.error("  data.json write failed: %s", e)
        return False


def _extract_summary_from_md(markdown: str) -> str:
    """Extract the first paragraph under a summary heading from markdown."""
    if not markdown:
        return ""
    lines = markdown.split("\n")
    in_summary = False
    summary_parts = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,3}\s*(今日摘要|摘要|Summary|今日简报)", stripped, re.IGNORECASE):
            in_summary = True
            continue
        if in_summary:
            if re.match(r"^#{1,3}\s", stripped):
                break
            if stripped and not stripped.startswith(">"):
                summary_parts.append(stripped)
    text = " ".join(summary_parts).strip()
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    return "".join(sentences[:5]).strip()


def _clean_title(title: str) -> str:
    """Clean up clickbait / news prefixes from titles."""
    # Common prefixes to strip
    prefixes = [
        r"^刚刚[，,：:]?\s*",
        r"^独家[丨|]?\s*",
        r"^独家[：:]\s*",
        r"^独家实拍[丨|]?\s*",
        r"^雷峰网[：:]?\s*",
        r"^36氪[丨|]?\s*",
        r"^36氪独家[丨|]?\s*",
        r"^IT之家[：:]?\s*",
    ]
    for p in prefixes:
        title = re.sub(p, "", title)
    # Clean trailing markers like ｜
    title = re.sub(r"\s*[丨|]\s*$", "", title)
    return title.strip()


def _categorize_news(item: dict) -> str:
    """Rule-based category guessing. Used as fallback when DeepSeek unavailable."""
    title = item.get("title", "")
    source = item.get("source", "")
    url = item.get("link", "")
    all_text = f"{title} {source} {url}".lower()

    # Keyword-based classification
    if any(k in all_text for k in ["arxiv.org", "论文", "benchmark", "sota", "模型", "transformer"]):
        return "学术论文"
    if any(k in all_text for k in ["融资", "估值", "投资", "轮融资", "亿元"]):
        return "投融资"
    if any(k in all_text for k in ["发布", "开源", "上线", "推出", "全新"]):
        return "产品发布"
    if any(k in all_text for k in ["收购", "ipo", "招股书", "财报", "营收", "挖走", "加盟"]):
        return "公司动态"
    if any(k in all_text for k in ["开源", "github", "开放源码"]):
        return "开源项目"
    if any(k in all_text for k in ["行业", "市场", "增长", "目标", "赛道"]):
        return "商业动态"
    return "行业动态"


def _enrich_news_deepseek(raw_news: list, target_date: date) -> list:
    """Use DeepSeek to enrich news items with tags, details, and scores."""
    if not raw_news:
        return []

    # Build a compact prompt
    news_text = ""
    for i, item in enumerate(raw_news[:25], 1):
        desc = item.get("description", "")[:100]
        news_text += f"{i}. {item['title']}\n   {desc}\n"

    system_prompt = """你是一个 AI 新闻编辑。你的任务是将原始新闻列表加工成结构化数据。
对每条新闻，你需要：
1. 清洗标题：去掉"刚刚"、"独家｜"等前缀，改写成"主体 + 事件"的新闻标题风格
2. 添加分类标签：从 [公司动态, 产品发布, 行业动态, 投融资, 学术论文, 商业动态, 开源项目] 中选择
3. 写一句话简介（15-30字）
4. 写详细摘要（2-4句话，50-100字）
5. 给重要度打分（1-100），综合考量行业影响力、话题热度和技术突破性

严格按照以下 JSON 格式输出（只输出 JSON，不要其他文字）：
{
  "news": [
    {
      "title": "清洗后的标题",
      "description": "一句话简介",
      "detail": "详细摘要（2-4句话）",
      "tag": "分类标签",
      "sort_score": 85
    }
  ]
}"""

    user_prompt = f"请处理以下 {target_date} 的 AI 新闻：\n\n{news_text}"

    result = call_deepseek(system_prompt, user_prompt,
                           temperature=0.2, max_tokens=4096)
    if not result:
        return []

    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", result)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = result.strip()

    try:
        parsed = json.loads(json_str)
        enriched = parsed.get("news", [])
        if enriched:
            logger.info("  DeepSeek enriched %d news items", len(enriched))
            return enriched
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("  DeepSeek JSON parse failed: %s", e)

    return []


def _build_json_entry(target_date: date, papers: list, news_api: list,
                      reddit: list, rss: list, md_content: str,
                      max_news: int = 20) -> dict:
    """Build a single data.json entry from all collected data sources."""
    weekday_str = WEEKDAY_CN[target_date.weekday()]

    # ── Summary ──
    summary = _extract_summary_from_md(md_content)
    if not summary:
        paper_count = len(papers)
        news_count = len(news_api) + len(rss)
        summary = f"今日共收录 {paper_count} 篇论文、{news_count} 条新闻。"
        if rss:
            sources = set(item.get("source", "") for item in rss)
            summary += f" 来源包括：{', '.join(sorted(sources))}。"

    # ── Build raw news items with URL dedup ──
    raw_items = []
    seen_urls = set()

    # RSS first
    for item in rss:
        url = item.get("link", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            raw_items.append(item)

    # NewsAPI
    for item in news_api:
        url = item.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            raw_items.append({
                "title": item["title"],
                "description": item.get("description", ""),
                "link": url,
                "source": item.get("source", ""),
            })

    # Top papers (with summaries)
    for p in papers:
        url = p.get("link", "")
        if not url or url in seen_urls:
            continue
        desc = p.get("summary", "")[:200]
        if not desc:
            desc = f"arXiv · {', '.join(p.get('categories', []))}"
        raw_items.append({
            "title": p["title"],
            "description": desc,
            "link": url,
            "source": "arxiv",
        })
        seen_urls.add(url)
        if len(raw_items) >= max_news + 10:
            break

    raw_items = raw_items[:max_news]

    # ── Enrich with DeepSeek ──
    enriched = _enrich_news_deepseek(raw_items, target_date)

    if enriched and len(enriched) == len(raw_items):
        # Merge enriched fields back, preserving URLs
        news_items = []
        for raw, enr in zip(raw_items, enriched):
            news_items.append({
                "title": enr.get("title", _clean_title(raw.get("title", ""))),
                "link": raw.get("link", raw.get("url", "")),
                "description": enr.get("description", raw.get("description", ""))[:200],
                "detail": enr.get("detail", raw.get("description", ""))[:500],
                "tag": enr.get("tag", _categorize_news(raw)),
                "sort_score": enr.get("sort_score", 70),
            })
    else:
        # Fallback: rule-based
        news_items = []
        for item in raw_items:
            title = _clean_title(item.get("title", ""))
            desc = item.get("description", "")[:200]
            news_items.append({
                "title": title,
                "link": item.get("link", item.get("url", "")),
                "description": desc,
                "detail": desc,
                "tag": _categorize_news(item),
                "sort_score": 70,
            })

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "weekday": weekday_str,
        "summary": summary,
        "news": news_items,
    }


def update_data_json(filepath: str, papers: list, news_api: list,
                     reddit: list, rss: list, md_content: str,
                     target_date: date, max_news: int = 20) -> bool:
    """Update data.json: add/replace today's entry, keep existing ones."""
    logger.info("[JSON]  Updating %s …", filepath)

    old_data = _load_data_json(filepath)
    entry = _build_json_entry(target_date, papers, news_api, reddit, rss,
                              md_content, max_news)

    # Replace if same date exists, else prepend
    date_str = entry["date"]
    found = False
    for i, d in enumerate(old_data):
        if d.get("date") == date_str:
            old_data[i] = entry
            found = True
            logger.info("  Replaced existing entry for %s", date_str)
            break

    if not found:
        old_data.insert(0, entry)
        logger.info("  Added new entry for %s", date_str)

    # Sort by date descending
    old_data.sort(key=lambda d: d.get("date", ""), reverse=True)

    return _save_data_json(old_data, filepath)


# ═══════════════════════════════════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_print(text: str):
    """Print safely for Windows GBK terminals."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


def output_briefing(content: str, target_date: date, output_dir: str = ".") -> str:
    """Print to console and save to file. Returns file path."""
    # Console
    _safe_print("\n" + "=" * 60)
    _safe_print(content)
    _safe_print("=" * 60 + "\n")

    # File
    filename = f"ai_daily_{target_date.strftime('%Y-%m-%d')}.md"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Saved: %s", filepath)
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
#  Date Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_date(date_str: Optional[str]) -> date:
    """Flexible date parser. Supports 'today', 'yesterday', 'YYYY-MM-DD', +/-N."""
    if not date_str or date_str.lower() in ("today", ""):
        return date.today()
    if date_str.lower() == "yesterday":
        return date.today() - timedelta(days=1)
    if date_str.lower() == "tomorrow":
        return date.today() + timedelta(days=1)

    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        pass

    try:
        return datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError:
        pass

    m = re.match(r"^([+-]?)(\d+)$", date_str)
    if m:
        sign = 1 if m.group(1) != "-" else -1
        return date.today() + timedelta(days=sign * int(m.group(2)))

    logger.warning("  Unrecognized date '%s', using today", date_str)
    return date.today()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AI Daily Briefing Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_daily.py                                    # today
  python ai_daily.py --date 2026-06-01                  # specific date
  python ai_daily.py --date yesterday                    # yesterday
  python ai_daily.py -o ./briefings                     # custom output dir
  python ai_daily.py --skip-summary                     # skip DeepSeek
  python ai_daily.py --proxy http://127.0.0.1:7890       # proxy
  python ai_daily.py --verbose                          # verbose logging

  # Generate + update data.json for web page (one-command workflow):
  python ai_daily.py --update-json
  python ai_daily.py --update-json -o E:/Claude\\ code/AI\\日报
        """,
    )
    parser.add_argument("--date", "-d", default=None,
                        help="Date: YYYY-MM-DD / yesterday / today / +/-N")
    parser.add_argument("--output", "-o", default=".",
                        help="Output directory (default: current dir)")
    parser.add_argument("--skip-summary", action="store_true",
                        help="Skip DeepSeek summarization")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    parser.add_argument("--proxy", default=None,
                        help="Proxy, e.g. http://127.0.0.1:7890")
    parser.add_argument("--no-reddit", action="store_true",
                        help="Skip Reddit source")
    parser.add_argument("--update-json", action="store_true",
                        help="Update data.json for web page (combine with -o to set path)")
    parser.add_argument("--max-news", type=int, default=20,
                        help="Max news items in JSON output (default: 20)")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Proxy setup
    if args.proxy:
        os.environ.setdefault("HTTP_PROXY", args.proxy)
        os.environ.setdefault("HTTPS_PROXY", args.proxy)
        logger.info("Proxy: %s", args.proxy)
    elif _has_proxy():
        logger.info("Proxy detected from environment")
    else:
        logger.info("No proxy -- Reddit/NewsAPI skipped, arXiv/HF direct connect")

    target_date = parse_date(args.date)

    # Header
    logger.info("=" * 46)
    logger.info("  AI Daily Briefing Generator -- %s", target_date)
    logger.info("=" * 46)
    print()

    # ── Collect data (each source independent) ──

    # 1. arXiv
    papers_arxiv = fetch_arxiv(target_date)
    time.sleep(GENTLE_DELAY)

    # 2. Hugging Face
    papers_hf = fetch_huggingface(target_date)
    time.sleep(GENTLE_DELAY)

    # Merge (deduplicate by title)
    seen_titles = {p["title"].lower() for p in papers_arxiv}
    for p in papers_hf:
        if p["title"].lower() not in seen_titles:
            papers_arxiv.append(p)

    # 3. NewsAPI
    news = fetch_newsapi(target_date)
    time.sleep(GENTLE_DELAY)

    # 4. Reddit
    reddit = fetch_reddit(target_date) if not args.no_reddit else []
    time.sleep(GENTLE_DELAY)

    # 5. RSS feeds (Chinese media — always fetched)
    rss = fetch_rss_feeds(target_date)

    # ── Summary & Generate ──

    logger.info("")
    logger.info("-" * 46)
    logger.info("  Papers: %d  |  News: %d  |  Reddit: %d  |  RSS: %d",
                len(papers_arxiv), len(news), len(reddit), len(rss))
    logger.info("-" * 46)
    logger.info("")

    content = generate_briefing(
        papers_arxiv, news, reddit, rss, target_date, args.skip_summary
    )

    output_briefing(content, target_date, args.output)

    # ── Update data.json if requested ──
    if args.update_json:
        json_path = os.path.join(args.output, DATA_JSON_FILENAME)
        update_data_json(json_path, papers_arxiv, news, reddit, rss,
                         content, target_date, args.max_news)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
