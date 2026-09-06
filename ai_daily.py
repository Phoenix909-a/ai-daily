#!/usr/bin/env python3
"""
AI Daily Briefing Generator — 每日 AI 进展自动简报生成器

聚合精选 AI 媒体（The Decoder、TechCrunch AI、The Verge AI、量子位、新智元、
InfoQ 中文等）、Reddit、Hacker News、GitHub Trending 等多个信源，调用
DeepSeek API 生成结构化中文简报并输出 Markdown 文件。

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
    HTTP_PROXY          可选   HTTP 代理地址
    HTTPS_PROXY         可选   HTTPS 代理地址 (requests 自动读取)

    # 邮件推送（可选，使用 --send-email 时需配置）
    EMAIL_HOST          可选   SMTP 服务器地址 (默认 smtp.163.com)
    EMAIL_SMTP_PORT     可选   SMTP 端口 (默认 465)
    EMAIL_USER          可选   发件邮箱地址
    EMAIL_PASSWORD      可选   SMTP 授权码（不是登录密码）
    EMAIL_TO            可选   收件邮箱地址（默认发给发件人）
"""

import os
import re
import sys
import json
import time
import html
import logging
import argparse
import smtplib
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from typing import Optional
from email.utils import parsedate_to_datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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

# 精选 AI 信源（focus: 模型进展 / Agent / 新功能 / 对比 / 使用趋势）
# 每项: (名称, Feed 地址, 关键词过滤 or None=全部保留)
_AI_FEED_KEYWORDS = (
    "ai", "llm", "gpt", "claude", "gemini", "deepseek", "agent",
    "openai", "anthropic", "mistral", "copilot", "chatbot", "grok",
    "llama", "qwen", "kimi", "大模型", "模型", "人工智能", "智能体",
    "ai 助手", "ai工具",
)

NEWS_FEEDS = [
    # ── 国外源（需代理时走 HTTP_PROXY） ──
    ("The Decoder",    "https://www.the-decoder.com/feed/",                     None),
    ("TechCrunch AI",  "https://techcrunch.com/category/artificial-intelligence/feed/", None),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/",             None),
    ("The Verge AI",   "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", None),
    ("TLDR AI",        "https://tldr.tech/rss",                                 _AI_FEED_KEYWORDS),
    ("Ben's Bites",    "https://bensbites.com/feed/",                           None),
    ("Simon Willison", "https://simonwillison.net/atom/everything/",            _AI_FEED_KEYWORDS),
    ("OpenAI News",    "https://openai.com/news/rss.xml",                       None),
    ("DeepMind Blog",  "https://deepmind.google/blog/rss.xml",                  None),
    ("Product Hunt",   "https://www.producthunt.com/feed",                      _AI_FEED_KEYWORDS),
    # ── 国内源（直连） ──
    ("量子位",          "https://www.qbitai.com/feed",                            None),
    ("新智元",          "https://www.aiera.com.cn/feed",                          None),
    ("InfoQ 中文",      "https://www.infoq.cn/feed",                              _AI_FEED_KEYWORDS),
    ("爱范儿",          "https://www.ifanr.com/feed",                             _AI_FEED_KEYWORDS),
    ("少数派",          "https://sspai.com/feed",                                 _AI_FEED_KEYWORDS),
    ("极客公园",        "https://www.geekpark.net/rss",                           _AI_FEED_KEYWORDS),
    ("雷锋网",          "https://www.leiphone.com/feed",                          None),
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
#  Source 1: Reddit (AI 社区热帖 — needs proxy in China)
# ═══════════════════════════════════════════════════════════════════════════════

REDDIT_SUBREDDITS = ["LocalLLaMA", "ClaudeAI", "AI_Agents", "OpenAI"]


def fetch_reddit(target_date: date) -> list[dict]:
    """Fetch top posts from AI-focused subreddits. Needs proxy in China."""
    if not _has_proxy():
        logger.info("[Reddit]  Skipped (no proxy, blocked in China)")
        return []

    logger.info("[Reddit]  Fetching …")

    all_posts = []
    for sub in REDDIT_SUBREDDITS:
        posts = _fetch_subreddit_day(sub, target_date)
        if not posts:
            posts = _fetch_reddit_hot(sub, target_date)
        all_posts.extend(posts)
        time.sleep(GENTLE_DELAY)

    # Deduplicate by URL, keep highest score
    seen = set()
    unique = []
    for p in sorted(all_posts, key=lambda x: x["score"], reverse=True):
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)

    logger.info("  -> %d posts", len(unique))
    return unique[:15]


def _fetch_subreddit_day(sub: str, target_date: date) -> list[dict]:
    """Search one subreddit for posts on the target date."""
    dt_start = datetime(target_date.year, target_date.month, target_date.day)
    dt_end = dt_start + timedelta(days=1)

    url = f"https://www.reddit.com/r/{sub}/search.json"
    params = {
        "q": f"timestamp:{int(dt_start.timestamp())}..{int(dt_end.timestamp())}",
        "restrict_sr": "on", "sort": "top", "syntax": "cloudsearch", "limit": 100,
    }
    headers = {"User-Agent": f"{USER_AGENT} (by /u/ai_briefing_bot)"}

    resp = _rget(url, params=params, headers=headers)
    if not resp:
        return []

    try:
        body = resp.json()
    except json.JSONDecodeError:
        return []

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
    logger.debug("  %s: %d posts", sub, len(posts))
    return posts[:15]


def _fetch_reddit_hot(sub: str, target_date: date) -> list[dict]:
    """Fallback: paginate hot listing for one subreddit."""
    logger.debug("  [Fallback] %s hot listing …", sub)

    url = f"https://www.reddit.com/r/{sub}/hot.json"
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
    logger.debug("  %s: %d posts", sub, len(posts))
    return posts[:15]


# ═══════════════════════════════════════════════════════════════════════════════
#  Source 2: RSS Feeds (精选国内外 AI 媒体)
# ═══════════════════════════════════════════════════════════════════════════════

def _matches_ai_keywords(title: str, desc: str, keywords) -> bool:
    """Check if a feed item looks AI-related (word-boundary aware)."""
    text = f"{title} {desc}".lower()
    for kw in keywords:
        if kw.isascii() and len(kw) <= 3:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True
        elif kw in text:
            return True
    return False


def fetch_rss_feeds(target_date: date) -> list[dict]:
    """Fetch from curated international + Chinese AI media feeds."""
    logger.info("[RSS]  Fetching curated AI feeds (%d sources) …", len(NEWS_FEEDS))

    items = []
    for name, feed_url, keywords in NEWS_FEEDS:
        logger.debug("  Fetching %s …", name)
        resp = _rget(feed_url)
        if not resp:
            logger.warning("  %s unavailable", name)
            continue

        try:
            raw_xml = resp.content.decode("utf-8", errors="ignore")
            raw_xml = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw_xml)
            root = ET.fromstring(raw_xml)
        except ET.ParseError as e:
            logger.warning("  %s XML error: %s", name, e)
            continue

        # RSS 2.0
        for entry in root.findall(".//item"):
            title = entry.findtext("title", "Untitled")
            link = entry.findtext("link", "")
            desc = entry.findtext("description", "")
            pub_str = entry.findtext("pubDate", "")

            if keywords and not _matches_ai_keywords(title, desc or "", keywords):
                continue

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
            desc = re.sub(r"<[^>]+>", "", el.text or "")[:500] if el is not None else ""

            if keywords and not _matches_ai_keywords(title, desc, keywords):
                continue

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
#  Source 6: Hacker News (via Firebase API — free, no key needed)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_hackernews(target_date: date) -> list[dict]:
    """Fetch top stories from Hacker News for the target date."""
    logger.info("[HN]  Fetching …")

    # Get top story IDs
    resp = _rget("https://hacker-news.firebaseio.com/v0/topstories.json")
    if not resp:
        return []

    try:
        ids = resp.json()[:60]  # top 60
    except json.JSONDecodeError:
        return []

    items = []
    for story_id in ids:
        resp = _rget(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
        if not resp:
            continue
        try:
            story = resp.json()
        except json.JSONDecodeError:
            continue

        ts = story.get("time", 0)
        story_date = datetime.utcfromtimestamp(ts).date()
        if story_date != target_date:
            continue

        title = story.get("title", "")
        if not _matches_ai_keywords(title, "", _AI_FEED_KEYWORDS):
            continue
        url = story.get("url", f"https://news.ycombinator.com/item?id={story_id}")
        score = story.get("score", 0)
        descendants = story.get("descendants", 0)
        by = story.get("by", "")

        items.append({
            "title": title,
            "link": url,
            "description": f"[HN] +{score} points · {descendants} comments · by {by}",
            "score": score,
            "source": "hackernews",
        })
        time.sleep(0.1)  # be gentle

    items.sort(key=lambda x: x["score"], reverse=True)
    logger.info("  -> %d stories", len(items))
    return items[:15]


# ═══════════════════════════════════════════════════════════════════════════════
#  Source 7: GitHub Trending (via GitHub Search API — no key needed for public)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_github_trending(target_date: date) -> list[dict]:
    """Fetch trending AI/ML/Agent repos recently active on GitHub."""
    logger.info("[GitHub]  Fetching trending repos …")

    # Search for recently-active repos with AI/ML/Agent topics, sorted by stars
    date_s = target_date.strftime("%Y-%m-%d")
    query = (
        f"(topic:artificial-intelligence OR topic:machine-learning "
        f"OR topic:deep-learning OR topic:llm OR topic:large-language-model "
        f"OR topic:generative-ai OR topic:ai-agents OR topic:agents)"
        f"+pushed:>={date_s}"
    )

    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": 15,
    }

    resp = _rget("https://api.github.com/search/repositories", params=params)
    if not resp:
        # Try trending page fallback
        logger.debug("  Search API failed, trying trending RSS …")
        return _fetch_github_trending_rss(target_date)

    try:
        body = resp.json()
    except json.JSONDecodeError:
        return []

    items = []
    for r in body.get("items", []):
        items.append({
            "title": r["full_name"],
            "link": r["html_url"],
            "description": (r.get("description") or "")[:300],
            "stars": r.get("stargazers_count", 0),
            "forks": r.get("forks_count", 0),
            "lang": r.get("language") or "unknown",
            "source": "github",
        })

    items.sort(key=lambda x: x["stars"], reverse=True)
    logger.info("  -> %d repos", len(items))
    return items[:10]


def _fetch_github_trending_rss(target_date: date) -> list[dict]:
    """Fallback: GitHub trending page scraped via RSS bridge."""
    logger.debug("  [Fallback] GitHub trending …")

    url = "https://github.com/trending"
    resp = _rget(url, headers={"User-Agent": USER_AGENT,
                                "Accept": "text/html"})
    if not resp:
        return []

    # Parse HTML for trending repos (basic regex approach)
    html = resp.text
    items = []

    # Match repo blocks: <h2><a href="/owner/repo">...</a></h2>
    repo_pattern = re.compile(r'href="/([^/"]+/([^/"]+))"[^>]*>.*?</h2>')
    desc_pattern = re.compile(r'<p class="col-9[^"]*"[^>]*>(.*?)</p>')
    star_pattern = re.compile(r'(?:(\d[\d,]*)\s+stars)', re.IGNORECASE)

    repo_matches = repo_pattern.findall(html)
    desc_matches = desc_pattern.findall(html)

    for i, (full_name, _) in enumerate(repo_matches[:15]):
        desc = ""
        if i < len(desc_matches):
            desc = re.sub(r"<[^>]+>", "", desc_matches[i]).strip()

        items.append({
            "title": full_name,
            "link": f"https://github.com/{full_name}",
            "description": desc[:300],
            "stars": 0,
            "source": "github",
        })

    logger.debug("  -> %d repos", len(items))
    return items


# ═══════════════════════════════════════════════════════════════════════════════
#  Email Sender
# ═══════════════════════════════════════════════════════════════════════════════

def _render_inline_html(text: str) -> str:
    """Render one markdown-ish line as safe inline HTML (links + bold)."""
    text = html.escape(text, quote=False)
    text = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        lambda m: f'<a href="{m.group(2)}" style="color:#2563eb;text-decoration:none;">{m.group(1)}</a>',
        text,
    )
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    return text


def _render_inline_plain(text: str) -> str:
    """Strip markdown markers from one line for the plain-text version."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1：\2', text)
    return text


def _score_badge_html(score: int) -> str:
    """A small inline score chip, colored by importance band."""
    if score >= 90:
        bg, fg = "#dbeafe", "#1d4ed8"
    elif score >= 75:
        bg, fg = "#dcfce7", "#15803d"
    elif score >= 60:
        bg, fg = "#fef3c7", "#b45309"
    else:
        bg, fg = "#f3f4f6", "#6b7280"
    return (
        f'<span style="display:inline-block;background:{bg};color:{fg};'
        f'font-size:12px;font-weight:600;line-height:1;padding:3px 8px;'
        f'border-radius:10px;margin:0 8px 2px 0;vertical-align:1px;">'
        f'重要度 {score}</span>'
    )


def _normalize_url(url: str) -> str:
    """Strip query string, fragment and trailing slash for score matching."""
    return url.split("?")[0].split("#")[0].rstrip("/")


def _best_score_in_block(block: list, score_by_url: dict) -> Optional[int]:
    """Find the highest matching enrichment score for URLs inside a block."""
    if not score_by_url:
        return None
    text = "\n".join(block)
    urls = re.findall(r'https?://[^\s\)\]》>]+', text)
    scores = [
        score_by_url[_normalize_url(u)]
        for u in urls
        if _normalize_url(u) in score_by_url
    ]
    return max(scores) if scores else None


def _md_to_email_html(md: str, score_by_url: Optional[dict] = None) -> str:
    """Convert the DeepSeek briefing markdown to block-per-item HTML.

    Each non-heading run of lines becomes its own <p> block (lines joined
    with <br>), so every news item starts on its own line instead of being
    collapsed into one wall of text by the mail client.
    """
    out: list = []
    block: list = []
    score_by_url = score_by_url or {}

    def flush():
        if not block:
            return
        badge = ""
        if block[0].startswith("**"):
            score = _best_score_in_block(block, score_by_url)
            if score is not None:
                badge = _score_badge_html(score)
        rendered = [_render_inline_html(line) for line in block]
        body = "<br>".join(rendered)
        out.append(
            f'<p style="margin:10px 0;line-height:1.75;color:#333;">{badge}{body}</p>'
        )
        block.clear()

    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("### "):
            flush()
            out.append(
                f'<h3 style="margin:18px 0 6px;font-size:16px;color:#111;">'
                f'{_render_inline_html(line[4:])}</h3>'
            )
            continue
        if line.startswith("## "):
            flush()
            out.append(
                f'<h2 style="margin:22px 0 6px;padding-bottom:6px;'
                f'border-bottom:1px solid #eee;font-size:18px;color:#111;">'
                f'{_render_inline_html(line[3:])}</h2>'
            )
            continue
        if line.startswith("# "):
            # Skip the markdown document title; the wrapper already has a header.
            flush()
            continue
        if line.startswith(("- ", "* ")):
            if block:
                flush()
            block.append("• " + line[2:])
            continue
        if block and line.startswith("**"):
            flush()
        block.append(line)

    flush()
    return "\n".join(out)


def _md_to_plain_text(md: str, score_by_url: Optional[dict] = None) -> str:
    """Convert briefing markdown into a clean line-per-item plain-text body."""
    out: list = []
    block: list = []
    score_by_url = score_by_url or {}

    def flush():
        if not block:
            return
        rendered = [_render_inline_plain(line) for line in block]
        if block[0].startswith("**"):
            score = _best_score_in_block(block, score_by_url)
            if score is not None:
                rendered[0] = f"[重要度 {score}] {rendered[0]}"
        out.append("\n".join(rendered))
        out.append("")
        block.clear()

    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("### "):
            flush()
            out.append(line[4:])
            out.append("")
            continue
        if line.startswith("## "):
            flush()
            out.append(line[3:])
            out.append("")
            continue
        if line.startswith("# "):
            flush()
            continue
        if line.startswith(("- ", "* ")):
            if block:
                flush()
            block.append("• " + line[2:])
            continue
        if block and line.startswith("**"):
            flush()
        block.append(line)

    flush()
    return "\n".join(out).strip()


def _build_email_html(content: str, target_date: date,
                      score_by_url: Optional[dict] = None) -> str:
    """Build a nice HTML email from the markdown briefing content."""
    body_html = _md_to_email_html(content, score_by_url)

    # Wrap in proper document
    date_str = target_date.strftime("%Y-%m-%d")
    full_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:640px;margin:0 auto;padding:20px;background:#f8f9fa;">
<div style="background:#fff;border-radius:12px;padding:24px;box-shadow:0 1px 3px rgba(0,0,0,0.08);">
<div style="text-align:center;margin-bottom:20px;">
<h1 style="font-size:22px;margin:0;color:#111;">📰 AI 新闻日报</h1>
<p style="color:#888;font-size:13px;">{date_str}</p>
</div>
{body_html}
<hr style="border:none;border-top:1px solid #eee;margin:24px 0;">
<p style="text-align:center;font-size:12px;color:#aaa;">
本简报由 AI 自动生成 · <a href="https://phoenix909-a.github.io/ai-daily/" style="color:#2563eb;">查看网页版</a>
</p>
</div>
</body>
</html>"""

    return full_html


def send_email(content: str, target_date: date,
               news_items: Optional[list] = None) -> bool:
    """Send the briefing via email using SMTP.

    Requires env vars: EMAIL_USER, EMAIL_PASSWORD (SMTP authorization code).
    Optional: EMAIL_HOST (default smtp.163.com), EMAIL_SMTP_PORT (default 465),
              EMAIL_TO (default = EMAIL_USER).
    Optional: news_items = data.json news list; used to show the DeepSeek
              importance score (sort_score) next to each item in the email.
    """
    smtp_server = os.environ.get("EMAIL_HOST", "smtp.163.com")
    smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", "465"))
    email_from = os.environ.get("EMAIL_USER", "")
    email_pass = os.environ.get("EMAIL_PASSWORD", "")
    email_to = os.environ.get("EMAIL_TO", email_from)

    if not email_from or not email_pass:
        logger.warning("[Email]  EMAIL_USER or EMAIL_PASSWORD not set, skipped")
        return False

    logger.info("[Email]  Sending to %s via %s:%d …", email_to, smtp_server, smtp_port)

    date_s = target_date.strftime("%Y-%m-%d")
    subject = f"AI 新闻日报 — {date_s}"

    # Link each curated briefing item to its enrichment score (by source URL)
    score_by_url: dict = {}
    for item in (news_items or []):
        link = item.get("link", "")
        score = item.get("sort_score")
        if link and isinstance(score, (int, float)):
            score_by_url[_normalize_url(link)] = int(score)

    # Build HTML email
    html_content = _build_email_html(content, target_date, score_by_url)
    plain_content = f"AI 新闻日报 — {date_s}\n\n" + _md_to_plain_text(
        content, score_by_url
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = email_from
    msg["To"] = email_to

    # Plain text fallback
    msg.attach(MIMEText(plain_content, "plain", "utf-8"))
    # HTML version
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) as server:
            server.login(email_from, email_pass)
            server.sendmail(email_from, email_to, msg.as_string())
        logger.info("  -> Email sent successfully!")
        return True
    except smtplib.SMTPException as e:
        logger.error("  SMTP error: %s", e)
        return False
    except Exception as e:
        logger.error("  Email error: %s", e)
        return False


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

SYSTEM_PROMPT = """你是一个专业的 AI 领域新闻简报编辑。你的任务是根据提供的原始素材，生成一份聚焦「AI 模型与 Agent 进展」的每日中文简报。

内容定位（按重要性排序）：
1. 新模型 / 新版本发布与能力更新（OpenAI、Anthropic、Google、Meta、DeepSeek、Mistral 等）
2. Agent 进展（智能体框架、多智能体、Agent 能力与评测）
3. 新功能上线与产品更新（各 AI 产品、工具的新功能）
4. 模型 / Agent 横向对比与评测
5. 大家都在用什么（热门工具、GitHub 项目、社区讨论趋势）

请严格按照以下格式输出（Markdown 格式）：

# AI 每日简报 — YYYY-MM-DD

## 今日摘要
3-5 句话概括今日最重要的 AI 进展。

## 重磅动态
每项包含：标题、一句话中文解读（30字以内）、来源链接。聚焦模型发布 / Agent 进展 / 新功能。

## 对比与评测
（如果素材中有模型或 Agent 对比、测评类内容）

## 工具与趋势
热门工具、GitHub 项目、社区在讨论什么。

## 社区热议
Reddit / Hacker News 高热度讨论，每项包含：帖子标题、关键观点摘录、链接。

注意事项：
- 所有解读和摘要使用中文
- 突出重点，宁缺毋滥，不罗列无关新闻
- 保持客观，不添加素材中没有的信息
- 每条不超过 100 字"""


def _format_prompt(media: list, reddit: list, hackernews: list, github: list) -> str:
    """Format raw data into a prompt for the LLM."""
    parts = []

    # 精选媒体（模型/Agent/新功能/对比/使用趋势）
    if media:
        parts.append("## 今日媒体动态（模型 / Agent / 新功能 / 对比 / 使用趋势）\n")
        for i, item in enumerate(media[:25], 1):
            parts.append(f"{i}. [{item['title']}]({item['link']})  [{item.get('source', '')}]")
            if item.get("description"):
                parts.append(f"   {item['description'][:300]}")
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

    # Hacker News
    if hackernews:
        parts.append("\n## Hacker News 讨论\n")
        for i, item in enumerate(hackernews[:10], 1):
            parts.append(f"{i}. [{item['title']}]({item['link']})  [{item.get('description', '')}]")
            parts.append("")

    # GitHub 热门项目
    if github:
        parts.append("\n## GitHub 热门 AI 项目\n")
        for i, item in enumerate(github[:10], 1):
            stars = item.get("stars", "")
            parts.append(f"{i}. [{item['title']}]({item['link']})  ⭐{stars}")
            if item.get("description"):
                parts.append(f"   {item['description'][:200]}")
            parts.append("")

    return "\n".join(parts)


def _raw_briefing(media: list, reddit: list, hackernews: list, github: list,
                  target_date: date) -> str:
    """Generate a simple markdown briefing without AI (fallback)."""
    date_s = target_date.strftime("%Y-%m-%d")
    lines = [
        f"# AI Daily Briefing -- {date_s}",
        "",
        "> Note: Generated from raw data (DeepSeek summary unavailable)",
        "",
    ]

    if media:
        lines += ["## 今日动态", ""]
        for item in media[:25]:
            lines.append(f"- [{item['title']}]({item['link']})  [{item.get('source', '')}]")
            if item.get("description"):
                lines.append(f"  {item['description'][:200]}")
            lines.append("")

    if reddit:
        lines += ["## Reddit", ""]
        for post in reddit[:10]:
            lines.append(f"- [{post['title']}]({post['url']})  (+{post['score']})")
            lines.append("")

    if hackernews:
        lines += ["## Hacker News", ""]
        for item in hackernews[:10]:
            lines.append(f"- [{item['title']}]({item['link']})")
            lines.append("")

    if github:
        lines += ["## GitHub 热门 AI 项目", ""]
        for item in github[:10]:
            stars = item.get("stars", "")
            lines.append(f"- [{item['title']}]({item['link']})  ⭐{stars}")
            lines.append("")

    return "\n".join(lines)


def generate_briefing(media: list, reddit: list, hackernews: list, github: list,
                      target_date: date, skip_summary: bool = False) -> str:
    """Generate the briefing via DeepSeek or raw fallback."""
    if skip_summary:
        logger.info("[Main]  Skip-summary mode")
        return _raw_briefing(media, reddit, hackernews, github, target_date)

    user_prompt = _format_prompt(media, reddit, hackernews, github)

    if len(user_prompt) > 120_000:
        logger.warning("  Truncating prompt from %d to 120k chars", len(user_prompt))
        user_prompt = user_prompt[:120_000]

    result = call_deepseek(SYSTEM_PROMPT, user_prompt)
    if result:
        return result

    logger.warning("[Main]  DeepSeek failed, using raw briefing")
    return _raw_briefing(media, reddit, hackernews, github, target_date)


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

    def kw_in(keywords) -> bool:
        for kw in keywords:
            if kw.isascii():
                if re.search(rf"\b{re.escape(kw)}\b", all_text):
                    return True
            elif kw in all_text:
                return True
        return False

    # Keyword-based classification
    if kw_in(["agent", "智能体", "autonomous", "multi-agent", "mcp",
              "computer use", "computer-use", "copilot"]):
        return "Agent 进展"
    if kw_in(["对比", "vs", "benchmark", "评测", "测评", "排行榜", "跑分",
              "comparison", "tested"]):
        return "对比评测"
    if kw_in(["发布", "上新", "推出", "开源", "release", "launch", "新模型", "升级",
              "gpt-", "claude", "gemini", "deepseek", "llama", "qwen", "mistral",
              "grok", "opus", "sonnet", "haiku", "新版本", "模型"]):
        return "模型发布"
    if kw_in(["新功能", "功能", "update", "更新", "上线", "feature", "auto mode",
              "canvas", "artifacts", "插件", "集成"]):
        return "新功能"
    if kw_in(["github", "trending", "开源项目", "工具", "教程", "使用", "prompt",
              "api", "插件"]):
        return "工具与生态"
    if kw_in(["融资", "估值", "ipo", "收购", "财报", "营收", "裁员", "并购"]):
        return "公司动态"
    return "行业动态"


_OFF_TOPIC_HINTS = (
    "scam", "fraud", "lawsuit", "court", "sued", "energy", "electricity",
    "carbon", "hiring", "layoff", "jobs", "salary", "employment",
    "移民", "诉讼", "裁员",
)


def _fallback_sort_score(item: dict, tag: str) -> int:
    """Rule-based importance score used when DeepSeek enrichment is off."""
    base = {
        "模型发布": 90, "Agent 进展": 88, "对比评测": 86, "新功能": 82,
        "工具与生态": 76, "公司动态": 68, "行业动态": 58,
    }.get(tag, 60)
    text = f"{item.get('title', '')} {item.get('description', '')}".lower()
    if any(k in text for k in _OFF_TOPIC_HINTS):
        base = min(base, 35)
    return base


def _enrich_news_deepseek(raw_news: list, target_date: date) -> list:
    """Use DeepSeek to enrich news items with tags, details, and scores."""
    if not raw_news:
        return []

    # Build a compact prompt
    news_text = ""
    for i, item in enumerate(raw_news, 1):
        desc = item.get("description", "")[:100]
        news_text += f"{i}. {item['title']}\n   {desc}\n"

    system_prompt = """你是一个 AI 新闻编辑。你的任务是将原始新闻列表加工成结构化数据。
对每条新闻，你需要：
1. 清洗标题：去掉"刚刚"、"独家｜"等前缀，改写成"主体 + 事件"的新闻标题风格
2. 添加分类标签：从 [模型发布, Agent 进展, 新功能, 对比评测, 工具与生态, 公司动态, 行业动态] 中选择
3. 写一句话简介（15-30字）
4. 写详细摘要（2-4句话，50-100字）
5. 给重要度打分（1-100）。打分标准：
   - 新模型/新版本发布：90-100
   - Agent 能力或框架重大进展：85-95
   - 重要新功能上线：80-90
   - 模型/Agent 对比评测：75-90
   - 热门工具与使用趋势：70-85
   - 普通公司动态：60-75
   - 泛泛而谈、与 AI 模型/Agent 无关的新闻：0-40

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
                           temperature=0.2, max_tokens=6000)
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


def _build_json_entry(target_date: date, rss: list, md_content: str,
                      max_news: int = 20,
                      hackernews: Optional[list] = None,
                      github: Optional[list] = None) -> dict:
    """Build a single data.json entry from all collected data sources."""
    weekday_str = WEEKDAY_CN[target_date.weekday()]
    hackernews = hackernews or []
    github = github or []

    # ── Summary ──
    summary = _extract_summary_from_md(md_content)
    if not summary:
        total = len(rss) + len(hackernews) + len(github)
        summary = f"今日共收录 {total} 条 AI 动态。"
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

    # Hacker News
    for item in hackernews:
        url = item.get("link", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            raw_items.append({
                "title": item["title"],
                "description": item.get("description", ""),
                "link": url,
                "source": "hackernews",
            })

    # GitHub Trending
    for item in github:
        url = item.get("link", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            raw_items.append({
                "title": item["title"],
                "description": item.get("description", ""),
                "link": url,
                "source": "github",
            })

    raw_items = raw_items[:max_news]

    # ── Enrich with DeepSeek ──
    enriched = _enrich_news_deepseek(raw_items, target_date)

    # Merge DeepSeek results per-item; fall back to rules for anything missing
    news_items = []
    for i, raw in enumerate(raw_items):
        enr = enriched[i] if i < len(enriched) and isinstance(enriched[i], dict) else None
        if enr:
            news_items.append({
                "title": enr.get("title", _clean_title(raw.get("title", ""))),
                "link": raw.get("link", raw.get("url", "")),
                "description": enr.get("description", raw.get("description", ""))[:200],
                "detail": enr.get("detail", raw.get("description", ""))[:500],
                "tag": enr.get("tag", _categorize_news(raw)),
                "sort_score": enr.get("sort_score", 70),
                "source": raw.get("source", ""),
            })
        else:
            title = _clean_title(raw.get("title", ""))
            desc = raw.get("description", "")[:200]
            tag = _categorize_news(raw)
            news_items.append({
                "title": title,
                "link": raw.get("link", raw.get("url", "")),
                "description": desc,
                "detail": desc,
                "tag": tag,
                "sort_score": _fallback_sort_score(raw, tag),
                "source": raw.get("source", ""),
            })

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "weekday": weekday_str,
        "summary": summary,
        "news": news_items,
    }


def update_data_json(filepath: str, rss: list, md_content: str,
                     target_date: date, max_news: int = 20,
                     hackernews: Optional[list] = None,
                     github: Optional[list] = None) -> bool:
    """Update data.json: add/replace today's entry, keep existing ones."""
    logger.info("[JSON]  Updating %s …", filepath)

    old_data = _load_data_json(filepath)
    entry = _build_json_entry(target_date, rss, md_content,
                              max_news, hackernews, github)

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

  # Generate + update data.json + send to email (one-command workflow):
  python ai_daily.py --update-json --send-email
  python ai_daily.py --update-json --send-email -o E:/Claude\\ code/AI\\日报
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
    parser.add_argument("--max-news", type=int, default=35,
                        help="Max news items in JSON output (default: 35)")
    parser.add_argument("--send-email", action="store_true",
                        help="Send briefing via email (needs EMAIL_* env vars)")
    parser.add_argument("--no-hackernews", action="store_true",
                        help="Skip Hacker News source")
    parser.add_argument("--no-github", action="store_true",
                        help="Skip GitHub Trending source")
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
        logger.info("No proxy -- Reddit skipped, 国际源直连（部分源可能超时）")

    target_date = parse_date(args.date)

    # Header
    logger.info("=" * 46)
    logger.info("  AI Daily Briefing Generator -- %s", target_date)
    logger.info("=" * 46)
    print()

    # ── Collect data (each source independent) ──

    # 1. Reddit（需代理）
    reddit = fetch_reddit(target_date) if not args.no_reddit else []
    time.sleep(GENTLE_DELAY)

    # 2. RSS feeds（精选国内外 AI 媒体）
    rss = fetch_rss_feeds(target_date)
    time.sleep(GENTLE_DELAY)

    # 3. Hacker News
    hackernews = fetch_hackernews(target_date) if not args.no_hackernews else []
    time.sleep(GENTLE_DELAY)

    # 4. GitHub Trending
    github = fetch_github_trending(target_date) if not args.no_github else []
    time.sleep(GENTLE_DELAY)

    # ── Summary & Generate ──

    logger.info("")
    logger.info("-" * 58)
    logger.info("  Media(RSS): %d  |  Reddit: %d  |  HN: %d  |  GitHub: %d",
                len(rss), len(reddit),
                len(hackernews), len(github))
    logger.info("-" * 58)
    logger.info("")

    content = generate_briefing(
        rss, reddit, hackernews, github, target_date, args.skip_summary
    )

    output_briefing(content, target_date, args.output)

    # ── Update data.json if requested ──
    json_path = os.path.join(args.output, DATA_JSON_FILENAME)
    if args.update_json:
        update_data_json(json_path, rss, content, target_date, args.max_news,
                         hackernews, github)

    # ── Send email if requested ──
    if args.send_email:
        # Reuse today's enriched data.json entry so the email can show the
        # DeepSeek importance score next to each news item.
        news_items = None
        if args.update_json:
            date_s = target_date.strftime("%Y-%m-%d")
            for d in _load_data_json(json_path):
                if d.get("date") == date_s:
                    news_items = d.get("news") or []
                    break
        send_email(content, target_date, news_items)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
