import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
from urllib.parse import urljoin, urlparse
from collections import deque

def same_domain(url, base_domain):
    return urlparse(url).netloc == base_domain


def is_article_url(url):
    bad_keywords = ["video", "photo", "login", "register", "tag", "search", "rss", "mailto", "javascript"]
    if any(keyword in url.lower() for keyword in bad_keywords):
        return False
    if url.endswith(".html") or url.endswith(".htm"):
        return True
    return False

def get_domain(url):
    return urlparse(url).netloc

def collect_article_links(seed_urls, max_links=1000, max_pages=300):
    article_links = set()
    visited_pages = set()
    queue = deque()

    for seed_url in seed_urls:
        seed_domain = get_domain(seed_url)
        queue.append((seed_url, seed_domain))

    while queue and len(article_links) < max_links and len(visited_pages) < max_pages:
        current_url, base_domain = queue.popleft()
        if current_url in visited_pages:
            continue
        visited_pages.add(current_url)
        html = get_html(current_url)
        if html is None:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(current_url, href)
            full_url = full_url.split("#")[0]
            if not same_domain(full_url, base_domain):
                continue
            if is_article_url(full_url):
                article_links.add(full_url)
                if len(article_links) >= max_links:
                    break
            else:
                if full_url not in visited_pages:
                    queue.append((full_url, base_domain))
        print(f"Đã lấy được {len(article_links)} link bài báo từ link {current_url}")
        time.sleep(1)
    return list(article_links)


def get_html(url):
    headers = {"User-Agent": "Mozilla/5.0",
               "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8"}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
        return res.text

    except Exception as e:
        print("Lỗi tải:", url, e)
        return None


def get_article(url, crawl_title=False, crawl_content=False, crawl_lead_p=False):
    results = {"url": url}
    response_text = get_html(url)
    if response_text is None:
        return results

    soup = BeautifulSoup(response_text, "html.parser")

    if crawl_title:
        title_tag = (soup.find("h1", class_="title-detail") or
                     soup.find("h1"))
        if title_tag:
            title = title_tag.get_text(" ", strip=True)
        else:
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
        results["title"] = title

    if crawl_lead_p:
        lead_paragraph_tag = (soup.find("p", class_="lead") or
                              soup.find("p", class_="sapo") or
                              soup.find("p", class_="description") or
                              soup.find("h2"))
        lead_paragraph = ""
        if lead_paragraph_tag:
            lead_paragraph = lead_paragraph_tag.get_text(" ", strip=True)
        results["lead_paragraph"] = lead_paragraph

    if crawl_content:
        paragraphs = soup.find_all("p")
        content = []
        for p in paragraphs:
            text = p.get_text(" ", strip=True)
            if text:
                content.append(text)
        results["content"] = " ".join(content)
    return results

def main():
    seed_urls = ["https://thanhnien.vn/",
                 "https://vnexpress.net/doi-song",
                 "https://ngoisao.vnexpress.net/showbiz/viet-nam",
                 "https://kenh14.vn/xa-hoi.chn",
                 "https://eva.vn/",
                 "https://suckhoedoisong.vn/",
                 "https://www.saostar.vn/",
                 "https://tuoitre.vn/",
                 "https://nld.com.vn/"]
    urls = collect_article_links(seed_urls, max_links=3000, max_pages=200)
    data = []
    for url in tqdm(urls):
        article = get_article(url, True, True, True)
        if article is not None:
            data.append(article)
        time.sleep(1)
    df = pd.DataFrame(data)
    print(df.head())
    save_path = r"D:\private\clickbait_detect_proj\data\raw\articles_3000.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("Đã lưu file articles2.csv")

if __name__ == "__main__":
    main()
