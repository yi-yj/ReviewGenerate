from scholarly import scholarly
import arxiv
import os
import requests
from difflib import SequenceMatcher
import pdfplumber
import re
import json
import time

def is_pdf_corrupted(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            # 尝试读取第一页的文本
            if len(pdf.pages) > 0: 
                _ = pdf.pages[0].extract_text()
            return False  # PDF 文件正常
    except Exception as e:
        os.remove(file_path)  # 删除损坏的 PDF 文件
        print(f"PDF is corrupted: {str(e)}")
        return True

def similarity_by_min_length(str1, str2):
    # 转为小写以忽略大小写差异
    str1 = str1.lower()
    str2 = str2.lower()
    
    # 创建 SequenceMatcher 对象
    matcher = SequenceMatcher(None, str1, str2)
    
    # 获取匹配字符数（公共子序列的总长度）
    matching_blocks = matcher.get_matching_blocks()
    matched_chars = sum(block.size for block in matching_blocks)
    
    # 获取最小单词长度
    min_length = min(len(str1), len(str2))
    
    # 计算相似度
    if min_length == 0:  # 防止除以零
        return 0.0
    similarity = matched_chars / min_length
    
    return similarity

def similarity_by_ratio(str1, str2):
    # 转为小写以忽略大小写差异
    str1 = str1.lower()
    str2 = str2.lower()
    
    # 创建 SequenceMatcher 对象
    matcher = SequenceMatcher(None, str1, str2)
    
    # 获取匹配度
    similarity = matcher.ratio()
    
    return similarity


class PDFDownloaderByKeywords:
    def __init__(self, download_dir="download_files", max_results=5,):
        self.download_dir = download_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        self.max_results = max_results

    def search(self, keywords, max_retries=10):
        """
        参数:
        keywords: 要搜索的论文关键词
        返回:
        abstract: 成功获得的文献摘要（优先返回同时下载成功的）
        """
        print('Searching for paper, keywords:', keywords)
        temp_abstract = None
        method_list = [self._search_from_arxiv, self._search_from_semantic, self._search_from_google]

        for search_method in method_list:
            abstract, isdownload = search_method(keywords)
            
            # 如果搜索方法返回 'retry'，则做短暂重试
            if abstract == 'retry':
                for _ in range(max_retries):
                    time.sleep(5)
                    abstract, isdownload = self._search_from_semantic(keywords)
                    if abstract not in (None, 'retry'):
                        print("get abstract successfully!")
                        break
                if abstract in (None, 'retry'):
                    continue  # 本方法尝试后仍未获取摘要，切换下一个方法

            # 如果没有摘要，则直接尝试下一个方法
            if abstract is None:
                continue

            # 如果获得摘要且同时下载成功，则直接返回摘要
            if isdownload:
                print("get abstract and download successfully!")
                return abstract

            # 如果获得摘要但下载未成功，则保留此摘要，继续尝试下一个方法下载 PDF
            if temp_abstract is None:
                temp_abstract = abstract
            print("Got abstract but download failed, trying next method...")

        # 如果所有方法尝试后下载都不成功，但保留了摘要，则返回之前获得的摘要
        if temp_abstract:
            print("Returning abstract from previous attempt (download failed).")
            return temp_abstract
        else:
            return None


    def _search_from_google(self, keywords):
        """
        参数:
        keywords: 要搜索的论文关键词
        返回:
        list: 成功下载的PDF文件路径列表
        """
        print("Trying Google Scholar search...")
        file_path = []
        abstracts = []
        undownload = []

        try:
            search_ = scholarly.search_pubs(keywords)
            search_query = []
            for _ in range(self.max_results):
                try:
                    search_query.append(search_.__next__())
                except StopIteration:
                    break

            for pub in search_query:
                print("Searching:", pub['bib'].get('title'))
                isdownload = False
                
                safety_title = re.sub(r'[\\/:*?"<>|]', ' ', pub['bib'].get('title')).lower()
                safety_title = re.sub(r'\s+', ' ', safety_title).strip()
                if pub['bib'].get('abstract'):
                    abstracts.append({'title': safety_title, 'abstract': pub['bib'].get('abstract')})
                if 'eprint_url' in pub:
                    pdf_url = pub['eprint_url']
                    pdf_path = self.download_from_url(pdf_url, safety_title)
                    if pdf_path and not is_pdf_corrupted(pdf_path):
                        file_path.append(pdf_path)
                        isdownload = True

                if not isdownload:
                    undownload.append({'title': safety_title, 'abstract': pub['bib'].get('abstract')})
                                 
            return abstracts, file_path, undownload
        
        except Exception as e:
            return abstracts, file_path, undownload

    def _search_from_arxiv(self, keywords):
        """
        参数:
        keywords: 要搜索的论文关键词
        返回:
        list: 成功下载的PDF文件路径列表
        """
        print("Trying arXiv search...")
        file_path = []
        abstracts = []

        try:
            search = arxiv.Search(
                query=keywords,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate if self.sort_by_date else arxiv.SortCriterion.Relevance
            )
            client = arxiv.Client()
            results = []
            
            for result in client.results(search):
                results.append(result)
            
            if not results:
                return abstracts, file_path
            
            for res in results:
                safety_title = re.sub(r'[\\/:*?"<>|]', ' ', res.title).lower()
                safety_title = re.sub(r'\s+', ' ', safety_title).strip()
                if res.summary:
                    abstracts.append({'title': safety_title, 'summary': res.summary})

                pdf_path = res.download_pdf(dirpath=self.download_dir, filename=f"{safety_title}.pdf")
                if pdf_path and not is_pdf_corrupted(pdf_path):
                    file_path.append(pdf_path)

                return abstracts, file_path
        
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return abstracts, file_path
    
    def _search_from_semantic(self, keywords):
        """
        参数:
        keywords: 要搜索的论文关键词
        返回:
        list: 成功下载的PDF文件路径列表
        """
        print("Trying semantic search...")
        abstracts = []
        file_path = []
        undownloaded = []
        for i in range(self.max_results//100):
            for attempt in range(20):
                url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={keywords}&offset={i * 100}&limit=100&fields=title,abstract,openAccessPdf"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        papers = data.get('data', [])
                        for paper in papers:
                            isdownload = False
                            safety_title = re.sub(r'[\\/:*?"<>|]', ' ', paper['title']).lower()
                            safety_title = re.sub(r'\s+', ' ', safety_title).strip()
                            if paper.get('abstract'):
                                abstracts.append({'title': safety_title, 'abstract': paper.get('abstract')})
                                open_access_pdf = paper.get('openAccessPdf', {})
                                pdf_url = open_access_pdf.get('url')
                                if pdf_url:
                                    pdf_path = self.download_from_url(pdf_url, safety_title)
                                    if pdf_path and not is_pdf_corrupted(pdf_path):
                                        file_path.append(pdf_path)
                                        isdownload = True
                            if not isdownload:
                                undownloaded.append({'title': safety_title, 'abstract': paper.get('abstract')})
                        break
                    elif response.status_code in {404, 429}:
                        continue
                    else:
                        continue
                except Exception as e:
                    print(f"Search failed: {str(e)}")
                    continue
        return abstracts, file_path, undownloaded
    
    def download_from_url(self, url, safe_title):
        """
        参数:
        url: 要下载的 PDF 链接
        title: 文件名
        返回:
        bool: 是否成功下载PDF
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            file_path = os.path.join(self.download_dir, f"{safe_title}.pdf")
            
            with open(file_path, "wb") as file:
                file.write(response.content)
                
            print(f"PDF downloaded successfully: {file_path}")
            return file_path
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to download PDF: {str(e)}")
            return False
        except OSError as e:
            print(f"Failed to save file: {str(e)}")
            return False
        
if __name__ == "__main__":
    get_paper = PDFDownloaderByKeywords(download_dir="download_files2", max_results=600)
    keywords = "sparse adaptive filtering"
    abstracts, file_path, undownloaded = get_paper._search_from_semantic(keywords)
    abstract_file = os.path.join(get_paper.download_dir, "abstracts.json")
    undownload_file = os.path.join(get_paper.download_dir, "undownloaded.json")
    
    with open(abstract_file, "w", encoding="utf-8") as f:
        json.dump(abstracts, f, ensure_ascii=False, indent=4)
    print(f"Abstracts written to {abstract_file}")
    
    with open(undownload_file, "w", encoding="utf-8") as f:
        json.dump(undownloaded, f, ensure_ascii=False, indent=4)
    print(f"Undownloaded written to {undownload_file}")
