from base_agent import AIAgent,ReviewAgent,AsyncAIAgent
import arxiv
import os
import glob
import json
import ast
import re
import time
import asyncio

from rich.progress import Progress
from rich import print

from prompt import (
    OUTLINE_SYSTEM_PROMPT,
    OUTLINE_PROMPT,
    FILTER_FOR_OUTLINE_SYSTEM_PROMPT,
    FILTER_FOR_OUTLINE_PROMPT,
    FIRST_OUTLINE_SYSTEM_PROMPT,
    FIRST_OUTLINE_PROMPT,
    SECOND_OUTLINE_SYSTEM_PROMPT,
    SECOND_OUTLINE_PROMPT,
    DRAFT_SYSTEM_PROMPT,
    DRAFT_PROMPT,
    FILTER_SYSTEM_PROMPT,
    FILTER_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    REVIEW_PROMPT,
    REVISE_SYSTEM_PROMPT,
    REVISE_PROMPT,
    generate_prompt
)

from ai_researcher import DeepReviewer
import sys

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

log_file = open("output.log", "w", encoding="utf-8")

sys.stdout = Tee(log_file, sys.__stdout__)
sys.stderr = Tee(log_file, sys.__stderr__)

def search_from_arxiv(keywords,max_results,download_dir = "download"):
    """
    参数:
    keywords: 要搜索的论文关键词
    返回:
    lists: 下载的论文pdf地址列表
    abstracts: 论文摘要字典
    """
    lists = []
    abstracts = {}
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    try:
        search = arxiv.Search(
            query=keywords,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        client = arxiv.Client()
        results = []
        for result in client.results(search):
            found_title = result.title.lower()
            results.append(result)
        if not results:
            print("No results found")
            return lists,abstracts
        
        results = results[:max_results]
            
        for result in results:
            res = result
            print(f"\nChecking paper: {res.title}")
            try:
                safe_title = "".join(c for c in res.title if c.isalnum() or c in " _-").rstrip()[:200]
                file_path = os.path.join(download_dir, f"{safe_title}.pdf")
                res.download_pdf(dirpath=download_dir, filename=f"{safe_title}.pdf")
                print(f"PDF downloaded successfully: {file_path}")
                lists.append(file_path)

                abstracts[safe_title]=res.summary
                print(f"Successfully get abstract")
            except Exception as e:
                print(f"Failed to get paper: {str(e)}")
                continue
        
        return lists,abstracts

    except Exception as e:
            print(f"Search failed: {str(e)}")
            return lists,abstracts

def safe_parse(output: str) -> dict:
    """安全解析大模型输出的字典字符串"""
    try:
        # 删除可能干扰的换行和空格
        cleaned = output.replace("\n", "").strip()
        # 识别合法字典结构
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return ast.literal_eval(cleaned)
        else:
            # 尝试提取字典部分
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            return ast.literal_eval(cleaned[start:end])
    except (SyntaxError, ValueError) as e:
        print(f"解析失败: {e}")
        return {}

def outline2dict(outline_str):
    """
    将大模型输出的markdown格式的文献综述大纲转换为字典
    outline: 文献综述大纲
    """
    outline_dict={}
    lines = outline_str.split('\n')
    standard = True
    for line in lines:
        if line.strip().startswith('### '):
            standard = False
            break
    if standard:
        for line in lines:
            line = line.strip()
            if line.startswith('# ') and not line.startswith('## '):
                # 一级标题
                current_key = line[2:]
                outline_dict[current_key] = {}
            elif line.startswith('## '):
                # 二级标题
                if current_key is not None:
                    sub_title = line[3:]
                    outline_dict[current_key][sub_title]=""
    else:
        for line in lines:
            line = line.strip()
            if line.startswith('## ') and not line.startswith('### '):
                # 一级标题
                current_key = line[3:]
                outline_dict[current_key] = {}
            elif line.startswith('### '):
                # 二级标题
                if current_key is not None:
                    sub_title = line[4:]
                    outline_dict[current_key][sub_title]=""

    for key, val in outline_dict.items():
        if isinstance(val, dict):
            if len(val) == 0:
                outline_dict[key] = ""
    return outline_dict

def outline_normalize(outline, mode = 'outline'):
    """
    对生成的 outline 字典进行规范化：
    1. 统一为每个一级大纲添加序列号格式，形如 "数字. 标题"。
    2. 对于存在子大纲的一级大纲，其子大纲统一添加序列号格式，形如 "一级序号.子序号. 标题"。
    3. 如果已有的标题前存在乱序或缺失的序号，会自动去除，再统一添加。
    4. 当 mode 为 "full text" 时，删除值为空（空字符串或空字典）的节。

    参数:
        outline: 已生成的 outline 字典，键为大纲标题，可能带有错误或混乱的序号。
        mode: "outline" 或 "full text"，分别表示保留所有节或删除值为空的节。

    返回:
        规范化后的 outline 字典
    """
    assert mode in ['outline', 'full text']

    new_outline = {}
    top_index = 1
    seen_subtitles = set()  # 用于存储已见的子标题，避免重复
    for top_key, top_value in outline.items():
        # 去除一级标题前的已存在序号（如 "2.1. " 或 "3. "）
        new_top_title = re.sub(r'^\d+(\.\d+)*\.\s*', '', top_key).strip()
        new_top_key = f"{top_index}. {new_top_title}"
        if isinstance(top_value, dict):
            top_value = dict(list(top_value.items())[:6])
        if isinstance(top_value, dict):
            new_subsections = {}
            sub_index = 1
            for sub_key, sub_value in top_value.items():
                clean_title = re.sub(r'^\d+(\.\d+)*\.?\s*', '', sub_key).strip()
                if clean_title in seen_subtitles:
                    continue
                seen_subtitles.add(clean_title)
                # 去除二级标题前的已存在序号，包括以单独数字开头的情况
                new_sub_title = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', sub_key).strip()
                new_sub_title = re.sub(r'^\d+(\.\d+)*\.\s*', '', new_sub_title).strip()
                new_sub_key = f"{top_index}.{sub_index}. {new_sub_title}"
                # 如果在 full text 模式下，子节内容为空，则跳过
                if mode == "full text" and (not sub_value or str(sub_value).strip() == ""):
                    sub_index += 1
                    continue
                new_subsections[new_sub_key] = sub_value
                sub_index += 1
            # 当一级节为字典时，如果 full text 模式下且子节为空，则不保留此一级节
            if mode == "full text" and not new_subsections:
                top_index += 1
                continue
            new_outline[new_top_key] = new_subsections
        else:
            # 对于非字典内容（字符串），若内容为空且在 full text 模式下，则跳过
            if mode == "full text" and (not top_value or str(top_value).strip() == ""):
                top_index += 1
                continue
            new_outline[new_top_key] = top_value
        top_index += 1

    return new_outline

def generate_document(outline_with_content: dict) -> str:
    """
    根据提供的文献综述大纲生成文档。
    """
    document = ""
    for section, value in outline_with_content.items():
        if value == "":  # 如果一级内容为空，直接退出
            continue
        section = re.sub(r'[#*_`>-]', '', section)
        document += f"\n# {section}\n"
        
        if isinstance(value, dict):
            for subsection, content in value.items():
                if content == "":  # 如果二级内容为空，直接退出
                    continue
                subsection = re.sub(r'[#*_`>-]', '', subsection)
                document += f"\n## {subsection}\n"
                document += content
        else:
            # 如果一级词典下有内容（非字典类型），直接添加
            document += value
    
    document += '\n'
    return document

class ReviewFlow():
    def __init__(self,api_key, model, reason_model,base_url, pdf_dir,max_retries=3):
        self.keywords_agent = AIAgent(api_key=api_key,model=model,base_url=base_url,temperature=0.5,max_retries=max_retries)
        self.check_agent = AIAgent(api_key=api_key,model=model,base_url=base_url,temperature=0.8,max_retries=max_retries)
        self.filter_agent = AIAgent(api_key=api_key,model=model,base_url=base_url,temperature=0.1,max_retries=max_retries)
        self.outline_agent = AIAgent(api_key=api_key,model=model,base_url=base_url,temperature=0.2,max_retries=max_retries)
        self.outline_agent_async = AsyncAIAgent(api_key=api_key,model=model,base_url=base_url,temperature=0.2,max_retries=max_retries)
        self.draft_agent = AIAgent(api_key=api_key,model=model,base_url=base_url,temperature=0.2,max_retries=max_retries)
        self.review_agent = ReviewAgent(api_key=api_key,model=model,reason_model=reason_model,base_url=base_url,temperature=0.4,max_retries=max_retries,pdf_dir=pdf_dir)
        self.score_agent= AsyncAIAgent(api_key=api_key,model=model,base_url=base_url,temperature=0.,max_retries=max_retries)
        self.revise_agent = AsyncAIAgent(api_key=api_key,model=model,base_url=base_url,temperature=0.4,max_retries=max_retries)
        self.compare_agent = AIAgent(api_key=api_key,model=model,base_url=base_url,temperature=0.,max_retries=max_retries)
        self.pdf_dir = pdf_dir
        self.score = {}
    def forward(self,topic,search_times,score_times,save_dir = 'review',example_review_outline = None):
        keywords_system_prompt = f"""
        You are a {topic} researcher specializing in literature retrieval using arXiv's API. Your task is to convert research topics into optimized search strings following these rules:
        Generate arXiv API search queries in plain text format. Given a research topic:

        1.Identify core concepts and map them to arXiv fields (ti, abs, cat, etc.)
        2.Expand terms with synonyms/acronyms
        3.Apply Boolean logic (AND, OR, ANDNOT)
        4.Output ONLY the final query string with no explanations, examples, or formatting.
        Avoid all markdown, code blocks, illegal character,or non-query text.
        """
        keywords_prompt = f"""
        I hope to study the field of {topic} and need to search for literature on arxiv. Your task is providing keyword
        example output:
        (ti:"quantum spin liquid" OR ti:QSL) AND (abs:anyon OR abs:spinon OR abs:"Majorana fermion") AND cat:cond-mat.str-el ANDNOT abs:"strong correlation"
        """

        check_system_prompt = f"""
        You are a meticulous research assistant specializing in literature analysis and gap identification in the field of {topic}. 
        Your task is to critically analyze provided abstracts to uncover potential research gaps, overlooked themes, 
        and methodological limitations. Prioritize identifying areas where additional literature is needed, including emerging trends, 
        underrepresented perspectives, or conflicting findings. Structure your analysis to guide targeted follow-up searches 
        while adhering to academic rigor.
        """
        final_lists=[]
        final_abstracts = {}
        for i in range(search_times):
            keywords = self.keywords_agent.execute(keywords_prompt,keywords_system_prompt)
            print(f"The Keywords:\n{keywords}")
            lists,abstracts = search_from_arxiv(keywords,max_results=10)
            final_lists += lists
            final_lists = list(set(final_lists))
            final_abstracts.update(abstracts)
            check_prompt = f"""
            Analyze the attached abstracts for my literature review on {topic}. Specifically:

            Gap Identification: Highlight gaps in:
            Emerging sub-topics 
            Understudied populations/geographies
            Interdisciplinary connections (e.g., [field X + field Y])
            Contradictory findings or unresolved debates
            Methodological Critique: Flag overused/absent methodologies (e.g., lack of longitudinal studies).
            Search Recommendations: Suggest:
            Boolean search terms (e.g., '[concept A] AND [method B] NOT [method C]')
            Target databases 
            Filters (e.g., 2019-2024, high-citation journals)
            Prioritization: Rank gaps by urgency.
            Format output as bullet points with brief justifications.
            Existing abstracts:
            """
            for title,abstract in final_abstracts.items():
                check_prompt += f"title:{title},abstract:{abstract}\n"
            
            literature_gaps = self.check_agent.execute(check_prompt,check_system_prompt)
            keywords_prompt = f"""
            Please continue to provide keyword based on the provided suggestions for search.
            Generate non-overlapping search terms targeting those gaps
            Suggestions:{literature_gaps}

            The new keyword must contain at least 1 conceptual component absent from original queries
            Use Boolean exclusion patterns (e.g., NOT, NEAR) to filter redundant results
            Prioritize cross-domain terminology over incremental variations

            Keep the output format consistent with before and only generate one keyword
            Don't make keyword too long to avoid not finding results in the search.
            You can change some of the previous keyword appropriately to target gaps.
            """

        
        filter_system_prompt = f"""
        Act as a rigorous academic filter. Strictly evaluate provided paper title-abstract pairs. Retain ONLY entries that:
        1. Directly address {topic}
        2. Review or Survey
        Output MUST be a valid Python string in format "title 1,title 2,...title k" WITHOUT any commentary.
        """
        filter_prompt = '\n'.join([f"title:{k},abstract:{v}" for k,v in final_abstracts.items()])

        filtered_abstract = self.filter_agent.execute(filter_prompt,filter_system_prompt)
        final_abstracts = safe_parse(filtered_abstract)

        with open(f'{topic}_reference.json','w',encoding='utf-8') as json_file:
            json.dump(final_abstracts,json_file,ensure_ascii=False,indent=4)

        outline_system_prompt = f"""
        You are an academic research assistant specialized in synthesizing literature reviews.
        Analyze the provided collection of literature abstracts, identify key themes, methodologies, findings, and gaps. 
        Generate a structured two-level outline in English using plain markdown formatting. 
        Present only the outline itself without any explanations or commentary.
        """
        outline_prompt = f"""
        Based on the following collection of research paper abstracts: {filtered_abstract}, generate a comprehensive literature review outline containing exactly two hierarchical levels. Format the output as a markdown text following this exact pattern:

        # MAIN_CATEGORY_1
        ## 1.1. SUBTOPIC_1
        ## 1.2. SUBTOPIC_2
        ## 1.k. SUBTOPIC_K
        # MAIN_CATEGORY_2
        ## 2.1. SUBTOPIC_1
        ## 2.2. SUBTOPIC_2
        ...
        Ensure all entries use title case and maintain strict hierarchical relationships between levels. Exclude section numbering explanations and focus on conceptual organization."

        The outline depth and specific categories will vary based on the input abstracts' content.
        """
        if example_review_outline:
            outline_prompt += f"example outline:{example_review_outline}"
        outline = self.outline_agent.execute(outline_prompt,outline_system_prompt)

        outline_dict = outline2dict(outline)

        review_system_prompt = f"""
        You are an academic writing expert tasked with synthesizing specific sections of literature reviews in the field of {topic}. 
        Analyze the provided research abstracts and the target subsection outline to generate focused, logically 
        structured academic content. Maintain rigorous citation practices, highlight key findings and research gaps, 
        and ensure critical analysis of relationships between studies. Write in formal academic English without subjective language.
        abstract:{filtered_abstract}
        """
        first_generate = True
        for section,value in outline_dict.items():
            if isinstance(value,dict):
                for subsection,_ in value.items():
                    
                    review_prompt = f"""
                    Using the following collection of research abstracts:
                    {filtered_abstract}
                    The total outline is :
                    {outline}
                    And this target subsection outline:
                    {subsection}

                    Prioritize:

                    Thematic cohesion over chronological reporting
                    Explicit linkage between study findings
                    Identification of contradictory evidence

                    Only return plain text content without markdown
                    """ if first_generate else f"""
                    And this target subsection outline:
                    {subsection}

                    Prioritize:

                    Thematic cohesion over chronological reporting
                    Explicit linkage between study findings
                    Identification of contradictory evidence

                    Only return plain text content without markdown
                    
                    """
                    review_subsection = self.review_agent.execute(review_prompt,review_system_prompt)
                    outline_dict[section][subsection] = review_subsection

                    first_generate = False
        with open(f'{topic}_review.json','w',encoding='utf-8') as json_file:
            json.dump(outline_dict,json_file,ensure_ascii=False,indent=4)

        review = generate_document(outline_dict)

        safe_title = "".join(c for c in topic if c.isalnum() or c in " _-").rstrip()[:200] + "_original.md"
        save_path = os.path.join(save_dir,safe_title)

        with open(save_path,'w',encoding='utf-8') as write_file:
            write_file.write(review)

        score_system_prompt = f"""
        Act as a rigorous academic peer reviewer specializing in literature review evaluation. Analyze the provided literature review content against three defined quality criteria: Coverage, Structure, and Relevance. Conduct systematic assessment by:

        Matching content to evaluation rubric descriptions
        Citing specific textual evidence for scoring decisions
        Generating actionable revision suggestions
        Maintain objective tone and focus on measurable improvements.Present findings in structured JSON format without markdown.
        
        """

        revise_system_prompt = f"""
        Act as an academic editor implementing peer review suggestions. Revise the literature review by strictly following the provided evaluation feedback while preserving core arguments. Prioritize:

        Precise incorporation of missing content specified in 'suggestion' fields
        Structural adjustments for improved logical flow
        Enhanced thematic focus through relevance optimizations
        Maintenance of original citation integrity and academic tone
        Return only the revised text without commentary or markup.
        """

        for i in range(score_times):

            score_prompt = f"""
            Evaluate this literature review subsection:
            {review}
            All references:
            {filtered_abstract}
            Against these evaluation criteria: 
            Coverage: Coverage assesses the extent to which the survey encapsulates all relevant aspects of the topic, ensuring comprehensive discussion on both central and peripheral topics.
            score 1:The survey has very limited coverage, only touching on a small portion of the topic and lacking discussion on key areas.
            score 2:The survey covers some parts of the topic but has noticeable omissions, with significant areas either underrepresented or missing.
            score 3:The survey is generally comprehensive in coverage but still misses a few key points that are not fully discussed.'
            score 4:The survey covers most key areas of the topic comprehensively, with only very minor topics left out.
            score 5:The survey comprehensively covers all key and peripheral topics, providing detailed discussions and extensive information.

            Structure: Structure evaluates the logical organization and coherence of sections and subsections, ensuring that they are logically connected.
            score 1:The survey lacks logic, with no clear connections between sections, making it difficult to understand the overall framework.
            score 2:The survey has weak logical flow with some content arranged in a disordered or unreasonable manner.
            score 3:The survey has a generally reasonable logical structure, with most content arranged orderly, though some links and transitions could be improved such as repeated subsections.
            score 4:The survey has good logical consistency, with content well arranged and natural transitions, only slightly rigid in a few parts.
            score 5:The survey is tightly structured and logically clear, with all sections and content arranged most reasonably, and transitions between adajecent sections smooth without redundancy.

            Relevance: Relevance measures how well the content of the survey aligns with the research topic and maintain a clear focus.
            score 1:The  content is outdated or unrelated to the field it purports to review, offering no alignment with the topic.
            score 2:The survey is somewhat on topic but with several digressions; the core subject is evident but not consistently adhered to.
            score 3:The survey is generally on topic, despite a few unrelated details.
            score 4:The survey is mostly on topic and focused; the narrative has a consistent relevance to the core subject with infrequent digressions.
            score 5:The survey is exceptionally focused and entirely on topic; the article is tightly centered on the subject, with every piece of information contributing
                        to a comprehensive understanding of the topic.
            """
            suggestion = self.score_agent.execute(score_prompt,score_system_prompt)

            revise_prompt = f"""
            Original literature review content:
            {review}
            Required modifications:
            {suggestion}

            Generate the revised subsection following these requirements:  
            - Implement all suggestions from 'Composite_Feedback'  
            - Preserve 3-paragraph structure unless restructuring is explicitly requested   
            - Highlight changes through improved transitional phrases and expanded comparisons  

            Return only the revised text without commentary or markup.
            """
            review = self.revise_agent.execute(revise_prompt,revise_system_prompt)
        
        safe_title = "".join(c for c in topic if c.isalnum() or c in " _-").rstrip()[:200] + ".md"
        save_path = os.path.join(save_dir,safe_title)

        with open(save_path,'w',encoding='utf-8') as write_file:
            write_file.write(review)

    def outline_generate(self,abstracts,outline_template):
        outline_template = "\n".join([line for line in outline_template.splitlines() if line.lstrip().startswith("# ") and not line.lstrip().startswith("##")])

        filter_system_prompt = generate_prompt(FILTER_FOR_OUTLINE_SYSTEM_PROMPT)
        first_outline_system_prompt = generate_prompt(FIRST_OUTLINE_SYSTEM_PROMPT)
        abstracts_str = '\n'.join([f"[{i+1}] title:{abstract['title']},abstract:{abstract['abstract']}" for i,abstract in enumerate(abstracts)])
        first_outline_prompt = generate_prompt(FIRST_OUTLINE_PROMPT,abstracts_str=abstracts_str,example_outline=outline_template)
        first_outline = asyncio.run(self.outline_agent_async.execute_once(first_outline_prompt,first_outline_system_prompt))
        first_outline_dict = outline2dict(first_outline)
        second_outline_keys = list(first_outline_dict.keys())[1:6]
        
        second_outline_system_prompt = generate_prompt(SECOND_OUTLINE_SYSTEM_PROMPT)
        requests = []
        for section in second_outline_keys:
            all_scores_list, valid_abstracts = self.scoring(abstracts,section,'',FILTER_FOR_OUTLINE_PROMPT,filter_system_prompt)

            scores_abstracts_pairs = list(zip(valid_abstracts, all_scores_list))
            # 按照分数排序
            sorted_pairs = sorted(scores_abstracts_pairs, key=lambda x: x[1], reverse=True)
            current_abstracts = [pair[0] for pair in sorted_pairs if pair[1]][:40]
            current_abstracts_str = '\n'.join([f"[] title:{abstract['title']},abstract:{abstract['abstract']}" for abstract in current_abstracts])
            second_outline_prompt = generate_prompt(SECOND_OUTLINE_PROMPT,abstracts_str=current_abstracts_str,section=section)
            requests.append((second_outline_prompt,second_outline_system_prompt,None))
        results = asyncio.run(self.outline_agent_async.execute(requests))
        outline = {"1.Introduction":""}
        for result in results:
            result_dict = outline2dict(result)
            for key, val in result_dict.items():
                if key not in outline:
                    outline[key] = {}
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        outline[key][sub_key] = sub_val
                else:
                    outline[key] = val
        for key in list(first_outline_dict.keys())[6:]:
            outline[key] = ""
        outline_dict = outline_normalize(outline,mode='outline')

        with open(f'outline.json','w',encoding='utf-8') as json_file:
            json.dump(outline_dict,json_file,ensure_ascii=False,indent=4)
        
        outline_str = ""
        for section, subsections in outline_dict.items():
            outline_str += f"{section}\n"
            if isinstance(subsections, dict):
                for subsection in subsections:
                    outline_str += f"    {subsection}\n"
        outline = outline_str.strip()
        return outline,outline_dict

    def scoring(self,abstracts,section,subsection,filter_prompt_template,filter_system_prompt):
        group_num = len(abstracts) // 30 if len(abstracts) % 30 == 0 else len(abstracts) // 30 + 1

        all_scores_list = []
        valid_abstracts = []
        requests = []

        for i in range(group_num):
            filtered_abstracts_patch = abstracts[i * 30:(i + 1) * 30]
            abstracts_str = '\n'.join([f"[] title:{abstract['title']},abstract:{abstract['abstract']}" for abstract in filtered_abstracts_patch])
            if filter_prompt_template == FILTER_FOR_OUTLINE_PROMPT:
                filter_prompt = generate_prompt(filter_prompt_template, abstracts_str=abstracts_str, ref_num=len(filtered_abstracts_patch), section=section)
            elif filter_prompt_template == FILTER_PROMPT:
                filter_prompt = generate_prompt(filter_prompt_template, abstracts_str=abstracts_str, ref_num=len(filtered_abstracts_patch), section=section, subsection=subsection)
            requests.append((filter_prompt,filter_system_prompt,None))
               
        results = asyncio.run(self.score_agent.execute(requests))
        max_attempts = 3
        for i in range(group_num):
            if results[i]:
                attempt = 0
                len_abstracts = 30 if i < group_num - 1 else len(abstracts) % 30
                while attempt < max_attempts:
                    try:
                        scores_str = results[i]
                        scores_list = [int(num) for num in re.findall(r':\s*(\d+)\b', scores_str)]
                        if len(scores_list) == len_abstracts and all(1 <= score <= 10 for score in scores_list):
                            all_scores_list.extend(scores_list)
                            valid_abstracts.extend(abstracts[i * 30:(i + 1) * 30])
                            break
                        else:
                            scores_str = asyncio.run(self.score_agent.execute([requests[i]]))
                            print(f"Invalid scores length or range: {scores_list}")
                    except Exception as e:
                        scores_str = asyncio.run(self.score_agent.execute([requests[i]]))
                        print(f"Error during filtering: {e}")
                        import traceback
                        traceback.print_exc()
                    attempt += 1
        return all_scores_list,valid_abstracts

    def subsection_review_generate(self, 
                                   topic,
                                   abstracts, 
                                   section, 
                                   subsection, 
                                   outline_dict, 
                                   draft,
                                   pdf_accessible,
                                   filter_system_prompt,
                                   used_abstracts,):
        
        all_scores_list, valid_abstracts = self.scoring(abstracts,section,subsection,FILTER_PROMPT,filter_system_prompt)

        high_score_count = sum(1 for score in all_scores_list if score >= 6)
        ratio = int((high_score_count / len(all_scores_list) - 0.6)*50)

        scores_abstracts_pairs = list(zip(valid_abstracts, all_scores_list))
        section_key = section if not subsection else f"{section}_{subsection}"
        for i in range(len(scores_abstracts_pairs)):
            self.score[scores_abstracts_pairs[i][0]['title']][section_key] = scores_abstracts_pairs[i][1]
        #按照分数排序
        sorted_pairs = sorted(scores_abstracts_pairs, key=lambda x: x[1], reverse=True)
        current_abstracts = [pair[0] for pair in sorted_pairs if pair[1]>=8]

        new_index = len(used_abstracts) + 1  # 新编号起点
        numbered_abstracts = []
        for abstract in current_abstracts:
            if abstract in used_abstracts:
                # 如果文献已存在，使用其在 used_abstracts 中的编号
                index = next(j + 1 for j, used_abstract in enumerate(used_abstracts) if used_abstract == abstract)
            else:
                # 如果是新文献，分配新的编号
                index = new_index
                new_index += 1
            numbered_abstracts.append(f"[{index}] title:{abstract['title']},abstract:{abstract['abstract']}")
        current_abstracts_str = '\n'.join(numbered_abstracts)

        for abstract in current_abstracts:
            if abstract not in used_abstracts:
                used_abstracts.append(abstract)
        
        review_generated = generate_document(outline_dict)

        pdf_accessible_current = [
            pdf for pdf in pdf_accessible
            if any(pdf in abstract['title'] for abstract in current_abstracts)
        ]
        pdf_accessible_str = '\n'.join([f"{pdf}" for i,pdf in enumerate(pdf_accessible_current)])

        outline_str = ""
        for key, value in outline_dict.items():
            outline_str += f"{key}\n"
            if isinstance(value, dict):
                for sub in value:
                    outline_str += f"    {sub}\n"
        outline = outline_str.strip()
        review_system_prompt = generate_prompt(REVIEW_SYSTEM_PROMPT,topic=topic,pdf_accessible_str=pdf_accessible_str)
        review_prompt = generate_prompt(REVIEW_PROMPT,current_abstracts_str=current_abstracts_str,section = section,subsection=subsection,outline=outline,draft=draft,review_generated=review_generated)

        if subsection:
            review_subsection = self.review_agent.execute_with_functions(review_prompt,review_system_prompt,max_tokens=800+20*ratio)
            outline_dict[section][subsection] = review_subsection
        else:
            review_subsection = self.review_agent.execute_with_functions(review_prompt,review_system_prompt,max_tokens=1000+20*ratio)
            outline_dict[section] = review_subsection
        self.review_agent.clear_history()

    def revise(self,outline_dict,topic,used_abstracts_str,suggestion):
        """
        使用多线程对每个 section 调用 revise_section，并按顺序组合最后的结果。
        """
        revise_system_prompt = generate_prompt(REVISE_SYSTEM_PROMPT,topic=topic)
        def process_section(section, value):
            """
            处理单个 section 的修订。
            """
            requests = []
            if isinstance(value, dict):
                for subsection, _ in value.items():
                    review_section = outline_dict[section][subsection]
                    revise_prompt = generate_prompt(REVISE_PROMPT,review_section=review_section,used_abstracts_str=used_abstracts_str,suggestion=suggestion)
                    requests.append((revise_prompt, revise_system_prompt, None))
            else:
                review_section += outline_dict[section]
                revise_prompt = generate_prompt(REVISE_PROMPT,review_section=review_section,used_abstracts_str=used_abstracts_str,suggestion=suggestion)
                requests.append((revise_prompt, revise_system_prompt, None))
            return requests

        requests = []
        for section, value in outline_dict.items():
            requests.extend(process_section(section, value))
        results = asyncio.run(self.revise_agent.execute(requests))
        final_review = ""
        for result in results:
            if result:
                final_review += result + "\n\n"

        return final_review

    def forward_with_references(self,topic,abstracts,outline_template = None,save_dir = 'review',):
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        abstracts_str = '\n'.join([f"[{i+1}] title:{abstract['title']},abstract:{abstract['abstract']}" for i,abstract in enumerate(abstracts)])

        # outline_system_prompt = generate_prompt(OUTLINE_SYSTEM_PROMPT,topic=topic)

        # if not outline_template:
        #     outline_template = f"""
        #     # MAIN_CATEGORY_1
        #     ## 1.1. SUBTOPIC_1
        #     ## 1.2. SUBTOPIC_2
        #     ## 1.k. SUBTOPIC_K
        #     # MAIN_CATEGORY_2
        #     ## 2.1. SUBTOPIC_1
        #     ## 2.2. SUBTOPIC_2
        #     ## ...
        #     ## 2.n. SUBTOPIC_N
        #     # ...
        #     """
        
        # outline_prompt = generate_prompt(OUTLINE_PROMPT,abstracts_str=abstracts_str,example_outline=outline_template)
        
        # outline = self.outline_agent.execute(outline_prompt,outline_system_prompt)

        # outline_dict = outline2dict(topic,outline)

        outline,outline_dict = self.outline_generate(abstracts,outline_template)
        print(f"Outline:\n{outline}")
        for abstract in abstracts:
            self.score[abstract['title']] = {}
            for section ,subsection in outline_dict.items():
                if isinstance(subsection,dict):
                    for sub in subsection.keys():
                        self.score[abstract['title']][f"{section}_{sub}"] = 0
                else:
                    self.score[abstract['title']][section] = 0

        draft_system_prompt = generate_prompt(DRAFT_SYSTEM_PROMPT)
        draft_prompt = generate_prompt(DRAFT_PROMPT,topic=topic,abstracts_str=abstracts_str)

        draft = self.draft_agent.execute(draft_prompt,draft_system_prompt)
        total_subsections = sum(len(v) if isinstance(v, dict) else 1 for v in outline_dict.values())

        used_abstracts = []
        filter_system_prompt = generate_prompt(FILTER_SYSTEM_PROMPT)

        pdf_files = glob.glob(os.path.join(self.pdf_dir, "*.[pP][dD][fF]"))
        pdf_accessible = [os.path.splitext(os.path.basename(pdf))[0] for pdf in pdf_files]

        with Progress() as progress:
            # 创建进度条任务
            task = progress.add_task("[bold green]生成内容...", total=total_subsections)

            for section,value in outline_dict.items():
                if isinstance(value,dict):
                    for subsection,_ in value.items():

                        progress.print(f"正在处理: [bold yellow]{section}[/] -> [bold cyan]{subsection}[/]")

                        self.subsection_review_generate(
                            topic=topic,
                            abstracts=abstracts,
                            section=section,
                            subsection=subsection,
                            outline_dict=outline_dict,
                            draft=draft,
                            pdf_accessible=pdf_accessible,
                            filter_system_prompt=filter_system_prompt,
                            used_abstracts=used_abstracts
                        )

                        progress.update(task, advance=1)
                else:
                    progress.print(f"正在处理: [bold yellow]{section}")
                    self.subsection_review_generate(
                        topic=topic,
                        abstracts=abstracts,
                        section=section,
                        subsection=value,
                        outline_dict=outline_dict,
                        draft=draft,
                        pdf_accessible=pdf_accessible,
                        filter_system_prompt=filter_system_prompt,
                        used_abstracts=used_abstracts
                    )
                    progress.update(task, advance=1)

        with open(f'ref_score.json','w',encoding='utf-8') as json_file:
            json.dump(self.score,json_file,ensure_ascii=False,indent=4)
        
        with open(f'{topic}_review.json','w',encoding='utf-8') as json_file:
            json.dump(outline_dict,json_file,ensure_ascii=False,indent=4)

        review = generate_document(outline_dict)

        safe_title = "".join(c for c in topic if c.isalnum() or c in " _-").rstrip()[:200] + "_original.md"
        save_path = os.path.join(save_dir,safe_title)

        used_abstracts_str = '\n'.join([f"[{i+1}] title:{abstract['title']},abstract:{abstract['abstract']}" for i,abstract in enumerate(used_abstracts)])
        ref = '\n'.join([f"[{i+1}] {abstract['title']}" for i,abstract in enumerate(used_abstracts)])

        with open(f'{topic}_used_abstracts.json', 'w', encoding='utf-8') as json_file:
            json.dump(used_abstracts, json_file, ensure_ascii=False, indent=4)

        review_ref = review + f"\nReferences\n\n{ref}\n"

        with open(save_path,'w',encoding='utf-8') as write_file:
            write_file.write(review_ref)
        
        try:
            deep_reviewer = DeepReviewer(model_size='14B',device='cuda:2')
            review_result = deep_reviewer.evaluate(
                paper_context=review,
                mode="Standard Mode",
                reviewer_num=4,
            )
            meta_review = review_result[0].get('meta_review')
            if isinstance(meta_review, dict):
                suggestion = meta_review.get('suggestions')

            if suggestion:
                review = self.revise(outline_dict,topic,used_abstracts_str,suggestion)

                review += f"\nReferences\n\n{ref}\n"

                safe_title = "".join(c for c in topic if c.isalnum() or c in " _-").rstrip()[:200] + ".md"
                save_path = os.path.join(save_dir,safe_title)

                with open(save_path,'w',encoding='utf-8') as write_file:
                    write_file.write(review)

                review_result = deep_reviewer.evaluate(
                    paper_context=review,
                    mode="Standard Mode",
                    reviewer_num=4,
                )
                review_result = review_result[0]['meta_review']['content']
                with open(f"{save_dir}/deepreviewer_result.md",'a',encoding='utf-8') as file:
                    file.write(f"{topic}\n{review_result}\n")
        except Exception as e:
            print(f"Error during review: {e}")
            import traceback
            traceback.print_exc()

        print('[red]Workflow completed[/red]')


if __name__ == "__main__":
    start_time = time.time()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    reason_model = "deepseek-reasoner"
    model = "deepseek-chat"
    base_url = "https://api.deepseek.com"
    pdf_dir="ref_get/download_files"
    review_flow = ReviewFlow(api_key=api_key, model=model, reason_model=reason_model, base_url=base_url, pdf_dir=pdf_dir, max_retries=3)
    topic = "graph signal sampling and reconstruction"
    with open("ref_get/template_outline.md", "r", encoding="utf-8") as f:
        outline_template = f.read()
    with open("ref_get/download_files/filtered_abstracts.json", "r", encoding="utf-8") as f:
        abstracts = json.load(f)
    # review_flow.outline_generate(topic, abstracts, outline_template=outline_template)
    review_flow.forward_with_references(topic, abstracts, outline_template=outline_template, save_dir='review')
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"用时{hours}小时{minutes}分{seconds}秒")
