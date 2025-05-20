from base_agent import AIAgent
import json
import os

with open("ref_get/download_files/abstracts.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)

filtered_abstracts = [item for item in abstracts if item.get("abstract") is not None]


api_key = os.getenv("DEEPSEEK_API_KEY")
model = "deepseek-chat"
base_url = "https://api.deepseek.com"
filter_agent = AIAgent(api_key=api_key, model=model, base_url=base_url, temperature=0.1, max_retries=3)
literature_num = 100
topic = "graph signal sampling and reconstruction"

group_num = len(filtered_abstracts)//30 + 1
all_scores_list = []
for i in range(group_num):
    filtered_abstracts_patch = filtered_abstracts[i*30:(i+1)*30]
    if not filtered_abstracts_patch:
        break
    abstracts_str = '\n'.join([f"[] title:{abstract['title']},abstract:{abstract['abstract']}" for j,abstract in enumerate(filtered_abstracts_patch)])
    
    filter_system_prompt = f"""
    You are an intelligent scholar specializing in academic literature review assistance. 
    Your task is to evaluate the relevance of research papers to a literature review being written.
    Output must strictly adhere to the following rules:
    1. The output must be a valid Python string in the format  title1:score1;title2:score2;...;titleN:scoreN 
    2. Each score must be a number between 1 and 10.
    3. The number of scores must exactly match the number of papers provided.
    4. Do not include any commentary, explanations, or invalid characters in the output.
    """
    filter_prompt = f"""
    I have a collection of {len(filtered_abstracts_patch)} paper titles and abstracts stored in abstracts_str, formatted as [] title1 abstract1. 
    Currently, I'm working on a literature review. For each paper in abstracts_str, 
    please assign a relevance score ranging from 1 to 10 based on how closely related the paper's content is to the 
    topic of the literature review. After evaluating all {len(filtered_abstracts_patch)} papers, output only a list of exactly {len(filtered_abstracts_patch)} scores 
    in the format  title1:score1;title2:score2;...;titleN:scoreN  without any additional text or explanations. 
    Ensure that:
    1. Each score is a number between 1 and 10.
    2. The number of scores matches the number of papers ({len(filtered_abstracts_patch)}).
    The scoring criteria are as follows:
    1. A score of 10 indicates the paper is highly relevant and directly addresses the topic.
    2. A score of 5 indicates the paper is somewhat relevant but not directly focused on the topic.
    3. A score of 1 indicates the paper is unrelated to the topic.
    The abstracts_str is:
    {abstracts_str}
    The Topic is:
    {topic}
    """

    max_attempts = 3
    attempt = 0
    scores_list = []
    while attempt < max_attempts:
        try:
            # 调用过滤器代理执行过滤操作
            scores_str = filter_agent.execute(filter_prompt, filter_system_prompt)
            filter_agent.clear_history()
            pairs = scores_str.strip().split(';')
            scores_list = [float(pair.split(':')[1]) for pair in pairs if pair]
            if len(scores_list) == len(filtered_abstracts_patch) and all(1 <= score <= 10 for score in scores_list):
                break
            attempt += 1
        except Exception as e:
            print(f"Error during filtering: {e}")
            attempt += 1
                
    if len(scores_list) == len(filtered_abstracts_patch) and all(1 <= score <= 10 for score in scores_list):
        all_scores_list.extend(scores_list)
    else:
        print(f"Error: Invalid scores list length or values. Attempt {attempt+1} failed.")
        continue

scores_abstracts_pairs = list(zip(filtered_abstracts, all_scores_list))
# 按照分数排序
sorted_pairs = sorted(scores_abstracts_pairs, key=lambda x: x[1], reverse=True)

filtered_abstracts = [pair[0] for pair in sorted_pairs[:250]]

print(f"Filtered abstracts count: {len(filtered_abstracts)}")

with open("ref_get/download_files/filtered_abstracts.json", "w", encoding="utf-8") as f:
    json.dump(filtered_abstracts, f, ensure_ascii=False, indent=4)

