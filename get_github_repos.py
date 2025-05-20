import requests
from urllib.parse import quote

def get_all_issues(github_token, owner, repo):
    """
    获取指定仓库下所有 Issues
    :param github_token: GitHub 令牌
    :param owner: 仓库拥有者
    :param repo: 仓库名称
    :return: 一个包含所有 Issue 数据的列表
    """
    issues = []
    per_page = 100  # 每页最大数量
    page = 1
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    while True:
        url = f'https://api.github.com/repos/{owner}/{repo}/issues?state=all&per_page={per_page}&page={page}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            page_issues = response.json()
            if not page_issues:
                break
            issues.extend(page_issues)
            page += 1
        else:
            raise Exception(f"获取Issues失败，状态码 {response.status_code}: {response.text}")
    return issues

def get_issue_comments(github_token, owner, repo, issue_number):
    """
    获取指定 Issue 下的评论
    :param github_token: GitHub 令牌
    :param owner: 仓库拥有者
    :param repo: 仓库名称
    :param issue_number: Issue 编号
    :return: 评论列表
    """
    comments = []
    page = 1
    per_page = 100
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    while True:
        url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page={per_page}&page={page}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            page_comments = response.json()
            if not page_comments:
                break
            comments.extend(page_comments)
            page += 1
        else:
            raise Exception(f"获取评论失败，状态码 {response.status_code}: {response.text}")
    return comments


def search_github_repos(github_token,query, limit=5):
    """
    Search GitHub public repositories based on a keyword.

    :param query: The query to search for in repository names or descriptions.
    :param limit: The total number of repositories to return.
    :return: A list of dictionaries containing repository details, limited to the specified number.
    """

    repos = []
    per_page = 10
    page = 1

    while len(repos) < limit:
        encoded_query = quote(query)
        url = f'https://api.github.com/search/repositories?q={encoded_query}&per_page={per_page}&page={page}'

        headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            items = response.json().get('items', [])
            for item in items:
                formatted_repo = {
                    "name": f"{item['name']}",
                    "author": item['owner']['login'],
                    "description": item['description'],
                    "link": item['html_url'], 
                    "stars": item['stargazers_count'], 
                    "created_at": item['created_at'], 
                    "language": item['language']
                }
                # print(item)
                repos.append(formatted_repo)
                if len(repos) >= limit:
                    break

            if len(items) < per_page:  # Stop if there are no more repos to fetch
                break
            page += 1
        else:
            raise Exception(f"GitHub API request failed with status code {response.status_code}: {response.text}")

    return_str = f"The results of searching {query} on GitHub: \n"

    repo = repos[0]
    return_str += f"""
    Name: {repo['name']}
    Description: {repo['description']}
    Link: {repo['link']}
    Stars: {repo['stars']}
    Created at: {repo['created_at']}
    Language: {repo['language']}
    """
    
    issue_str = ""
    issues = get_all_issues(github_token,repo["author"],repo["name"])
    for issue in issues:
        if issue["state"] == "open":
            issue_str += f"""
            Issue Title: {issue['title']}
            Issue Link: {issue['html_url']}
            """
            comments = get_issue_comments(github_token,repo["author"],repo["name"],issue["number"])
            for comment in comments:
                issue_str += f"""
                Comment by {comment['user']['login']}:
                {comment['body']}
                """

    return return_str,issue_str

if __name__ =="__main__":
    github_token = "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"
    query = "Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction"
    repo,issue = search_github_repos(github_token,query,limit=5)
    print(repo)
    print(issue)