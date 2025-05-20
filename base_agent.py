import os
import json

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    wait_fixed
)
import pdfplumber
import logging
import glob
import asyncio

from concurrent.futures import ThreadPoolExecutor, as_completed

class AIAgent:
    def __init__(self, api_key, model, base_url, temperature=0.2,max_retries=3):
        """
        初始化OpenAI Agent
        :param api_key: OpenAI API密钥
        :param max_retries: 最大重试次数
        :param model: 使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.history = []
        self.logger = logging.getLogger(__name__)
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _create_chat_completion(self, messages, functions=None, function_call='auto', max_tokens=None):
        """实际执行API调用"""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            tools=functions,
            tool_choice=function_call,
            max_tokens=max_tokens,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (openai.OpenAIError, openai.APIError, openai.Timeout)
        ),
    )
    def _call_api_with_retry(self, messages,functions,function_call,max_tokens = None):
        """带重试机制的API调用"""
        try:
            return self._create_chat_completion(messages = messages,functions = functions,function_call = function_call,max_tokens = max_tokens)
        except openai.APIError as e:
            self.logger.warning(f"OpenAI API错误: {e}")
            raise
        except openai.APIConnectionError as e:
            self.logger.warning(f"网络连接错误: {e}")
            raise
        except openai.RateLimitError as e:
            self.logger.warning(f"API限速: {e}")
            raise
        except openai.Timeout as e:
            self.logger.warning(f"请求超时: {e}")
            raise

    def _format_history(self,system_prompt):
        """将历史记录格式化为OpenAI消息格式"""
        messages = [{"role":"system","content":system_prompt},]
        for item in self.history:
            messages.append({"role": "user", "content": item["user_input"]})
            if item["assistant_response"]:
                messages.append({"role": "assistant", "content": item["assistant_response"]})
        return messages

    def execute_once(self,user_input,system_prompt,max_tokens = None):
        try:
            messages = [{"role":"system","content":system_prompt},
                       {"role":"user","content":user_input}]
            response = self._call_api_with_retry(messages = messages,functions = None,function_call = None,max_tokens = max_tokens)
            assistant_response = response.choices[0].message.content.strip()
            return assistant_response
        except openai.APIError as e:
            self.logger.warning(f"OpenAI API错误: {e}")
            raise
        except openai.APIConnectionError as e:
            self.logger.warning(f"网络连接错误: {e}")
            raise
        except openai.RateLimitError as e:
            self.logger.warning(f"API限速: {e}")
            raise
        except openai.Timeout as e:
            self.logger.warning(f"请求超时: {e}")
            raise

    def execute(self, user_input,system_prompt,max_tokens = None):
        """
        执行对话
        :param user_input: 用户输入文本
        :return: 助手回复文本
        """
        try:
            # 添加用户输入到历史
            self.history.append({
                "user_input": user_input,
                "assistant_response": None
            })

            # 格式化历史消息
            messages = self._format_history(system_prompt)

            # 调用带重试的API
            response = self._call_api_with_retry(messages,functions = None,function_call = None,max_tokens = max_tokens)

            assistant_response = response.choices[0].message.content.strip()

            # 更新历史记录
            self.history[-1]["assistant_response"] = assistant_response

            return assistant_response

        except openai.APIError as e:
            self.logger.warning(f"OpenAI API错误: {e}")
            raise
        except openai.APIConnectionError as e:
            self.logger.warning(f"网络连接错误: {e}")
            raise
        except openai.RateLimitError as e:
            self.logger.warning(f"API限速: {e}")
            raise
        except openai.Timeout as e:
            self.logger.warning(f"请求超时: {e}")
            raise

    def clear_history(self):
        """清空对话历史"""
        self.history = []

    def get_history(self):
        """获取对话历史"""
        return self.history

class AsyncAIAgent:
    def __init__(self, api_key, model, base_url, temperature=0.2,max_retries=3):
        """
        初始化OpenAI Agent
        :param api_key: OpenAI API密钥
        :param max_retries: 最大重试次数
        :param model: 使用的模型名称
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        self.temperature = temperature
        self.client = openai.AsyncOpenAI(api_key=self.api_key, base_url=base_url)

    async def _create_chat_completion(self, messages, max_tokens=None):
        """实际执行API调用"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return response
    
    async def _call_api_with_retry(self, messages, max_tokens=None):
        """带重试机制的API调用"""
        for i in range(self.max_retries):
            try:
                return await self._create_chat_completion(messages=messages, max_tokens=max_tokens)
            except openai.APIError as e:
                self.logger.warning(f"OpenAI API错误: {e}")
                if i == self.max_retries - 1:
                    raise
            except openai.APIConnectionError as e:
                self.logger.warning(f"网络连接错误: {e}")
                if i == self.max_retries - 1:
                    raise
            except openai.RateLimitError as e:
                self.logger.warning(f"API限速: {e}")
                if i == self.max_retries - 1:
                    raise
            except openai.Timeout as e:
                self.logger.warning(f"请求超时: {e}")
                if i == self.max_retries - 1:
                    raise
            except Exception as e:
                self.logger.warning(f"发生错误: {e}")
                if i == self.max_retries - 1:
                    raise
 
    async def execute_once(self, user_input, system_prompt, max_tokens=None):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}]
        response = await self._call_api_with_retry(messages=messages, max_tokens=max_tokens)
        assistant_response = response.choices[0].message.content.strip()
        return assistant_response
    
    async def execute(self, requests):
        results = await asyncio.gather(*(
            self.execute_once(user_input, system_prompt, max_tokens)
            for user_input, system_prompt, max_tokens in requests))
        return results
            
class ReviewAgent(AIAgent):
    def __init__(self, api_key, model, reason_model, base_url, pdf_dir, temperature=0.2, max_retries=3):
        super().__init__(api_key, model, base_url, temperature, max_retries)
        self.pdf_dir = pdf_dir
        self.reason_model = reason_model
        self.temperature = temperature
        self.read_agent = AsyncAIAgent(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_retries=max_retries
        )
    
    def _create_reason_chat_completion(self, messages, max_tokens=None):
        return self.client.chat.completions.create(
            model=self.reason_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (openai.OpenAIError, openai.APIError, openai.Timeout)
        ),
    )
    def _call_reason_api_with_retry(self, messages, max_tokens=None):
        """带重试机制的API调用"""
        try:
            return self._create_reason_chat_completion(messages = messages, max_tokens = max_tokens)
        except openai.APIError as e:
            self.logger.warning(f"OpenAI API错误: {e}")
            raise
        except openai.APIConnectionError as e:
            self.logger.warning(f"网络连接错误: {e}")
            raise
        except openai.RateLimitError as e:
            self.logger.warning(f"API限速: {e}")
            raise
        except openai.Timeout as e:
            self.logger.warning(f"请求超时: {e}")
            raise

    
    def read_pdf(self, title_list, interest_list):
        all_results = [None] * len(title_list)
        valid_requests = []
        valid_indices = []
        for idx, (title, interest) in enumerate(zip(title_list, interest_list)):
            try:
                pdf_path = os.path.join(self.pdf_dir, f"{title}.pdf")
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = ''
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + '\n'
                system_prompt = "You are a research assistant responsible for extracting information of interest to users from literature."
                prompt = f"""Extract the information of interest to the user from the literature. The information of interest is: {interest}. 
                The literature is: {full_text}"""
                valid_requests.append((prompt, system_prompt, None))
                valid_indices.append(idx)
            except Exception as e:
                self.logger.warning(f"Error processing PDF for title '{title}': {e}")
                all_results[idx] = "Sorry"
        
        if valid_requests:
            results = asyncio.run(self.read_agent.execute(valid_requests))
            for i, res in zip(valid_indices, results):
                all_results[i] = res
        
        return all_results

    def execute_with_functions(self, user_input, system_prompt, max_tokens=None):
        """
        执行对话
        :param user_input: 用户输入文本
        :param system_prompt: 系统提示文本
        :return: 助手回复文本
        """
        try:
            # 添加用户输入到历史
            self.history.append({
                "user_input": user_input,
                "assistant_response": None
            })

            # 格式化历史消息
            messages = self._format_history(system_prompt)

            functions = [
                {
                    "type": "function",
                    "function":{
                        "name": "read_pdf",
                        "description": "Read a PDF file and extract text content.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "The title of the PDF file."
                                },
                                "interest": {
                                    "type": "string",
                                    "description": "The information of interest to the user."
                                }
                            },
                            "required": ["title", "interest"]
                        }
                    }
                }
            ]

            # 调用带重试的API
            response = self._call_api_with_retry(messages = messages,functions=functions,function_call={"type": "function", "function": {"name": "read_pdf"}},max_tokens = max_tokens)
            if response.choices[0].finish_reason == "tool_calls":
                messages.append(response.choices[0].message)
                tool_calls = response.choices[0].message.tool_calls
                tool_id_list = []
                title_list = []
                interest_list = []
                for tool in tool_calls:
                    if tool.function.name == "read_pdf":
                        # 获取函数参数
                        tool_arguments = json.loads(tool.function.arguments)
                        title = tool_arguments.get("title")
                        interest = tool_arguments.get("interest")

                        title_list.append(title.lower())
                        interest_list.append(interest)
                        tool_id_list.append(tool.id)

                results = self.read_pdf(title_list, interest_list)
                # 重新调用API
                information_from_tool = "Here is the information extracted from the literature:\n"
                for title,result in zip(title_list, results):
                    if result != "Sorry":
                        information_from_tool += f"Title: {title}\n"
                        information_from_tool += f"Content: {result}\n"
                messages = self._format_history(system_prompt)
                messages[-1]["content"] += information_from_tool
                response = self._call_reason_api_with_retry(messages=messages,max_tokens=max_tokens)
                
            assistant_response = response.choices[0].message.content.strip()

            # 更新历史记录
            self.history[-1]["assistant_response"] = assistant_response

            return assistant_response
            
        except openai.APIError as e:
            self.logger.warning(f"OpenAI API错误: {e}")
            raise
        except openai.APIConnectionError as e:
            self.logger.warning(f"网络连接错误: {e}")
            raise
        except openai.RateLimitError as e:
            self.logger.warning(f"API限速: {e}")
            raise
        except openai.Timeout as e:
            self.logger.warning(f"请求超时: {e}")
            raise
        except Exception as e:
            self.logger.warning(f"发生错误: {e}")
            raise
                

if __name__ == "__main__":
    import os

    # 初始化Agent
    agent = AsyncAIAgent(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model = "deepseek-chat",
        base_url="https://api.deepseek.com",
        max_retries=3
    )
    import time

    requests = [
        ("你好，请介绍一下Python。", "系统：请回答用户问题。", 100),
        ("你好，请介绍一下Java。", "系统：请回答用户问题。", 100),
        ("你好，请介绍一下C++。", "系统：请回答用户问题。", 100)
        ]

    start_time = time.time()
    results = asyncio.run(agent.execute(requests))
    end_time = time.time()
    print(f"总耗时：{end_time - start_time}秒")