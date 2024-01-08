'''
Descripttion: 
Author: Duke 叶兀
E-mail: ljyduke@gmail.com
Date: 2024-01-06 08:55:19
LastEditors: Duke 叶兀
LastEditTime: 2024-01-08 23:30:57
'''
from llm_service.yi import YIService
from llm_service.glm import GLMService

yi = YIService()
glm = GLMService()

# def LLM():
#     """ initialize different LLM instance according to the key field existence"""
#     # TODO a little trick, can use registry to initialize LLM instance further
#     if CONFIG.openai_api_key:
#         llm = OpenAIGPTAPI()
#     elif CONFIG.claude_api_key:
#         llm = Claude()
#     elif CONFIG.spark_api_key:
#         llm = SparkAPI()
#     elif CONFIG.zhipuai_api_key:
#         llm = ZhiPuAIGPTAPI()
#     else:
#         raise RuntimeError("You should config a LLM configuration first")

#     return llm
