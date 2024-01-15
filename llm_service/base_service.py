'''
Descripttion: 
Author: Duke 叶兀
E-mail: ljyduke@gmail.com
Date: 2024-01-06 00:12:20
LastEditors: Duke 叶兀
LastEditTime: 2024-01-16 01:21:18
'''

import json
import requests
from config import BAIDU_API_KEY, BAIDU_SECRET_KEY


class LLMBaseService():
    """
    LLM模型基础服务
    TODO:
    1. 增加用户管理，从而增加历史记录
    """

    def __init__(self):
        pass


class BAIDULLMService(LLMBaseService):
    """
    LLM模型基础服务
    TODO:
    1. 增加用户管理，从而增加历史记录
    """

    def __init__(self):
        """
        https://cloud.baidu.com/doc/WENXINWORKSHOP/s/vlpteyv3c#header%E5%8F%82%E6%95%B0
        """
        self.temperature = 1
        self.top_k = 1
        self.top_p = 1
        self.penalty_score = 0.5
        self.stop = []
        self.user_id = ""

        pass

    def _get_access_token(self):
        """
        使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
        """

        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={BAIDU_API_KEY}&client_secret={BAIDU_SECRET_KEY}"
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")
