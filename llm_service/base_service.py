'''
Descripttion: 
Author: Duke 叶兀
E-mail: ljyduke@gmail.com
Date: 2024-01-06 00:12:20
LastEditors: Duke 叶兀
LastEditTime: 2024-01-17 00:47:53
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
        self.baidu_api_key = BAIDU_API_KEY
        self.baidu_secret_key = BAIDU_SECRET_KEY

        pass

    def _get_access_token(self, USER_BAIDU_API_KEY='', USER_BAIDU_SECRET_KEY=''):
        """
        使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
        """
        if USER_BAIDU_API_KEY and USER_BAIDU_SECRET_KEY:
            self.baidu_api_key = USER_BAIDU_API_KEY
            self.baidu_secret_key = USER_BAIDU_SECRET_KEY

        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.baidu_api_key}&client_secret={self.baidu_secret_key}"
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token", "")
