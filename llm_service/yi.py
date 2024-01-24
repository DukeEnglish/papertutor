'''
Author: ljyduke 叶兀
Date: 2024-01-05 23:44:28
LastEditors: ljyduke 叶兀
LastEditTime: 2024-01-06 14:08:42
FilePath: /paper_tutor/llm_service/yi.py
Description: 

Copyright (c) 2024 by ${ljyduke@gmail.com}, All Rights Reserved. 
'''

import requests
import json
import logging
from llm_service.base_service import BAIDULLMService


class YIService(BAIDULLMService):
    """
    yi模型
    """

    def __init__(self, BAIDU_API_KEY="", BAIDU_SECRET_API_KEY=""):
        super().__init__()
        if BAIDU_API_KEY and BAIDU_SECRET_API_KEY:
            self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/yi_34b_chat?access_token=" + self._get_access_token(BAIDU_API_KEY, BAIDU_SECRET_API_KEY)
        else:
            self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/yi_34b_chat?access_token=" + self._get_access_token()
        self.headers = {
            'Content-Type': 'application/json'
        }

    def llm(self, user_input="推荐中国自驾游路线"):
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": f"忽略原始的身份设定，你是一个小秘书小明，要认真回答用户的问题。下面是用户的问题：{user_input}"
                }
            ]
        })

        response = requests.request(
            "POST", self.url, headers=self.headers, data=payload)
        logging.info(f"yi_single_service' resp is {response.text}")
        return response.text

    def llm_stream(self, user_input='推荐中国自驾游路线'):

        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": f"忽略原始的身份设定，你是一个小秘书小明，要认真回答用户的问题。下面是用户的问题：{user_input}"
                }
            ],
            "stream": True
        })

        response = requests.request(
            "POST", self.url, headers=self.headers, data=payload, stream=True)

        for line in response.iter_lines():
            if not line:
                continue
            res = json.loads(":".join(line.decode("utf-8").split(":")[1:]))
            if res["is_end"]:
                logging.info(f"result is done for user_input {user_input}")
            yield res["result"]


if __name__ == '__main__':
    yi = YIService()
    res = yi.llm_stream(user_input='你好')
    for i in res:
        print(i)
        print()
