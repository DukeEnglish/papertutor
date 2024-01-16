'''
Descripttion: 
Author: Duke 叶兀
E-mail: ljyduke@gmail.com
Date: 2024-01-06 12:55:46
LastEditors: Duke 叶兀
LastEditTime: 2024-01-17 00:44:14
'''

import requests
import json
import logging
from llm_service.base_service import BAIDULLMService


class GLMService(BAIDULLMService):

    def __init__(self, BAIDU_API_KEY="",  BAIDU_SECRET_API_KEY=""):
        super().__init__()
        if BAIDU_API_KEY and BAIDU_SECRET_API_KEY:
            self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/chatglm2_6b_32k?access_token=" + self._get_access_token(BAIDU_API_KEY, BAIDU_SECRET_API_KEY)
        else:
            self.url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/chatglm2_6b_32k?access_token=" + self._get_access_token()
        self.headers = {
            'Content-Type': 'application/json'
        }

    def llm(self, user_input="推荐中国自驾游路线"):
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": f"下面是用户的问题：{user_input}"
                }
            ]
        })

        response = requests.request("POST", self.url, headers=self.headers, data=payload)
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

        response = requests.request("POST", self.url, headers=self.headers, data=payload, stream=True)

        for line in response.iter_lines():
            if not line:
                continue
            res = json.loads(":".join(line.decode("utf-8").split(":")[1:]))
            if res["is_end"]:
                logging.info(f"result is done for user_input {user_input}")
            yield res["result"]


if __name__ == '__main__':
    # glm = GLMService()
    # res = glm.llm(user_input='你好，你是谁')
    # print(res)

    def test(messages: list[str]) -> str:
        print(type(messages[0]))
        return messages[0]

    print(test(["1"]))

    # for i in res:
    #     print(i)
    #     print()
