'''
Descripttion: 
Author: Duke 叶兀
E-mail: ljyduke@gmail.com
Date: 2024-01-07 17:31:50
LastEditors: Duke 叶兀
LastEditTime: 2024-01-08 23:32:33
'''
from paper_related import arxiv_client
from datetime import datetime
import os
import requests
import json
import pdfplumber
import codecs
import io
from llm_service import glm

today_date = datetime.today().strftime('%Y%m%d')
print(f"today is {today_date}")

PAPER_CLS_LIST = ["cs.CL"]


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")


class PaperParser:
    """Paper parser. It should get metainfo of paper and parse its info
    """

    def __init__(self,):
        """init
        """
        self.arxiv_client = arxiv_client
        self.metainfo_save_path = None
        self.paper_parse_data_save_path = None
        self._init_save_path()
        self.paper_cls_list = PAPER_CLS_LIST

    def parse_paper(self,):
        # 将论文原始数据保留下来
        for paper_cls in self.paper_cls_list:
            self.get_paper(paper_cls)
        # 接下来读取保留下来的数据，然后进行下载
        # 打开data/20240108/meta/cs.CL.json，读取内容，根据其中的pdf_url和title，调用这个函数download_paper(url, title, self.paper_parse_data_save_path)
        for paper_cls in self.paper_cls_list:
            # 下载并且转化为了txt
            self.save_paper_txt(paper_cls)
        # 最后进行qa，暂时暂停这部分，先测试git action
        # for paper_cls in self.paper_cls_list:
        #     self._process_paper_data()

        # 将下载下来的论文进行解析，调用LLM

    def _process_paper_data(self,):
        parse_path = self.paper_parse_data_save_path
        paper_data_path = self.paper_parse_data_save_path  # os.path.join("data", today_date, "paper_data")

        for file_name in os.listdir(paper_data_path):
            if file_name.endswith(".txt"):
                title = os.path.splitext(file_name)[0]
                self.get_paper_qa(title, parse_path)

    @classmethod
    def read_json_file(cls, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error reading JSON file: {str(e)}")
            return None

    def save_paper_txt(self, paper_cls):
        # 读取 JSON 文件
        json_file_path = self.metainfo_save_path + f'/{paper_cls}.json'
        data = self.read_json_file(json_file_path)
        # 检查数据是否成功读取
        if data is not None:
            # 遍历数据并调用 download_paper() 函数
            for item in data:
                pdf_url = item['pdf_url']
                title = item['title']
                self.download_paper(pdf_url, title, self.paper_parse_data_save_path)

    def get_paper(self, paper_cls):
        """获取paper信息，然后进行存储，此时会自动下载meta信息，并将信息存储起来
        """
        # 得到meta信息的数组
        self.metainfo_of_paper = self.arxiv_client.get_paper_metadata(query=paper_cls)
        self._save_paper_metainfo_data(paper_cls)

    def _save_paper_metainfo_data(self, paper_cls):
        # 将metainfo组织好，保存到date下，还需要分类别保存
        self._save_json_data(self.metainfo_of_paper, self.metainfo_save_path, paper_cls + ".json")

    def _init_save_path(self,):
        # meta信息保存，这里不区分类别
        self.metainfo_save_path = f"data/{today_date}/meta"
        self.paper_parse_data_save_path = f"data/{today_date}/paper_data"

    @classmethod
    def download_paper(cls, url, title, doc_path):
        ensure_directory_exists(path=doc_path)
        pdf_path = f"{doc_path}/{title}.pdf"
        txt_path = f"{doc_path}/{title}.txt"
        print("sdfsdfsd", txt_path)
        if os.path.isfile(txt_path):
            # 已经有了就不要重新下载解析了
            print("it's done", url, title)
            return
        print("downloading", url, title)
        dta = requests.get(url)
        bytes_io = io.BytesIO(dta.content)
        with open(pdf_path, "wb") as f:
            f.write(bytes_io.getvalue())
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                f1 = codecs.open(txt_path, 'a', 'utf-8')
                f1.write(page.extract_text())
                f1.close()
        os.remove(pdf_path)

    @classmethod
    def get_paper_qa(cls, title="1", parse_path=""):
        question_prompt_dict = {
            "这篇论文试图解决什么问题？": "",
            "有哪些相关研究？": "",
            "论文如何解决这个问题？": "",
            "论文做了哪些实验？": "",
            "有什么可以进一步探索的点？": "",
            "总结一下论文的主要内容": "",
            "给这个论文提一些你的意见": ""
        }

        result_dict = {}
        for question in question_prompt_dict.keys():
            with open(f"{parse_path}/{title}.txt", 'r', encoding='utf-8') as f:
                paper = f.read(2000)

            prompt = f"""
                你是一名经验丰富的自然语言处理与计算机专业学者，请根据给定论文回答相应问题
                论文:{paper}
                问题:{question}
                """
            print(prompt)
            res = glm.llm(prompt)
            print(res)

            result_dict[question] = res

        # 将结果保存到文件
        result_file_path = os.path.join(parse_path, f"{title}_qa.json")
        with open(result_file_path, 'w') as f:
            json.dump(result_dict, f)

        print(f"QA results for {title} saved to {result_file_path}")

    @classmethod
    def _save_json_data(cls, json_data, path, filename):
        # 检查文件路径
        ensure_directory_exists(path)
        path_filename = path + "/" + filename
        try:
            with open(path_filename, 'w') as file:
                json.dump(json_data, file, indent=4)
            print(f"JSON data saved to file: {path_filename}")
        except Exception as e:
            print(f"Error saving JSON data to file: {str(e)}")

    @classmethod
    def _save_data(cls, data, path):
        # 检查文件路径
        ensure_directory_exists(path)
        # 保存数据到文件
        try:
            with open(path, 'w') as file:
                file.write(data)
            print(f"Data saved to file: {path}")
        except Exception as e:
            print(f"Error saving data to file: {str(e)}")


def main():
    """调用函数，实现数据自动更新
    1. 通过论文获取接口拿到最新的论文
    2. 通过LLM对论文进行解析，拿到解析数据结果
    3. 将上述内容分别组织
        3.1 论文meta信息每天都保存下来，放在data文件夹下的paper中，要每天都保存一个文件夹，以日期命名，里面保存前端需要用到的meta信息
        3.2 论文解析数据也需要每天都保存下来，在上述日期文件夹下单开一个文件夹，按照论文分类，分别创建文件夹，仅保存解析结果以减少数据量
    4. 线上前端界面每次刷新会从data文件夹中
    """
    paper_parser = PaperParser()
    paper_parser.parse_paper()


if __name__ == '__main__':
    main()
