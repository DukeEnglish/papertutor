import json
import os
import re
from datetime import datetime


def read_papers_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def generate_html_for_papers(papers, title_id):
    html = '''
    <section id="{}">
        <h2>{}</h2>
        <ul>
    '''.format(title_id, title_id)

    for paper in papers:
        summary = paper.get("summary", "").replace("\n", "")
        summary = re.sub(r'[^\w\s:/.\-]', '', summary)
        # 获取interpretation字段
        interpretation = paper.get('interpretation', {})
        interpretation_str = ''  # 初始化一个空字符串来构建interpretation的字符串表示

        # 如果interpretation是字典，将其转换为字符串
        if isinstance(interpretation, dict):
            interpretation_str = interpretation_str = '<br>'.join([f'<strong>{key}</strong><br>: {value}' for key, value in interpretation.items()])

        # 添加按钮和 Markdown 解读内容
        html += '''
            <li>
                <h3>{}</h3>
                <p>Authors: {}</p>
                <p><a href="{}">Link to paper</a></p>
                <p>{}</p>
                <p>Last Updated: {}</p>
                <button class="interpret-button" data-id="{}">Interpret</button>
                <div id="interpretation-{}" class="interpretation" style="display:none;">
                    <p>Interpretation: <br>{}</p>
                </div>
            </li>
        '''.format(paper['title'], paper['authors'], paper['links'], summary, paper['updated'], paper['id'], paper['id'], interpretation_str)

    html += '''
        </ul>
    </section>
    '''

    return html


def generate_html_from_files(json_files_path):
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Research Papers</title>
        <link rel="stylesheet" href="style.css">
        <script src="script.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    </head>
    <body>
        <div id="sidebar">
            <h3>Papers by Category:</h3>
            <ul id="categories">
    '''

    # 遍历meta文件夹中的所有JSON文件
    for filename in os.listdir(json_files_path):
        if filename.endswith('.json'):
            # 读取JSON文件中的论文数据
            papers = read_papers_from_json(os.path.join(json_files_path, filename))
            # 生成导航栏链接
            title_id = filename.replace('.json', '')
            html += '''
                <li><a href="#{}">{}</a></li>
            '''.format(title_id, filename.replace('.json', ''))

    html += '''
            </ul>
            <li>
                    <a href="https://github.com/DukeEnglish/papertutor" target="_blank" rel="noopener noreferrer">
                        <span class="fab fa-github"></span> GitHub
                    </a>
            </li>
        </div>
        <div id="content">
    '''

    # 遍历所有小标题，并生成内容
    for filename in os.listdir(json_files_path):
        if filename.endswith('.json'):
            # 读取JSON文件中的论文数据
            papers = read_papers_from_json(os.path.join(json_files_path, filename))
            # 生成HTML内容
            html += generate_html_for_papers(papers, filename.replace('.json', ''))

    current_time = datetime.now().strftime('%Y-%m-%d')
    html += '''
        <div id="last-updated">
            <p>Updated Time: {}</p>
        </div>
    '''.format(current_time)
    html += '''
        </div>
    </body>
    </html>
    '''

    return html


def main():
    # 假设你的JSON文件位于名为'data/20240117/meta'的文件夹中
    json_files_path = 'data/20240123/meta'

    # 生成HTML内容
    html_content = generate_html_from_files(json_files_path)

    # 将HTML内容保存到文件
    with open('research_papers.html', 'w') as file:
        file.write(html_content)


if __name__ == '__main__':
    main()
