'''
Author: ljyduke 叶兀
Date: 2024-01-03 22:51:43
LastEditors: Duke 叶兀
LastEditTime: 2024-01-08 23:32:00
FilePath: /paper_tutor/main.py
Description: 

Copyright (c) 2024 by ${ljyduke@gmail.com}, All Rights Reserved. 
'''
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder='app/templates')


# Simulated list of papers
papers = [
    {
        'title': 'Paper 1',
        'authors': 'Author 1',
        'links': 'http://example.com',
        'entry_id': '1',
        'pdf_url': 'http://example.com/paper1.pdf',
        'summary': 'Summary 1',
        'updated': '2022-01-01'
    },
    {
        'title': 'Paper 2',
        'authors': 'Author 2',
        'links': 'http://example.com',
        'entry_id': '2',
        'pdf_url': 'http://example.com/paper2.pdf',
        'summary': 'Summary 2',
        'updated': '2022-01-02'
    },
    # Add more papers here
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/papers')
def get_papers():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    # results = arxiv_client.get_paper_metadata()
    # if results:
    #     papers = results

    start_index = (page - 1) * per_page
    end_index = start_index + per_page

    papers_subset = papers[start_index:end_index]

    return jsonify(papers_subset)


@app.route('/parse_paper')
def get_parse_paper():
    paper_id = request.args.get('id')  # 获取论文的ID
    print(paper_id)
    # parse = papers.get(paper_id, {}).get('parse')  # 根据ID获取论文的解析内容

    return str(paper_id)


@app.route('/total_pages')
def get_total_pages():
    per_page = int(request.args.get('per_page', 10))
    total_pages = (len(papers) + per_page - 1) // per_page

    return str(total_pages)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
