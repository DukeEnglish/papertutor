from flask import Flask, jsonify, render_template, request
from paper_related.paper_arxiv import arxiv_client

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
    results = arxiv_client.get_paper_metadata()
    if results:
        papers = results

    start_index = (page - 1) * per_page
    end_index = start_index + per_page

    papers_subset = papers[start_index:end_index]

    return jsonify(papers_subset)

@app.route('/total_pages')
def get_total_pages():
    per_page = int(request.args.get('per_page', 10))
    total_pages = (len(papers) + per_page - 1) // per_page

    return str(total_pages)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
