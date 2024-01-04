
import arxiv
import json


class ArxivClient:
    def __init__(self) -> None:
        # Construct the default API client.
        self.client = arxiv.Client()
        # Search for the 10 most recent articles matching the keyword "quantum."
        self.search = arxiv.Search(
            query="cs.CL",
            max_results=10,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
    def get_paper_metadata(self, offset=0):
        results = self.client.results(self.search, offset)
        papers = []
        # `results` is a generator; you can iterate over its elements one by one...
        for idx, r in enumerate(results):
            paper = {
            'title': r.title,
            'authors': ''.join([i.name for i in r.authors]),
            'links': r.links[0].href,
            'entry_id': r.entry_id,
            'pdf_url': r.pdf_url,
            'summary': r.summary,
            'updated': r.updated.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'id': idx + 1
            }
            papers.append(paper)
        print(json.dumps(papers, indent=4))
        return papers
    
arxiv_client = ArxivClient()


if __name__ == '__main__':
    # arxiv_client.get_paper_metadata()
    import pdfplumber
    import codecs
    import requests
    import io
    url = "https://arxiv.org/pdf/2401.01830v1.pdf"
    dta = requests.get(url)
    bytes_io = io.BytesIO(dta.content)
    with open("data/ttt.pdf", "wb") as f:
        f.write(bytes_io.getvalue())
    
    with pdfplumber.open("data/ttt.pdf") as pdf:
        for page in pdf.pages:
            f1=codecs.open('data/ttt.txt','a','utf-8')
            f1.write(page.extract_text())
            f1.close()