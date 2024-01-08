'''
Descripttion: 
Author: Duke 叶兀
E-mail: ljyduke@gmail.com
Date: 2024-01-03 23:13:25
LastEditors: Duke 叶兀
LastEditTime: 2024-01-08 23:31:40
'''
import arxiv
import json


class ArxivClient:
    def __init__(self) -> None:
        # Construct the default API client.
        self.client = arxiv.Client()

    def get_paper_metadata(self, offset=0, max_results=3, query="cs.CL"):
        # Search for the 10 most recent articles matching the keyword "quantum."
        self.search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
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
            print(paper)
            papers.append(paper)
        print(json.dumps(papers, indent=4))

        return papers


if __name__ == '__main__':
    pass
    # arxiv_client.get_paper_qa()
    # arxiv_client._download_paper(url="http://arxiv.org/pdf/2401.02417v1",title="1")
