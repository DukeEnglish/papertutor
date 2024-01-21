'''
Descripttion: 
Author: Duke 叶兀
E-mail: ljyduke@gmail.com
Date: 2024-01-03 23:13:25
LastEditors: Duke 叶兀
LastEditTime: 2024-01-18 02:13:42
'''
import arxiv
import json


class ArxivClient:
    def __init__(self) -> None:
        # Construct the default API client.
        self.client = arxiv.Client()

    def get_paper_metadata(self, offset=0, max_results=5, query="cs.CL"):
        # Search for the 10 most recent articles matching the keyword "quantum."
        self.search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = self.client.results(self.search, offset)
        print("results from arxiv", results)
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
                'interpretation': 'example',
                'id': r.get_short_id()
            }
            print(paper)
            papers.append(paper)
        print("result", json.dumps(papers, indent=4))

        return papers


if __name__ == '__main__':
    arxiv_client = ArxivClient()
    arxiv_client.get_paper_metadata(query="cs.CL")
