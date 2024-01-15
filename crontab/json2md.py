'''
Descripttion: 
Author: Duke 叶兀
E-mail: ljyduke@gmail.com
Date: 2024-01-15 23:18:48
LastEditors: Duke 叶兀
LastEditTime: 2024-01-16 00:42:42
'''
import json

def json_to_markdown_table(data):
    table = ""
    
    for item in data:
        title = item.get("title", "")
        authors = item.get("authors", [])
        links = item.get("links", "")
        updated = item.get("updated", "")
        summary = item.get("summary", "")

        # Generate Markdown table rows for each paper
        row = "| Title: {}<br>Author: {}<br>Links: {}<br>Updated: {}<br>Summary: {} |\n".format(title, ", ".join(authors[:3]), links, updated, summary)
        table += row

    return table

def json_to_markdown_table(data):
    table = ""

    for item in data:

        table += "| Item |Content|\n"
        table += "| --- |---|\n"
        idx = item.get("id", "")
        title = item.get("title", "")
        authors = item.get("authors", [])
        links = item.get("links", "")
        updated = item.get("updated", "")
        summary = item.get("summary", "").replace("\n", "")

        # Generate Markdown table rows for each paper
        row = "|idx| {} |\n".format(idx)
        table += row

        row = "|title| {} |\n".format(title)
        table += row

        author_row = "|authors| {}\n".format(", ".join(authors.strip().split(",")[:3]))
        table += author_row

        links_row = "|links| {} |\n".format(links) 
        table += links_row
        updated_row = "|updated| {} |\n".format(updated)
        table += updated_row
        summary_row = "|summary| {} |\n".format(summary)
        table += summary_row
        table += "\n\n"

    return table

def json_to_markdown_table1(data):
    headers = ["ID", "Title", "Authors", "Links", "Summary", "Updated"]
    rows = []

    for item in data:
        id = str(item["id"])
        title = item["title"]
        authors = item["authors"]
        links = item["links"]
        summary = item["summary"]
        updated = item["updated"]

        row = [id, title, authors, links, summary, updated]
        rows.append(row)

    # Generate Markdown table
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for row in rows:
        table += "| " + " | ".join(row) + " |\n"

    return table

# Example JSON data
json_data = [
    {
        "title": "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism",
        "authors": "123",
        "links": "http://arxiv.org/abs/2401.02954v1",
        "entry_id": "http://arxiv.org/abs/2401.02954v1",
        "pdf_url": "http://arxiv.org/pdf/2401.02954v1",
        "summary": "as",
        "updated": "2024-01-05 18:59:13 UTC",
        "id": 1
    },
    {
        "title": "Towards ASR Robust Spoken Language Understanding Through In-Context Learning With Word Confusion Networks",
        "authors": "Kevin EversonYile GuHuck YangPrashanth Gurunath ShivakumarGuan-Ting LinJari KolehmainenIvan BulykoAnkur GandheShalini GhoshWael HamzaHung-yi LeeAriya RastrowAndreas Stolcke",
        "links": "http://arxiv.org/abs/2401.02921v1",
        "entry_id": "http://arxiv.org/abs/2401.02921v1",
        "pdf_url": "http://arxiv.org/pdf/2401.02921v1",
        "summary": "asd",
        "updated": "2024-01-05 17:58:10 UTC",
        "id": 2
    },
    {
        "title": "Introducing Bode: A Fine-Tuned Large Language Model for Portuguese Prompt-Based Task",
        "authors": "Gabriel Lino GarciaPedro Henrique PaiolaLuis Henrique MorelliGiovani CandidoArnaldo Cândido JúniorDanilo Samuel JodasLuis C. S. AfonsoIvan Rizzo GuilhermeBruno Elias PenteadoJoão Paulo Papa",
        "links": "http://arxiv.org/abs/2401.02909v1",
        "entry_id": "http://arxiv.org/abs/2401.02909v1",
        "pdf_url": "http://arxiv.org/pdf/2401.02909v1",
        "summary": "sdfsdf",
        "updated": "2024-01-05 17:15:01 UTC",
        "id": 3
    }
]
def json2md(json_file_path):

    # Read JSON data from file
    with open(json_file_path, "r") as file:
        json_data = json.load(file)

    # Convert JSON to Markdown table
    markdown_table = json_to_markdown_table(json_data)

    # Print the result
    return markdown_table


if __name__ == '__main__':

    # JSON file path
    json_file_path = "data/20240108/meta/cs.CL.json"
    print(json2md(json_file_path))