from langchain.document_loaders.csv_loader import CSVLoader

import re

loader = CSVLoader(file_path='./data/jesse-brag-doc.csv')

data = loader.load()
docs = [dict(doc) for doc in data]

for item in data:
    print(item)

def get_rows(docs, rows):
    return [doc['page_content'] for doc in docs if doc['metadata']['row'] in rows]

def clean_text(docs):
    return [re.sub(r'IC L2 Template\nPerf. ', '', doc) for doc in docs]

def fetch_data(categories, prompts):
    pass
    
category_rows = {1, 9, 18, 25}
categories = get_rows(docs, category_rows)
cleaned_categories = clean_text(categories)

prompt_rows = {3, 4, 5, 11, 12, 13, 14, 15, 20, 21, 22, 27, 28, 29}
prompts = get_rows(docs, prompt_rows)
cleaned_prompts = clean_text(prompts)
