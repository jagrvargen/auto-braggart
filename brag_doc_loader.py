from typing import Any, Dict, List, Optional

import gspread
from oauth2client.service_account import ServiceAccountCredentials
scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('./data/auto-braggart-9603339e3c33.json', scope)
client = gspread.authorize(creds)

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class BragDocLoader(BaseLoader):
    """Loads the question prompts from the 
       google sheets doc and stores them in
       Document objects.
    """
    
    def __init__(self, filepath) -> None:
        super().__init__()
        self.filepath = filepath

    def _is_title(self, text):
        return not text.startswith("You")

    def load(self) -> List[Document]:
        docs = []

        sheet = client.open(self.filepath)
        worksheet = sheet.get_worksheet(1)
        data = worksheet.get_all_records()

        for index, row in enumerate(data, start=1):
            if not self._is_title(row['Accomplishments']):
                docs.append(Document(page_content=row['Accomplishments'], metadata={'row': index}))

        return docs
