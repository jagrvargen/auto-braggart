from typing import Any, Dict, List, Optional

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader


class GoogleSheetsAuth:
    def __init__(self):
        self.scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        self.creds = ServiceAccountCredentials.from_json_keyfile_name('./data/auto-braggart-9603339e3c33.json', self.scope)
        self.client = gspread.authorize(self.creds)


class BragDocLoader(BaseLoader, GoogleSheetsAuth):
    """Loads the question prompts from the 
       google sheets doc and stores them in
       Document objects.
    """
    
    def __init__(self, filepath: str = 'Formatted Brag Doc - Jesse') -> None:
        super().__init__()
        self.filepath = filepath

    def _is_title(self, text):
        return not text.startswith("You")

    def load(self) -> List[Document]:
        docs = []

        sheet = self.client.open(self.filepath)
        worksheet = sheet.get_worksheet(1)
        data = worksheet.get_all_records()

        for index, row in enumerate(data, start=1):
            if not self._is_title(row['Accomplishments']):
                docs.append(Document(page_content=row['Accomplishments'], metadata={'row': index+1}))

        return docs


class BragDocWriter(GoogleSheetsAuth):
    """Writes the professional brag to the appropriate row in the brag document."""

    def __init__(self, filepath: str = "Formatted Brag Doc - Jesse") -> None:
        super().__init__()
        self.filepath = filepath

    def write(self, new_brag: Document) -> None:
        sheet = self.client.open('Formatted Brag Doc - Jesse')
        worksheet = sheet.get_worksheet(1)
        data = worksheet.get_all_records()

        for index, row in enumerate(data, start=1):
            if new_brag.metadata['row'] == index + 1:  # define the is_title function to determine whether the row is a title
                existing_answer = row['Notes']
                # Append your text
                new_answer = existing_answer + new_brag.page_content
                print(f"\n Writing the accomplishment {new_answer}\nTo the category: {row['Accomplishments']}")

                # Write back the new answer to the second column (Google Sheets is 1-indexed)
                worksheet.update_cell(index + 1, 2, new_answer)
                break
