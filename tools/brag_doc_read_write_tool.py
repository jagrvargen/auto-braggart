from langchain.docstore.document import Document
from langchain.tools import BaseTool, StructuredTool, Tool, YouTubeSearchTool, tool

from pydantic import BaseModel, Field
from typing import List, Type

from brag_doc_loader import BragDocLoader, BragDocWriter

class GoogleSheetsFilePath(BaseModel):
    """Input for CustomGoogleSheetsReaderTool."""

    filepath: str = Field(
        default="Formatted Brag Doc - Jesse",
        description="The name of the filepath used to open the Google Sheets doc.",
    )

class CustomGoogleSheetsReaderTool(BaseTool):
    name = "Google Sheets reader tool"
    description = "useful when you need to retrieve the engineer's competency framework question prompts/"
    args_schema: Type[BaseModel] = GoogleSheetsFilePath

    def _run(self, filepath: str) -> List[Document]:
        loader = BragDocLoader()
        return loader.load(filepath)
    
    async def _arun(self):
        raise NotImplementedError("Async not implemented")
    

class CustomGoogleSheetsWriterTool(BaseTool):
    name = "Google Sheets writer tool"
    description = "useful when you have a professional accomplishment you want to enter into the engineer's Google Sheets competency framework document"
    args_schema: Type[BaseModel] = Document

    def _run(self, new_brag: Document) -> None:
        writer = BragDocWriter()
        writer.write(new_brag)

    async def _arun(self):
        raise NotImplementedError("Async not implemented")
