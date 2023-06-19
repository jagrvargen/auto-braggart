from typing import Any
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma

class CompetencyFrameworkCategorizer(BaseTool):
    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)