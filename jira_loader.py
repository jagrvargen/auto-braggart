import os

from typing import List

from dotenv import load_dotenv
from jira import JIRA

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

load_dotenv()

options = {
    'server': 'https://travelperk.atlassian.net'
}


class JiraAuth:
    def __init__(self) -> None:
        self.jira = JIRA(options, basic_auth=('jesse.hedden@travelperk.com', os.getenv('JIRA_API_TOKEN')))


class JiraLoader(BaseLoader, JiraAuth):
    def __init__(self, jql: str = 'assignee = currentUser() AND updated >= -1w') -> None:
        super().__init__()
        self.jql = jql

    def load(self) -> List[Document]:
        jira_docs = []
        issues = self.jira.search_issues(self.jql)
        for issue in issues:
            ticket_content = f"""
            Issue Key: {issue.key}
            Summary: {issue.fields.summary}
            Description: {issue.fields.description}
            Status: {issue.fields.status.name}
            """
            jira_docs.append(Document(page_content=ticket_content))
        print(f"JIRA DOCS ~~~> {jira_docs}")
        return jira_docs