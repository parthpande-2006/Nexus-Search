import os
import requests
import base64
from typing import Generator, Dict, Any, List
from src.ingestion.parser import DocumentParser

class LocalFileConnector:
    """Connector to scan a directory recursively for files."""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def scan(self) -> Generator[Dict[str, Any], None, None]:
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Source directory does not exist: {self.root_dir}")

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                filepath = os.path.join(root, file)
                if any(part.startswith('.') for part in filepath.split(os.sep)):
                    continue
                try:
                    doc_data = DocumentParser.parse_file(filepath)
                    doc_data["source_type"] = "local"
                    yield doc_data
                except Exception:
                    continue


class GitHubConnector:
    """Connector to scan files from a GitHub repository via GitHub REST API."""
    def __init__(self, repo_owner: str, repo_name: str, branch: str = "main", github_token: str = None):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.branch = branch
        self.headers = {}
        if github_token:
            self.headers["Authorization"] = f"token {github_token}"
        self.headers["Accept"] = "application/vnd.github.v3+json"

    def scan(self) -> Generator[Dict[str, Any], None, None]:
        url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/git/trees/{self.branch}?recursive=1"
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch GitHub repo tree: {response.text}")
            
            tree_data = response.json()
            tree = tree_data.get("tree", [])
            
            for item in tree:
                if item.get("type") == "blob":
                    path = item.get("path", "")
                    if any(part.startswith('.') for part in path.split('/')):
                        continue
                    
                    _, ext = os.path.splitext(path.lower())
                    if ext in [".png", ".jpg", ".jpeg", ".gif", ".ico", ".bin", ".zip", ".tar", ".gz"]:
                        continue

                    file_url = item.get("url")
                    file_response = requests.get(file_url, headers=self.headers, timeout=10)
                    if file_response.status_code == 200:
                        file_data = file_response.json()
                        encoding = file_data.get("encoding")
                        content_encoded = file_data.get("content", "")
                        
                        if encoding == "base64":
                            try:
                                content = base64.b64decode(content_encoded).decode("utf-8", errors="ignore")
                            except Exception:
                                content = "[Base64 Decoding Failed]"
                        else:
                            content = content_encoded

                        doc_hash = DocumentParser.calculate_hash(content)
                        source_url = f"https://github.com/DirectLinkMock/{self.repo_owner}/{self.repo_name}/blob/{self.branch}/{path}"
                        
                        yield {
                            "title": os.path.basename(path),
                            "content": content,
                            "hash": doc_hash,
                            "source_type": "github",
                            "metadata": {
                                "title": os.path.basename(path),
                                "path": path,
                                "source_url": source_url,
                                "file_type": ext.replace(".", "") if ext else "unknown"
                            }
                        }
        except Exception as e:
            print(f"Error scanning GitHub: {e}")
            return
