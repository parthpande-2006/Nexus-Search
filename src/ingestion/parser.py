import os
import json
import hashlib

class DocumentParser:
    @staticmethod
    def calculate_hash(content: str) -> str:
        """Calculate MD5 hash of string content to detect duplicates."""
        return hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()

    @classmethod
    def parse_file(cls, filepath: str) -> dict:
        """Parse a local file and return document data including content, title, and file extension."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        content = ""
        metadata = {
            "title": filename,
            "source_url": filepath,
            "file_type": ext.replace(".", "")
        }

        try:
            if ext in [".txt", ".md", ".py", ".js", ".json", ".csv", ".html", ".xml"]:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    if ext == ".json":
                        try:
                            data = json.load(f)
                            content = json.dumps(data, indent=2)
                        except json.JSONDecodeError:
                            f.seek(0)
                            content = f.read()
                    else:
                        content = f.read()
            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(filepath)
                    text_parts = []
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text_parts.append(t)
                    content = "\n".join(text_parts)
                except ImportError:
                    content = f"[PDF File: Extraction requires pypdf library. File: {filename}]"
            else:
                content = f"[Binary File: {filename} of type {ext}]"
        except Exception as e:
            content = f"[Error parsing file {filename}: {str(e)}]"

        return {
            "title": filename,
            "content": content,
            "hash": cls.calculate_hash(content),
            "metadata": metadata
        }
