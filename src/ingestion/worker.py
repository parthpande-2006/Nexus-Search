import json
import time
import traceback
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from src.database import SessionLocal
from src.models import Document, DocumentACL, User
from src.ingestion.connectors import LocalFileConnector, GitHubConnector
from src.indexing.faiss_index import FAISSIndexManager
from src.indexing.bm25_index import BM25IndexManager
from src.services.redis_cache import base_cache

# Queue Name
INGEST_QUEUE = "nexus_search:ingest_queue"

def queue_ingestion_job(job_data: Dict[str, Any], db: Session = None) -> bool:
    """Queue an ingestion job. Executes synchronously if Redis is in fallback mode."""
    if base_cache.use_in_memory_fallback:
        print("Redis is unavailable. Running ingestion job synchronously...")
        own_db = False
        if db is None:
            db = SessionLocal()
            own_db = True
        try:
            process_ingestion_job(job_data, db)
            return True
        except Exception as e:
            print(f"Synchronous ingestion failed: {e}")
            return False
        finally:
            if own_db:
                db.close()
    
    try:
        job_str = json.dumps(job_data)
        base_cache.client.rpush(INGEST_QUEUE, job_str)
        print("Successfully queued ingestion job in Redis.")
        return True
    except Exception as e:
        print(f"Failed to queue ingestion job: {e}")
        return False

def process_ingestion_job(job_data: Dict[str, Any], db: Session):
    """Core logic to process an ingestion job, write to DB, and index in FAISS + BM25."""
    job_type = job_data.get("type")
    owner_id = job_data.get("owner_id")
    acls_config = job_data.get("acls", [])

    owner = db.query(User).filter(User.id == owner_id).first()
    if not owner:
        raise ValueError(f"Owner user ID {owner_id} not found in database.")

    if job_type == "local":
        path = job_data.get("path")
        if not path:
            raise ValueError("Local ingestion job requires a 'path' parameter.")
        connector = LocalFileConnector(path)
    elif job_type == "github":
        repo_owner = job_data.get("repo_owner")
        repo_name = job_data.get("repo_name")
        branch = job_data.get("branch", "main")
        token = job_data.get("github_token")
        if not repo_owner or not repo_name:
            raise ValueError("GitHub ingestion job requires 'repo_owner' and 'repo_name'.")
        connector = GitHubConnector(repo_owner, repo_name, branch, github_token=token)
    else:
        raise ValueError(f"Unknown job type: {job_type}")

    print(f"Starting ingestion scan for type {job_type}...")
    vector_mgr = FAISSIndexManager()
    keyword_mgr = BM25IndexManager()

    success_count = 0
    duplicate_count = 0

    for doc_item in connector.scan():
        doc_hash = doc_item["hash"]
        
        existing_doc = db.query(Document).filter(Document.hash == doc_hash).first()
        if existing_doc:
            doc_id = existing_doc.id
            existing_doc.title = doc_item["title"]
            existing_doc.content = doc_item["content"]
            existing_doc.owner_id = owner_id
            existing_doc.source_url = doc_item["metadata"].get("source_url")
            
            db.query(DocumentACL).filter(DocumentACL.document_id == doc_id).delete()
            duplicate_count += 1
        else:
            existing_doc = Document(
                title=doc_item["title"],
                content=doc_item["content"],
                source_type=doc_item["source_type"],
                source_url=doc_item["metadata"].get("source_url"),
                owner_id=owner_id,
                hash=doc_hash
            )
            db.add(existing_doc)
            db.flush()
            doc_id = existing_doc.id
            success_count += 1

        for acl in acls_config:
            acl_entry = DocumentACL(
                document_id=doc_id,
                accessor_id=acl["accessor_id"],
                accessor_type=acl["accessor_type"],
                permission=acl.get("permission", "read")
            )
            db.add(acl_entry)

        db.commit()

        vector_mgr.add_document(doc_id, existing_doc.title, existing_doc.content)
        chunks = vector_mgr.chunk_text(existing_doc.content)
        keyword_mgr.add_document(doc_id, existing_doc.title, existing_doc.content, chunks)

    print(f"Ingestion completed. Added {success_count} new docs. Updated {duplicate_count} existing docs.")

def run_worker():
    """Infinite loop for background worker daemon checking Redis queue."""
    if base_cache.use_in_memory_fallback:
        print("Cannot start async background worker: Redis is unavailable.")
        return

    print("Background Ingestion Worker started. Listening to Redis queue...")
    while True:
        try:
            job = base_cache.client.brpop(INGEST_QUEUE, timeout=5)
            if job:
                _, job_str = job
                job_data = json.loads(job_str.decode("utf-8"))
                print(f"Worker received job: {job_data}")
                
                db = SessionLocal()
                try:
                    process_ingestion_job(job_data, db)
                except Exception as e:
                    print(f"Worker error processing job: {e}")
                    traceback.print_exc()
                finally:
                    db.close()
        except KeyboardInterrupt:
            print("Shutting down worker...")
            break
        except Exception as e:
            print(f"Worker queue polling error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    run_worker()
