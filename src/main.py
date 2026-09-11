import time
from fastapi import FastAPI, Depends, HTTPException, status, Response, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.config import settings
from src.database import get_db, engine, Base
from src.models import User, Group, Document, AuditLog
from src.schemas import (
    UserCreate, UserResponse, Token, GroupCreate, GroupResponse,
    UserGroupAssignment, IngestRequest, SearchRequest, SearchHit,
    ChatRequest, ChatResponse
)
from src.security import (
    get_password_hash, verify_password, create_access_token, decode_access_token
)
from src.indexing.search_engine import SearchEngine
from src.ingestion.worker import queue_ingestion_job
from src.services.redis_cache import base_cache
from src.utils.logger import (
    logger, HTTP_REQUESTS_TOTAL, QUERY_LATENCY, CACHE_ACCESS, DOCUMENTS_INGESTED
)

# Initialize Database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
search_engine = SearchEngine()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    if request.url.path != "/metrics":
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        logger.info(f"{request.method} {request.url.path} status={response.status_code} duration={duration:.4f}s")
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if not payload:
        raise credentials_exception
    username: str = payload.get("username")
    user_id: int = payload.get("user_id")
    if username is None or user_id is None:
        raise credentials_exception
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation restricted to system administrators."
        )
    return current_user

# --- Authentication Routes ---
@app.post("/api/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user_in.username).first():
        raise HTTPException(status_code=400, detail="Username already registered.")
    if db.query(User).filter(User.email == user_in.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")
    
    first_user = db.query(User).first() is None
    role = "admin" if first_user else user_in.role

    hashed_pw = get_password_hash(user_in.password)
    new_user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hashed_pw,
        role=role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/api/auth/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"username": user.username, "user_id": user.id, "role": user.role}
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# --- Group Management ---
@app.post("/api/groups", response_model=GroupResponse, status_code=status.HTTP_201_CREATED)
def create_group(group_in: GroupCreate, db: Session = Depends(get_db), current_user: User = Depends(get_admin_user)):
    if db.query(Group).filter(Group.name == group_in.name).first():
        raise HTTPException(status_code=400, detail="Group name already exists.")
    new_group = Group(name=group_in.name, description=group_in.description)
    db.add(new_group)
    db.commit()
    db.refresh(new_group)
    return new_group

@app.post("/api/groups/assign")
def assign_user_to_group(assignment: UserGroupAssignment, db: Session = Depends(get_db), current_user: User = Depends(get_admin_user)):
    user = db.query(User).filter(User.id == assignment.user_id).first()
    group = db.query(Group).filter(Group.id == assignment.group_id).first()
    if not user or not group:
        raise HTTPException(status_code=404, detail="User or Group not found.")
    
    if group in user.groups:
        return {"message": f"User {user.username} is already assigned to group {group.name}."}
        
    user.groups.append(group)
    db.commit()
    return {"message": f"Assigned user {user.username} to group {group.name}."}

# --- Document Ingestion ---
@app.post("/api/documents/ingest", status_code=status.HTTP_202_ACCEPTED)
def ingest_documents(req: IngestRequest, db: Session = Depends(get_db), current_user: User = Depends(get_admin_user)):
    acls_data = []
    for acl in req.acls:
        if acl.accessor_type == "user":
            acc_exists = db.query(User).filter(User.id == acl.accessor_id).first() is not None
        else:
            acc_exists = db.query(Group).filter(Group.id == acl.accessor_id).first() is not None
            
        if not acc_exists:
            raise HTTPException(
                status_code=400,
                detail=f"Accessor {acl.accessor_id} of type '{acl.accessor_type}' does not exist."
            )
        acls_data.append({
            "accessor_id": acl.accessor_id,
            "accessor_type": acl.accessor_type,
            "permission": acl.permission
        })

    job_payload = {
        "type": req.type,
        "path": req.path,
        "repo_owner": req.repo_owner,
        "repo_name": req.repo_name,
        "branch": req.branch,
        "github_token": req.github_token,
        "owner_id": current_user.id,
        "acls": acls_data
    }

    success = queue_ingestion_job(job_payload, db=db)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to schedule ingestion job.")
    
    DOCUMENTS_INGESTED.inc()
    return {"status": "Ingestion job queued successfully"}

# --- Secured Search & Chat/RAG ---
@app.post("/api/search", response_model=list[SearchHit])
def search_documents(req: SearchRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    cached_results = base_cache.get_search_results(current_user.id, req.query)
    if cached_results is not None:
        CACHE_ACCESS.labels(outcome="hit").inc()
        return cached_results

    CACHE_ACCESS.labels(outcome="miss").inc()
    
    start_time = time.time()
    results = search_engine.search(
        db,
        req.query,
        current_user.id,
        current_user.role,
        top_k=req.top_k,
        rerank=req.rerank
    )
    duration = time.time() - start_time
    QUERY_LATENCY.labels(operation="search").observe(duration)

    base_cache.set_search_results(current_user.id, req.query, results)
    return results

@app.post("/api/chat", response_model=ChatResponse)
def chat_rag_endpoint(req: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    cached_response = base_cache.get_chat_response(current_user.id, req.query)
    if cached_response is not None:
        CACHE_ACCESS.labels(outcome="hit").inc()
        return cached_response

    CACHE_ACCESS.labels(outcome="miss").inc()

    start_time = time.time()
    response = search_engine.chat_rag(db, req.query, current_user.id, current_user.role)
    duration = time.time() - start_time
    QUERY_LATENCY.labels(operation="chat_rag").observe(duration)

    base_cache.set_chat_response(current_user.id, req.query, response)
    return response

# --- Admin Operations (Audit & Info) ---
@app.get("/api/admin/documents", response_model=list)
def admin_list_documents(db: Session = Depends(get_db), current_user: User = Depends(get_admin_user)):
    docs = db.query(Document).all()
    result = []
    for d in docs:
        result.append({
            "id": d.id,
            "title": d.title,
            "source_type": d.source_type,
            "owner_username": d.owner.username,
            "hash": d.hash,
            "created_at": d.created_at,
            "acls": [{"accessor_id": a.accessor_id, "accessor_type": a.accessor_type, "permission": a.permission} for a in d.acls]
        })
    return result

@app.get("/api/admin/audit-logs", response_model=list)
def admin_get_audit_logs(db: Session = Depends(get_db), current_user: User = Depends(get_admin_user)):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(100).all()
    result = []
    for l in logs:
        result.append({
            "id": l.id,
            "user_username": l.user.username if l.user else "Deleted User",
            "action": l.action,
            "query_string": l.query_string,
            "document_id": l.document_id,
            "status": l.status,
            "timestamp": l.timestamp
        })
    return result
