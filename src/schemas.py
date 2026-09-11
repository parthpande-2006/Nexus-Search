from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Auth & User Schemas ---
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Optional[str] = "user"  # "admin" or "user"

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[str] = None

# --- Group Schemas ---
class GroupCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    description: Optional[str] = None

class GroupResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]

    class Config:
        from_attributes = True

class UserGroupAssignment(BaseModel):
    user_id: int
    group_id: int

# --- Ingestion Request ---
class ACLRule(BaseModel):
    accessor_id: int  # user_id or group_id
    accessor_type: str  # "user" or "group"
    permission: Optional[str] = "read"  # "read" or "write"

class IngestRequest(BaseModel):
    type: str  # "local" or "github"
    path: Optional[str] = None  # for local scan
    repo_owner: Optional[str] = None  # for github
    repo_name: Optional[str] = None  # for github
    branch: Optional[str] = "main"  # for github
    github_token: Optional[str] = None  # for github
    acls: List[ACLRule] = []

# --- Search Schemas ---
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    rerank: Optional[bool] = True

class SearchHit(BaseModel):
    doc_id: int
    title: str
    text: str
    score: Optional[float] = None
    rrf_score: Optional[float] = None
    relevance_score: Optional[float] = None

# --- Chat/RAG Schemas ---
class ChatRequest(BaseModel):
    query: str

class Citation(BaseModel):
    ref_id: int
    doc_id: int
    title: str
    snippet: str

class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
