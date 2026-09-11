from datetime import datetime, timedelta
from typing import List, Set, Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import or_

from src.config import settings
from src.models import User, Document, DocumentACL

# Password Hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None

def filter_documents_by_acl(db: Session, doc_ids: List[int], user_id: int, user_role: str) -> Set[int]:
    """
    Given a list of document IDs and a user's details, filter and return
    only the subset of document IDs that the user is authorized to read.
    """
    if not doc_ids:
        return set()

    # Admin bypass: Admins have read access to all documents
    if user_role == "admin":
        return set(doc_ids)

    # Get group IDs for user
    user = db.query(User).filter(User.id == user_id).first()
    group_ids = [g.id for g in user.groups] if user else []

    # Query matching documents based on ACL rules
    acl_conditions = [
        # Match user explicit permissions
        (DocumentACL.accessor_id == user_id) & (DocumentACL.accessor_type == "user") & (DocumentACL.permission == "read")
    ]
    if group_ids:
        # Match user group permissions
        acl_conditions.append(
            (DocumentACL.accessor_id.in_(group_ids)) & (DocumentACL.accessor_type == "group") & (DocumentACL.permission == "read")
        )

    query = db.query(Document.id).outerjoin(DocumentACL).filter(
        Document.id.in_(doc_ids),
        or_(
            Document.owner_id == user_id,
            *acl_conditions
        )
    )

    allowed_ids = {row[0] for row in query.all()}
    return allowed_ids

def check_document_access(db: Session, doc_id: int, user_id: int, user_role: str, required_permission: str = "read") -> bool:
    """Check if a specific user has read/write access to a single document."""
    if user_role == "admin":
        return True

    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return False

    if doc.owner_id == user_id:
        return True

    user = db.query(User).filter(User.id == user_id).first()
    group_ids = [g.id for g in user.groups] if user else []

    acl_conditions = [
        (DocumentACL.accessor_id == user_id) & (DocumentACL.accessor_type == "user") & (DocumentACL.permission == required_permission)
    ]
    if group_ids:
        acl_conditions.append(
            (DocumentACL.accessor_id.in_(group_ids)) & (DocumentACL.accessor_type == "group") & (DocumentACL.permission == required_permission)
        )

    acl_exists = db.query(DocumentACL).filter(
        DocumentACL.document_id == doc_id,
        or_(*acl_conditions)
    ).first()

    return acl_exists is not None
