from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from datetime import datetime
from src.database import Base

# Association table for User <-> Group Many-to-Many relationship
user_groups = Table(
    "user_groups",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("group_id", Integer, ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True)
)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="user")  # "admin", "user"
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    groups = relationship("Group", secondary=user_groups, back_populates="users")
    owned_documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")


class Group(Base):
    __tablename__ = "groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String, nullable=True)

    # Relationships
    users = relationship("User", secondary=user_groups, back_populates="groups")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source_type = Column(String, nullable=False)  # "local", "github"
    source_url = Column(String, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    hash = Column(String, unique=True, nullable=False)  # For duplicate detection
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    owner = relationship("User", back_populates="owned_documents")
    acls = relationship("DocumentACL", back_populates="document", cascade="all, delete-orphan")


class DocumentACL(Base):
    __tablename__ = "document_acls"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    accessor_id = Column(Integer, nullable=False)  # Represents user_id or group_id
    accessor_type = Column(String, nullable=False)  # "user" or "group"
    permission = Column(String, default="read")    # "read" or "write"

    # Relationships
    document = relationship("Document", back_populates="acls")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String, nullable=False)          # "query", "view", "ingest"
    query_string = Column(String, nullable=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    status = Column(String, nullable=False)          # "allowed", "denied"
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="audit_logs")
