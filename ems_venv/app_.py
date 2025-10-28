# --- models/model.py ---

from datetime import datetime
# Use Mapped and mapped_column for modern SQLAlchemy
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, func
from sqlalchemy.orm import relationship, Mapped, mapped_column, declarative_base
from typing import List, Optional

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    # role: Mapped[str] = mapped_column(String(20), nullable=False, default="user") # Example default
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    projects: Mapped[List["Project"]] = relationship("Project", back_populates="user")
    initialisations: Mapped[List["Initialisation"]] = relationship("Initialisation", back_populates="user") # Added relationship

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    project_name: Mapped[str] = mapped_column(String, nullable=False)
    file_name: Mapped[str] = mapped_column(String, nullable=False) # Or URL for git
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow) # Added created_at

    # --- NEW: Store the local path where the code is stored ---
    local_path: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="projects")
    initialisations: Mapped[List["Initialisation"]] = relationship("Initialisation", back_populates="project") # Added relationship

class Initialisation(Base):
    __tablename__ = "initialisations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    persona: Mapped[str] = mapped_column(String, nullable=False) # 'SDE' or 'PM'
    status: Mapped[str] = mapped_column(String, default="started") # e.g., started, processing, completed, failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # --- NEW: Fields for tracking analysis ---
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    results: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # Store final state JSON
    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True) # Store error details

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="initialisations")
    project: Mapped["Project"] = relationship("Project", back_populates="initialisations")
