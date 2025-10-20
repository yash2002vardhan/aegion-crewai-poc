# utils/todo_manager.py
"""
Todo List Manager - Implements Claude Code-style task tracking
Provides dynamic, real-time task management with state transitions
"""
import os
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
import threading
import logging
from utils.memory import QdrantMemoryWithMetadata

logger = logging.getLogger("todo_manager")

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(BaseModel):
    """Individual todo item with dual-form description"""
    content: str = Field(..., description="Imperative form: 'Run tests', 'Fix bug'")
    activeForm: str = Field(..., description="Present continuous: 'Running tests', 'Fixing bug'")
    status: TodoStatus = Field(default="pending", description="Current status")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ["pending", "in_progress", "completed"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v

    def mark_in_progress(self):
        """Transition to in_progress state"""
        self.status = "in_progress"
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_completed(self):
        """Transition to completed state"""
        self.status = "completed"
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_display_text(self) -> str:
        """Get appropriate text based on status"""
        return self.activeForm if self.status == "in_progress" else self.content


class TodoListState(BaseModel):
    """Complete todo list state"""
    todos: List[TodoItem] = Field(default_factory=list)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def get_in_progress_count(self) -> int:
        """Count tasks currently in progress"""
        return sum(1 for todo in self.todos if todo.status == "in_progress")

    def get_pending_count(self) -> int:
        """Count pending tasks"""
        return sum(1 for todo in self.todos if todo.status == "pending")

    def get_completed_count(self) -> int:
        """Count completed tasks"""
        return sum(1 for todo in self.todos if todo.status == "completed")

    def get_progress_percentage(self) -> float:
        """Calculate completion percentage"""
        total = len(self.todos)
        if total == 0:
            return 0.0
        completed = self.get_completed_count()
        return round((completed / total) * 100, 2)

    def validate_state(self) -> tuple[bool, Optional[str]]:
        """
        Validate todo list state according to Claude Code rules:
        - Exactly ONE task should be in_progress at any time
        """
        in_progress_count = self.get_in_progress_count()
        if in_progress_count > 1:
            return False, f"Multiple tasks in progress ({in_progress_count}). Only ONE task should be in_progress at a time."
        return True, None


class TodoManager:
    """
    Manages todo list state with Claude Code behavior:
    - Exactly ONE task in_progress at a time
    - Immediate status updates (no batching)
    - Dynamic task addition/removal
    - Optional persistence via Qdrant
    """

    def __init__(self, persist_to_qdrant: bool = True):
        self._state: Optional[TodoListState] = None
        self._lock = threading.Lock()
        self._persist = persist_to_qdrant
        self._qdrant_memory = None
        self._event_callbacks = []

        if self._persist:
            try:
                self._qdrant_memory = QdrantMemoryWithMetadata(
                    collection_name=os.getenv("QDRANT_TODO_COLLECTION", "todo_list_state"),
                    qdrant_url=os.getenv("QDRANT_URL"),
                    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
                    embedding_model="text-embedding-3-large"
                )
                logger.info("TodoManager initialized with Qdrant persistence")
            except Exception as e:
                logger.warning(f"Qdrant persistence disabled: {e}")
                self._persist = False

    def register_callback(self, callback):
        """Register callback for todo state changes"""
        self._event_callbacks.append(callback)

    def _notify_callbacks(self, event_type: str, data: dict):
        """Notify all registered callbacks"""
        for callback in self._event_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_state(self) -> Optional[TodoListState]:
        """Get current todo list state"""
        with self._lock:
            return self._state

    def create_or_update(self, todos: List[TodoItem], user_id: Optional[str] = None, session_id: Optional[str] = None) -> TodoListState:
        """
        Create or update the entire todo list.
        Validates that exactly ONE task is in_progress.
        """
        with self._lock:
            new_state = TodoListState(
                todos=todos,
                user_id=user_id,
                session_id=session_id
            )

            # Validate state
            is_valid, error_msg = new_state.validate_state()
            if not is_valid:
                logger.warning(f"Todo list validation warning: {error_msg}")

            self._state = new_state

            # Persist to Qdrant if enabled
            if self._persist and self._qdrant_memory:
                self._persist_state(new_state)

            # Notify callbacks
            self._notify_callbacks("todos_updated", {
                "todos": [todo.dict() for todo in todos],
                "user_id": user_id,
                "session_id": session_id
            })

            logger.info(f"Todo list updated: {len(todos)} items (Pending: {new_state.get_pending_count()}, In Progress: {new_state.get_in_progress_count()}, Completed: {new_state.get_completed_count()})")

            return new_state

    def mark_in_progress(self, todo_index: int) -> bool:
        """
        Mark a specific todo as in_progress.
        Automatically marks the current in_progress task as completed first.
        """
        with self._lock:
            if not self._state or todo_index >= len(self._state.todos):
                return False

            # Find and complete current in_progress task
            for todo in self._state.todos:
                if todo.status == "in_progress":
                    todo.mark_completed()
                    logger.info(f"Auto-completed previous task: {todo.content}")

            # Mark new task as in_progress
            todo = self._state.todos[todo_index]
            todo.mark_in_progress()

            # Persist
            if self._persist and self._qdrant_memory:
                self._persist_state(self._state)

            # Notify
            self._notify_callbacks("todo_started", {
                "index": todo_index,
                "todo": todo.dict()
            })

            logger.info(f"Task started: {todo.activeForm}")
            return True

    def mark_completed(self, todo_index: int) -> bool:
        """Mark a specific todo as completed"""
        with self._lock:
            if not self._state or todo_index >= len(self._state.todos):
                return False

            todo = self._state.todos[todo_index]
            if todo.status != "completed":
                todo.mark_completed()

                # Persist
                if self._persist and self._qdrant_memory:
                    self._persist_state(self._state)

                # Notify
                self._notify_callbacks("todo_completed", {
                    "index": todo_index,
                    "todo": todo.dict()
                })

                logger.info(f"Task completed: {todo.content}")
                return True
            return False

    def add_todo(self, content: str, activeForm: str, status: TodoStatus = "pending") -> bool:
        """Dynamically add a new todo to the list"""
        with self._lock:
            if not self._state:
                self._state = TodoListState(todos=[])

            new_todo = TodoItem(content=content, activeForm=activeForm, status=status)
            self._state.todos.append(new_todo)

            # Persist
            if self._persist and self._qdrant_memory:
                self._persist_state(self._state)

            # Notify
            self._notify_callbacks("todo_added", {
                "index": len(self._state.todos) - 1,
                "todo": new_todo.dict()
            })

            logger.info(f"Task added: {content}")
            return True

    def remove_todo(self, todo_index: int) -> bool:
        """Remove a todo from the list"""
        with self._lock:
            if not self._state or todo_index >= len(self._state.todos):
                return False

            removed_todo = self._state.todos.pop(todo_index)

            # Persist
            if self._persist and self._qdrant_memory:
                self._persist_state(self._state)

            # Notify
            self._notify_callbacks("todo_removed", {
                "index": todo_index,
                "todo": removed_todo.dict()
            })

            logger.info(f"Task removed: {removed_todo.content}")
            return True

    def clear(self) -> bool:
        """Clear all todos"""
        with self._lock:
            self._state = None

            # Notify
            self._notify_callbacks("todos_cleared", {})

            logger.info("Todo list cleared")
            return True

    def _persist_state(self, state: TodoListState):
        """Persist todo list state to Qdrant"""
        try:
            # Create a searchable text representation
            text_parts = []
            for todo in state.todos:
                text_parts.append(f"[{todo.status}] {todo.content}")

            text_representation = "\n".join(text_parts)

            metadata = {
                "user_id": state.user_id or "default",
                "session_id": state.session_id or "default",
                "todo_count": len(state.todos),
                "pending_count": state.get_pending_count(),
                "in_progress_count": state.get_in_progress_count(),
                "completed_count": state.get_completed_count(),
                "progress_percentage": state.get_progress_percentage(),
                "role": "todo_state",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "todos_json": [todo.dict() for todo in state.todos]
            }

            self._qdrant_memory.add(text=text_representation, metadata=metadata)
            logger.debug("Todo state persisted to Qdrant")

        except Exception as e:
            logger.error(f"Failed to persist todo state: {e}")

    def get_summary(self) -> Dict:
        """Get a summary of the current todo list state"""
        with self._lock:
            if not self._state:
                return {
                    "total": 0,
                    "pending": 0,
                    "in_progress": 0,
                    "completed": 0,
                    "progress_percentage": 0.0,
                    "todos": []
                }

            return {
                "total": len(self._state.todos),
                "pending": self._state.get_pending_count(),
                "in_progress": self._state.get_in_progress_count(),
                "completed": self._state.get_completed_count(),
                "progress_percentage": self._state.get_progress_percentage(),
                "todos": [
                    {
                        "content": todo.content,
                        "activeForm": todo.activeForm,
                        "status": todo.status,
                        "display_text": todo.get_display_text()
                    }
                    for todo in self._state.todos
                ]
            }


# Global singleton instance
_global_todo_manager: Optional[TodoManager] = None


def get_todo_manager() -> TodoManager:
    """Get or create the global TodoManager instance"""
    global _global_todo_manager
    if _global_todo_manager is None:
        _global_todo_manager = TodoManager(persist_to_qdrant=True)
    return _global_todo_manager
