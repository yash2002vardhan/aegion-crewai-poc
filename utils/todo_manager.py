# utils/todo_manager.py
from typing import List, Dict, Optional
from datetime import datetime, timezone
import uuid
import logging

logger = logging.getLogger("todo_manager")


class TodoManager:
    """
    Manages a dynamic to-do list that can be created and updated by agents.
    Supports linear task execution with status tracking.
    """

    def __init__(self):
        # Storage: user_id -> list of tasks
        self.todo_lists: Dict[str, List[Dict]] = {}

    def create_list(self, user_id: str, tasks: List[str]) -> List[Dict]:
        """
        Create a new to-do list for a user.

        Args:
            user_id: Unique identifier for the user
            tasks: List of task descriptions (strings)

        Returns:
            List of task objects with metadata
        """
        task_objects = []
        for idx, description in enumerate(tasks):
            task = {
                "id": str(uuid.uuid4()),
                "index": idx,
                "description": description,
                "status": "pending",
                "result": None,
                "tools_used": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "started_at": None,
                "completed_at": None,
                "error": None
            }
            task_objects.append(task)

        self.todo_lists[user_id] = task_objects

        # Log the created to-do list
        logger.info(f"\n{'='*70}")
        logger.info(f"📋 TO-DO LIST CREATED for user: {user_id}")
        logger.info(f"{'='*70}")
        for idx, task in enumerate(task_objects, 1):
            logger.info(f"  {idx}. [{task['status'].upper()}] {task['description']}")
        logger.info(f"{'='*70}\n")

        return task_objects

    def add_task(self, user_id: str, description: str, index: Optional[int] = None) -> Dict:
        """
        Add a single task to existing list.

        Args:
            user_id: User identifier
            description: Task description
            index: Optional position to insert task (default: append to end)

        Returns:
            Created task object
        """
        if user_id not in self.todo_lists:
            self.todo_lists[user_id] = []

        task = {
            "id": str(uuid.uuid4()),
            "index": index if index is not None else len(self.todo_lists[user_id]),
            "description": description,
            "status": "pending",
            "result": None,
            "tools_used": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None
        }

        if index is not None:
            self.todo_lists[user_id].insert(index, task)
            # Re-index subsequent tasks
            for i in range(index + 1, len(self.todo_lists[user_id])):
                self.todo_lists[user_id][i]["index"] = i
        else:
            self.todo_lists[user_id].append(task)

        return task

    def update_status(
        self,
        user_id: str,
        task_id: str,
        status: str,
        result: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        error: Optional[str] = None
    ) -> Dict:
        """
        Update task status and optionally result/error.

        Args:
            user_id: User identifier
            task_id: Task UUID
            status: New status (pending, in_progress, completed, failed)
            result: Optional result/output of task execution
            tools_used: List of tools used to complete the task
            error: Error message if task failed

        Returns:
            Updated task object
        """
        if user_id not in self.todo_lists:
            raise ValueError(f"No to-do list found for user {user_id}")

        task = self._find_task(user_id, task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found for user {user_id}")

        old_status = task["status"]
        task["status"] = status

        if status == "in_progress" and task["started_at"] is None:
            task["started_at"] = datetime.now(timezone.utc).isoformat()

        if status in ["completed", "failed"]:
            task["completed_at"] = datetime.now(timezone.utc).isoformat()

        if result is not None:
            task["result"] = result

        if tools_used is not None:
            task["tools_used"] = tools_used

        if error is not None:
            task["error"] = error

        # Log the status update with current list state
        self._log_current_state(user_id, f"Task updated: {old_status} → {status}")

        return task

    def get_all_tasks(self, user_id: str) -> List[Dict]:
        """
        Get all tasks for a user.

        Args:
            user_id: User identifier

        Returns:
            List of all task objects (sorted by index)
        """
        if user_id not in self.todo_lists:
            return []

        return sorted(self.todo_lists[user_id], key=lambda x: x["index"])

    def get_next_pending(self, user_id: str) -> Optional[Dict]:
        """
        Get the next pending task (first one in sequence).

        Args:
            user_id: User identifier

        Returns:
            Next pending task object or None if all complete
        """
        tasks = self.get_all_tasks(user_id)
        for task in tasks:
            if task["status"] == "pending":
                return task
        return None

    def get_task_by_id(self, user_id: str, task_id: str) -> Optional[Dict]:
        """
        Get a specific task by ID.

        Args:
            user_id: User identifier
            task_id: Task UUID

        Returns:
            Task object or None
        """
        return self._find_task(user_id, task_id)

    def clear_list(self, user_id: str) -> None:
        """
        Clear all tasks for a user.

        Args:
            user_id: User identifier
        """
        if user_id in self.todo_lists:
            del self.todo_lists[user_id]

    def get_summary(self, user_id: str) -> Dict:
        """
        Get summary statistics for a user's to-do list.

        Args:
            user_id: User identifier

        Returns:
            Summary dict with counts by status
        """
        tasks = self.get_all_tasks(user_id)

        summary = {
            "total": len(tasks),
            "pending": sum(1 for t in tasks if t["status"] == "pending"),
            "in_progress": sum(1 for t in tasks if t["status"] == "in_progress"),
            "completed": sum(1 for t in tasks if t["status"] == "completed"),
            "failed": sum(1 for t in tasks if t["status"] == "failed"),
        }

        return summary

    def _find_task(self, user_id: str, task_id: str) -> Optional[Dict]:
        """Internal helper to find a task by ID."""
        if user_id not in self.todo_lists:
            return None

        for task in self.todo_lists[user_id]:
            if task["id"] == task_id:
                return task

        return None

    def _log_current_state(self, user_id: str, action: str = ""):
        """Log the current state of the to-do list."""
        tasks = self.get_all_tasks(user_id)
        if not tasks:
            return

        summary = self.get_summary(user_id)

        logger.info(f"\n{'='*70}")
        if action:
            logger.info(f"🔄 {action}")
        logger.info(f"📊 TO-DO LIST STATUS for user: {user_id}")
        logger.info(f"   Total: {summary['total']} | Pending: {summary['pending']} | "
                   f"In Progress: {summary['in_progress']} | Completed: {summary['completed']} | "
                   f"Failed: {summary['failed']}")
        logger.info(f"{'='*70}")

        for idx, task in enumerate(tasks, 1):
            status_icon = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(task['status'], "❓")

            logger.info(f"  {idx}. {status_icon} [{task['status'].upper()}] {task['description']}")

            if task['result'] and task['status'] == 'completed':
                result_preview = task['result'][:100] + "..." if len(task['result']) > 100 else task['result']
                logger.info(f"      └─ Result: {result_preview}")

            if task['error'] and task['status'] == 'failed':
                logger.info(f"      └─ Error: {task['error']}")

        logger.info(f"{'='*70}\n")
