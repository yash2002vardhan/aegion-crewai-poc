# utils/todo_tools.py
from crewai.tools import tool
from typing import List
import json
import logging

logger = logging.getLogger("todo_tools")

# Global todo_manager instance (will be set from main.py)
_todo_manager = None

def initialize_todo_tools(todo_manager):
    """Initialize the todo tools with a TodoManager instance."""
    global _todo_manager
    _todo_manager = todo_manager


def _print_current_todo_list(user_id: str):
    """Helper function to print the current state of the to-do list."""
    tasks = _todo_manager.get_all_tasks(user_id)
    summary = _todo_manager.get_summary(user_id)

    print("\n" + "─"*80)
    print(f"📊 Current To-Do List Status: {summary['completed']}/{summary['total']} completed")
    print("─"*80)

    for idx, task in enumerate(tasks, 1):
        status_icons = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "failed": "❌"
        }
        icon = status_icons.get(task['status'], "❓")
        print(f"  {idx}. {icon} [{task['status'].upper():12}] {task['description']}")

    print("─"*80 + "\n")


@tool("Create Todo List")
def create_todo_list(user_id: str, tasks: List[str]) -> str:
    """Creates a new to-do list for a user with a list of tasks.

    Each task should be a clear, actionable description.
    Tasks will be executed in the order provided (linear sequence).
    Returns the created to-do list with task IDs and metadata.

    Args:
        user_id: The user ID for whom the to-do list is being created
        tasks: List of task descriptions to add to the to-do list

    Returns:
        JSON string with the created to-do list
    """
    try:
        created_tasks = _todo_manager.create_list(user_id, tasks)

        # Print to console immediately (visible in CrewAI logs)
        print("\n" + "="*80)
        print(f"📋 PLANNER: Created To-Do List with {len(created_tasks)} tasks")
        print("="*80)
        for idx, task in enumerate(created_tasks, 1):
            print(f"  {idx}. ⏳ {task['description']}")
        print("="*80 + "\n")

        result = {
            "status": "success",
            "message": f"Created to-do list with {len(created_tasks)} tasks for user {user_id}",
            "tasks": created_tasks
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        error_msg = f"Failed to create to-do list: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}\n")
        return json.dumps({
            "status": "error",
            "message": error_msg
        })


@tool("Update Todo Status")
def update_todo_status(
    user_id: str,
    task_id: str,
    status: str,
    result: str = None,
    tools_used: List[str] = None,
    error: str = None
) -> str:
    """Updates the status of a specific task in the to-do list.

    Use this after executing a task to mark it as in_progress, completed, or failed.
    You can optionally provide the result of the task execution and which tools were used.

    Args:
        user_id: The user ID who owns the to-do list
        task_id: The UUID of the task to update
        status: New status: pending, in_progress, completed, or failed
        result: Optional result/output of the task execution
        tools_used: List of tools used to complete the task
        error: Error message if the task failed

    Returns:
        JSON string with the updated task
    """
    try:
        updated_task = _todo_manager.update_status(
            user_id=user_id,
            task_id=task_id,
            status=status,
            result=result,
            tools_used=tools_used,
            error=error
        )

        # Print status update to console
        status_icons = {
            "pending": "⏳",
            "in_progress": "🔄",
            "completed": "✅",
            "failed": "❌"
        }
        icon = status_icons.get(status, "❓")

        print(f"\n{icon} EXECUTOR: Task '{updated_task['description'][:60]}...' → {status.upper()}")

        if status == "completed" and result:
            result_preview = result[:150] + "..." if len(result) > 150 else result
            print(f"   └─ Result: {result_preview}")

        if status == "failed" and error:
            print(f"   └─ Error: {error}")

        # Print current list state after update
        _print_current_todo_list(user_id)

        response = {
            "status": "success",
            "message": f"Task '{updated_task['description']}' updated to status: {status}",
            "task": updated_task
        }

        return json.dumps(response, indent=2)
    except Exception as e:
        error_msg = f"Failed to update task status: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}\n")
        return json.dumps({
            "status": "error",
            "message": error_msg
        })


@tool("Get Todo List")
def get_todo_list(user_id: str) -> str:
    """Retrieves the current to-do list for a user, showing all tasks with their statuses.

    Useful for checking what tasks are pending, in progress, or completed.
    Returns tasks in sequential order.

    Args:
        user_id: The user ID whose to-do list to retrieve

    Returns:
        JSON string with all tasks and summary
    """
    try:
        tasks = _todo_manager.get_all_tasks(user_id)
        summary = _todo_manager.get_summary(user_id)

        result = {
            "status": "success",
            "user_id": user_id,
            "summary": summary,
            "tasks": tasks
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to retrieve to-do list: {str(e)}"
        })


@tool("Get Next Pending Task")
def get_next_pending_task(user_id: str) -> str:
    """Retrieves the next pending task in the to-do list (first task that hasn't started yet).

    Use this to get the next task to execute in the sequence.
    Returns None if all tasks are complete.

    Args:
        user_id: The user ID whose next task to retrieve

    Returns:
        JSON string with the next pending task or None if all complete
    """
    try:
        next_task = _todo_manager.get_next_pending(user_id)

        if next_task:
            print(f"\n▶️  EXECUTOR: Getting next task → '{next_task['description']}'")
            result = {
                "status": "success",
                "message": f"Next task: {next_task['description']}",
                "task": next_task
            }
        else:
            print("\n🎉 EXECUTOR: All tasks complete!")
            _print_current_todo_list(user_id)
            result = {
                "status": "success",
                "message": "No pending tasks - all tasks are complete!",
                "task": None
            }

        return json.dumps(result, indent=2)
    except Exception as e:
        error_msg = f"Failed to get next task: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}\n")
        return json.dumps({
            "status": "error",
            "message": error_msg
        })
