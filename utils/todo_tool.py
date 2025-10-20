# utils/todo_tool.py
"""
TodoWrite Tool for CrewAI Agents
Mirrors Claude Code's TodoWrite functionality
"""
from crewai.tools import tool
from typing import List, Dict
from utils.todo_manager import TodoItem, get_todo_manager
import logging

logger = logging.getLogger("todo_tool")


@tool("Manage Task List")
def todo_write(todos: List[Dict[str, str]]) -> str:
    """Manage a dynamic task list to track progress through complex, multi-step operations.

    USE THIS TOOL WHEN:
    1. Complex multi-step tasks - When a task requires 3 or more distinct steps
    2. Non-trivial complex tasks - Tasks requiring careful planning or multiple operations
    3. User explicitly requests todo list tracking
    4. User provides multiple tasks (numbered or comma-separated)
    5. After receiving new instructions - Immediately capture requirements as todos
    6. When starting work - Mark it as in_progress BEFORE beginning
    7. After completing a task - Mark as completed and add any new follow-up tasks

    DO NOT USE WHEN:
    - Single, straightforward task
    - Trivial task (< 3 steps)
    - Purely conversational or informational queries

    CRITICAL RULES:
    - Exactly ONE task must be 'in_progress' at any time (not less, not more)
    - Complete current tasks before starting new ones
    - Mark tasks completed IMMEDIATELY after finishing (don't batch)
    - Only mark completed when FULLY accomplished
    - If blocked/errors occur, keep as in_progress and create new task for resolution

    TASK STRUCTURE:
    Each todo must have:
    - content: Imperative form (e.g., "Run tests", "Fix authentication bug")
    - activeForm: Present continuous (e.g., "Running tests", "Fixing authentication bug")
    - status: "pending" | "in_progress" | "completed"

    Args:
        todos: List of todo items, each with 'content', 'activeForm', and 'status' keys

    Returns:
        Confirmation message with current todo list summary

    Example:
        todos_data = [
            {"content": "Search knowledge base", "activeForm": "Searching knowledge base", "status": "in_progress"},
            {"content": "Send response to customer", "activeForm": "Sending response to customer", "status": "pending"}
        ]
        result = todo_write(todos=todos_data)
    """
    try:
        # Validate input
        if not isinstance(todos, list):
            return "Error: todos must be a list of dictionaries"

        # Parse and create TodoItem objects
        todo_items = []
        in_progress_count = 0

        for idx, todo_dict in enumerate(todos):
            # Validate required fields
            if not isinstance(todo_dict, dict):
                return f"Error: Todo at index {idx} must be a dictionary"

            if "content" not in todo_dict or "activeForm" not in todo_dict or "status" not in todo_dict:
                return f"Error: Todo at index {idx} missing required fields (content, activeForm, status)"

            # Validate status
            if todo_dict["status"] not in ["pending", "in_progress", "completed"]:
                return f"Error: Invalid status '{todo_dict['status']}' at index {idx}. Must be: pending, in_progress, or completed"

            # Count in_progress tasks
            if todo_dict["status"] == "in_progress":
                in_progress_count += 1

            # Create TodoItem
            todo_item = TodoItem(
                content=todo_dict["content"],
                activeForm=todo_dict["activeForm"],
                status=todo_dict["status"]
            )
            todo_items.append(todo_item)

        # Validate exactly ONE in_progress task
        if in_progress_count > 1:
            logger.warning(f"Multiple tasks in_progress ({in_progress_count}). Only ONE task should be in_progress at a time.")

        # Update todo manager
        todo_manager = get_todo_manager()
        state = todo_manager.create_or_update(
            todos=todo_items,
            user_id="agent",
            session_id="current"
        )

        # Build response message
        summary = todo_manager.get_summary()

        response_lines = [
            "✓ Todo list updated successfully",
            "",
            f"Progress: {summary['completed']}/{summary['total']} tasks completed ({summary['progress_percentage']}%)",
            "",
            "Current tasks:"
        ]

        for idx, todo in enumerate(summary['todos'], 1):
            status_icon = {
                "completed": "✓",
                "in_progress": "→",
                "pending": "○"
            }.get(todo['status'], "?")

            display_text = todo['display_text']
            response_lines.append(f"  {idx}. [{status_icon}] {display_text}")

        # Highlight current task
        if summary['in_progress'] > 0:
            current_tasks = [
                todo['activeForm']
                for todo in summary['todos']
                if todo['status'] == 'in_progress'
            ]
            if current_tasks:
                response_lines.append("")
                response_lines.append(f"Currently working on: {current_tasks[0]}")

        return "\n".join(response_lines)

    except Exception as e:
        error_msg = f"Error updating todo list: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool("Mark Todo In Progress")
def todo_mark_in_progress(todo_index: int) -> str:
    """Mark a specific todo item as in_progress.

    Automatically completes the current in_progress task before starting the new one.

    Args:
        todo_index: Zero-based index of the todo to mark as in_progress

    Returns:
        Confirmation message
    """
    try:
        todo_manager = get_todo_manager()
        success = todo_manager.mark_in_progress(todo_index)

        if success:
            summary = todo_manager.get_summary()
            current_task = summary['todos'][todo_index]
            return f"✓ Task started: {current_task['activeForm']}"
        else:
            return f"Error: Could not mark todo at index {todo_index} as in_progress"

    except Exception as e:
        return f"Error: {str(e)}"


@tool("Mark Todo Completed")
def todo_mark_completed(todo_index: int) -> str:
    """Mark a specific todo item as completed.

    Use this IMMEDIATELY after finishing a task (don't batch completions).

    Args:
        todo_index: Zero-based index of the todo to mark as completed

    Returns:
        Confirmation message
    """
    try:
        todo_manager = get_todo_manager()
        success = todo_manager.mark_completed(todo_index)

        if success:
            summary = todo_manager.get_summary()
            return f"✓ Task completed ({summary['completed']}/{summary['total']} done)"
        else:
            return f"Error: Could not mark todo at index {todo_index} as completed"

    except Exception as e:
        return f"Error: {str(e)}"


@tool("Add Todo Item")
def todo_add(content: str, active_form: str, status: str = "pending") -> str:
    """Dynamically add a new todo item to the list.

    Use this when you discover new tasks during implementation.

    Args:
        content: Imperative form description (e.g., "Fix error handling")
        active_form: Present continuous form (e.g., "Fixing error handling")
        status: Initial status (default: "pending")

    Returns:
        Confirmation message
    """
    try:
        if status not in ["pending", "in_progress", "completed"]:
            return f"Error: Invalid status '{status}'. Must be: pending, in_progress, or completed"

        todo_manager = get_todo_manager()
        success = todo_manager.add_todo(content, active_form, status)

        if success:
            summary = todo_manager.get_summary()
            return f"✓ Task added: {content} (Total: {summary['total']} tasks)"
        else:
            return "Error: Could not add todo item"

    except Exception as e:
        return f"Error: {str(e)}"


# Export tools for use in CrewAI agents
todo_write_tool = todo_write
todo_mark_in_progress_tool = todo_mark_in_progress
todo_mark_completed_tool = todo_mark_completed
todo_add_tool = todo_add


if __name__ == "__main__":
    # Example usage - test the underlying function directly
    from utils.todo_manager import get_todo_manager, TodoItem

    test_todos = [
        TodoItem(
            content="Search knowledge base for product information",
            activeForm="Searching knowledge base for product information",
            status="in_progress"
        ),
        TodoItem(
            content="Draft customer response",
            activeForm="Drafting customer response",
            status="pending"
        ),
        TodoItem(
            content="Send response via Telegram",
            activeForm="Sending response via Telegram",
            status="pending"
        )
    ]

    manager = get_todo_manager()
    state = manager.create_or_update(todos=test_todos, user_id="test_user")
    summary = manager.get_summary()

    print("Todo List Summary:")
    print(f"Total: {summary['total']}")
    print(f"Progress: {summary['completed']}/{summary['total']} ({summary['progress_percentage']}%)")
    print("\nTasks:")
    for idx, todo in enumerate(summary['todos']):
        status_icon = {"completed": "✓", "in_progress": "→", "pending": "○"}.get(todo['status'], "?")
        print(f"  {idx}. [{status_icon}] {todo['display_text']}")
