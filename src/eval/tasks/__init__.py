"""Importing this package registers every native task module (Task 4).

Adding a benchmark: write tasks/your_task.py with a @register("name")
class, then add one import line below -- scripts/run_eval.py --list-tasks
and the runner pick it up automatically from there.
"""

from src.eval.tasks import alba, calame_pt, chatrag_hi, portugal_basic_qa, pt_culture

__all__ = ["alba", "calame_pt", "chatrag_hi", "portugal_basic_qa", "pt_culture"]
