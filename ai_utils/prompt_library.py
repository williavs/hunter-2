import json
import os
import threading
from typing import List, Dict, Optional

PROMPT_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), '../static/prompt_library.json')
_lock = threading.Lock()

def _read_prompts() -> List[Dict]:
    with _lock:
        if not os.path.exists(PROMPT_LIBRARY_PATH):
            return []
        with open(PROMPT_LIBRARY_PATH, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return []

def _write_prompts(prompts: List[Dict]):
    with _lock:
        with open(PROMPT_LIBRARY_PATH, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

def list_prompts() -> List[Dict]:
    """Return all prompt templates."""
    return _read_prompts()

def get_prompt(prompt_id: str) -> Optional[Dict]:
    """Get a prompt by its id."""
    prompts = _read_prompts()
    for prompt in prompts:
        if prompt.get('id') == prompt_id:
            return prompt
    return None

def add_prompt(prompt: Dict) -> bool:
    """Add a new prompt. Returns True if added, False if id exists."""
    prompts = _read_prompts()
    if any(p.get('id') == prompt.get('id') for p in prompts):
        return False
    prompts.append(prompt)
    _write_prompts(prompts)
    return True

def update_prompt(prompt_id: str, new_prompt: Dict) -> bool:
    """Update an existing prompt by id. Returns True if updated, False if not found."""
    prompts = _read_prompts()
    for i, prompt in enumerate(prompts):
        if prompt.get('id') == prompt_id:
            prompts[i] = new_prompt
            _write_prompts(prompts)
            return True
    return False

def delete_prompt(prompt_id: str) -> bool:
    """Delete a prompt by id. Returns True if deleted, False if not found."""
    prompts = _read_prompts()
    new_prompts = [p for p in prompts if p.get('id') != prompt_id]
    if len(new_prompts) == len(prompts):
        return False
    _write_prompts(new_prompts)
    return True 