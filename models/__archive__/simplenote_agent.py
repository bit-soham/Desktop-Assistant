import os
import time
import traceback
import simplenote
import sys
from typing import Callable, List
from openai import OpenAI

# ===================== Config =====================
EMAIL = "ntirth005@gmai.com"
PASSWORD = "simpleNote"
OLLAMA_URL = "http://localhost:11434/v1"
MODEL_NAME = "llama3.2:3b"

sn = simplenote.Simplenote(EMAIL, PASSWORD)

# List notes
notes = sn.get_note_list()

# Add note
sn.add_note({'content': 'My note content'})

print(notes)

if not EMAIL or not PASSWORD:
    raise RuntimeError("Simplenote credentials missing!")


client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

# ===================== Retry Decorator =====================
def retry(exceptions=(Exception,), tries=3, delay=1.0):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            _tries = tries
            while _tries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    _tries -= 1
                    if _tries == 0:
                        raise
                    print(f"Retrying after error: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

# ===================== Simplenote Client =====================
@retry()
def get_client():
    return simplenote.Simplenote(EMAIL, PASSWORD)

sn = get_client()

# ===================== Local LLM Wrapper =====================
class LocalOllamaChat:
    """Simple LLM wrapper using plain messages only."""

    def generate(self, messages: List[str]):
        """
        messages: list of strings (plain text)
        returns: string
        """
        chat_messages = [{"role": "user", "content": m} for m in messages]

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_messages,
            stream=True
        )

        collected = ""
        for chunk in stream:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and hasattr(delta, "content") and delta.content:
                print(delta.content, end="", flush=True)
                collected += delta.content
        print("\n--- Stream completed ---")
        return collected

llm = LocalOllamaChat()

# ===================== LLM Helper =====================
def llm_edit(content: str, instructions: str) -> str:
    prompt = f"""You are editing a markdown note.
Original:
{content}

Instructions:
{instructions}

Return only the updated content."""
    return llm.generate([prompt])

# ===================== Simplenote Functions =====================
@retry()
def add_note(input_text: str) -> str:
    parts = [p.strip() for p in input_text.split("|")]
    if len(parts) < 2:
        return "❌ Format: title | content"
    title, content = parts
    full = f"{title}\n{content}"
    success, _ = sn.add_note({"content": full})
    return "✅ Added" if success else "❌ Failed"

@retry()
def list_notes(_: str = "") -> str:
    success, notes = sn.get_note_list()
    if not success:
        return "❌ Error fetching notes"
    titles = [n.get("content", "").split("\n")[0] or "(Untitled)" for n in notes]
    return "\n".join(titles)

@retry()
def edit_note(input_text: str) -> str:
    if "|" not in input_text:
        return "❌ Format: keyword | instruction"
    keyword, instruction = [x.strip() for x in input_text.split("|", 1)]
    success, notes = sn.get_notes()
    if not success:
        return "❌ Could not fetch notes"
    for note in notes:
        content = note.get("content", "")
        if keyword.lower() in content.lower():
            new_content = llm_edit(content, instruction)
            sn.update_note({"key": note["key"], "content": new_content})
            return f"✅ Updated note containing '{keyword}'"
    return "❌ Note not found"

@retry()
def search_notes(input_text: str) -> str:
    keyword = input_text.strip().lower()
    success, notes = sn.get_notes()
    if not success:
        return "❌ Error fetching notes"

    results = []
    for note in notes:
        content = note.get("content", "")
        if keyword in content.lower():
            title = content.split("\n")[0]
            first_line = content.split("\n")[1] if len(content.split("\n")) > 1 else ""
            results.append(f"{title} — {first_line}")
    return "\n".join(results) if results else f"No notes found for '{input_text}'"

# ===================== Command Router =====================
def handle_command(user_input: str) -> str:
    cmd = user_input.strip().lower()
    if cmd.startswith("add|"):
        return add_note(user_input[4:])
    elif cmd.startswith("list"):
        return list_notes()
    elif cmd.startswith("edit|"):
        return edit_note(user_input[5:])
    elif cmd.startswith("search|"):
        return search_notes(user_input[7:])
    else:
        return "❌ Unknown command. Use add|, list, edit|, search|"

# ===================== Main Loop =====================
def main():
    print("✅ Simplenote LLM Agent Ready. Type 'exit' to quit.")
    while True:
        try:
            user = input("You: ")
            if user.lower() in ["exit", "quit"]:
                break
            response = handle_command(user)
            print("🤖", response)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit()
        except Exception:
            traceback.print_exc()

if __name__ == "__main__":
    main()
