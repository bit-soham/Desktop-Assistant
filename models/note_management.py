import os
import json
import torch
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer # type: ignore
import re
from .llm_interface import LLMInterface, USE_LOCAL_MODEL

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

class NoteManager:
    def __init__(self, notes_dir='user_data/notes', embeddings_dir='user_data/embeddings', embedding_model=None,
                 llm_interface=None, use_local_llm=USE_LOCAL_MODEL):
        self.notes_dir = notes_dir
        self.embeddings_dir = embeddings_dir
        self.embedding_model = embedding_model
        self.llm_interface = llm_interface
        self.use_local_llm = use_local_llm
        os.makedirs(notes_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)

    def load_notes_and_embeddings(self):
        """
        Load note text files from notes_dir and their embeddings from embeddings_dir.
        If an embedding file is missing, compute it and save as .pt.
        Returns: vault_content (list[str]), vault_titles (list[str]), vault_embeddings (torch.Tensor)
        """
        print("DEBUG: Loading notes and embeddings from folders...")
        vault_content = []
        vault_titles = []
        vault_embeddings_list = []

        for filename in sorted(os.listdir(self.notes_dir)):
            if not filename.endswith('.txt'):
                continue
            title = filename[:-4]
            note_path = os.path.join(self.notes_dir, filename)
            emb_path = os.path.join(self.embeddings_dir, f"{title}.pt")

            # Skip expired notes
            if self.is_expired(note_path):
                print(f"DEBUG: Deleting expired note: {title}")
                try:
                    os.remove(note_path)
                except OSError:
                    pass
                if os.path.exists(emb_path):
                    try:
                        os.remove(emb_path)
                    except OSError:
                        pass
                continue

            # Load note content (skip JSON metadata if present)
            with open(note_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('{"expiration":'):
                    json_end = content.find('\n')
                    try:
                        metadata = json.loads(content[:json_end])
                    except Exception:
                        metadata = {}
                    content = content[json_end+1:].strip()
                else:
                    metadata = {}

            # Load existing embedding or compute & save
            if os.path.exists(emb_path):
                try:
                    emb = torch.load(emb_path)
                    # ensure 2D (N x D)
                    if emb.dim() == 1:
                        emb = emb.unsqueeze(0)
                except Exception as e:
                    print(f"DEBUG: Failed to load embedding {emb_path}: {e}. Recomputing.")
                    emb_np = self.embedding_model.encode([content])
                    emb = torch.tensor(emb_np, dtype=torch.float32)
                    torch.save(emb, emb_path)
            else:
                emb_np = self.embedding_model.encode([content])
                emb = torch.tensor(emb_np, dtype=torch.float32)
                try:
                    torch.save(emb, emb_path)
                except Exception as e:
                    print(f"DEBUG: Warning: couldn't save embedding to {emb_path}: {e}")

            vault_titles.append(title)
            vault_content.append(content)
            vault_embeddings_list.append(emb)

        if vault_embeddings_list:
            vault_embeddings = torch.cat(vault_embeddings_list, dim=0)  # shape: (N, D)
        else:
            vault_embeddings = torch.tensor([])

        print(f"DEBUG: Loaded {len(vault_titles)} active notes and embeddings.")
        return vault_content, vault_titles, vault_embeddings

    def is_expired(self, note_path):
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('{"expiration":'):
                json_end = content.find('\n')
                metadata = json.loads(content[:json_end])
                if 'expiration' in metadata:
                    exp_time = datetime.fromisoformat(metadata['expiration'])
                    return datetime.now() > exp_time
        return False

    # Function to create or edit note
    def manage_note(self, action, title, text=None, seconds=None):
        note_path = os.path.join(self.notes_dir, f"{title}.txt")
        emb_path = os.path.join(self.embeddings_dir, f"{title}.pt")

        if action == "create" or action == "edit":
            if text is None:
                return False  # Need text for create/edit
            
            metadata = {}
            if seconds:
                exp_time = datetime.now() + timedelta(seconds=seconds)
                metadata['expiration'] = exp_time.isoformat()

            content = json.dumps(metadata) + '\n' + text if metadata else text

            with open(note_path, 'w' if action == "create" else 'a', encoding='utf-8') as f:
                f.write(content)

            # Compute and save embedding
            emb = torch.tensor(self.embedding_model.encode([text]))
            torch.save(emb, emb_path)
            print(f"DEBUG: {action.capitalize()}d note '{title}'.")
            return True

        elif action == "delete":
            success = False
            if os.path.exists(note_path):
                os.remove(note_path)
                success = True
            if os.path.exists(emb_path):
                os.remove(emb_path)
                success = True
            if success:
                print(f"DEBUG: Deleted note '{title}'.")
            return success

    # Function to list notes
    def list_notes(self):
        active_notes = []
        for filename in os.listdir(self.notes_dir):
            if filename.endswith('.txt'):
                title = filename[:-4]
                note_path = os.path.join(self.notes_dir, filename)
                if not self.is_expired(note_path):
                    active_notes.append(title)
        print(NEON_GREEN + "Active Notes: " + ", ".join(active_notes) + RESET_COLOR)

    def parse_duration_response(self, duration_input):
        """
        Parse user's duration response (e.g., "3 days", "1 month") and convert to seconds.
        If no unit, assume seconds. If invalid, return None (never expire).
        """
        print(f"DEBUG: Parsing duration response: {duration_input}")
        # Simple regex or keyword-based parsing
        match = re.match(r'(\d+)\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)?', duration_input.lower().strip())
        if not match:
            print("DEBUG: No valid number/unit found in duration input.")
            return None  # Never expire

        num = int(match.group(1))
        unit = match.group(2) or "seconds"  # Default to seconds if no unit

        # Conversion factors
        conversions = {
            'second': 1,
            'minute': 60,
            'hour': 3600,
            'day': 86400,
            'week': 604800,
            'month': 2629746,  # Approx 30.44 days
            'year': 31536000   # Approx 365 days
        }

        # Handle plural
        unit = unit.rstrip('s') if unit.endswith('s') else unit

        if unit in conversions:
            seconds = num * conversions[unit]
            print(f"DEBUG: Converted duration: {num} {unit}s = {seconds} seconds")
            return seconds
        else:
            print("DEBUG: Unknown unit; defaulting to never expire.")
            return None

    def parse_note_creation(self, content_after_command, llm_tokenizer=None, llm_model=None):
        """
        Parse note creation command using LLM (local or API based on configuration).
        
        Args:
            content_after_command (str): The text after "create note" command
            llm_tokenizer: Local model tokenizer (optional, for backward compatibility)
            llm_model: Local model (optional, for backward compatibility)
        
        Returns:
            dict: Parsed note with 'Title', 'Note', and 'Duration' keys
        """
        print(f"DEBUG: Parsing note creation input: {content_after_command[:50]}... (truncated)")
        
        # Initialize LLM interface if not already done
        if self.llm_interface is None:
            if self.use_local_llm and llm_tokenizer and llm_model:
                self.llm_interface = LLMInterface(
                    use_local=True,
                    llm_tokenizer=llm_tokenizer,
                    llm_model=llm_model
                )
            elif not self.use_local_llm:
                self.llm_interface = LLMInterface(use_local=False)
            else:
                raise ValueError("LLM interface not initialized and required models not provided")
        
        # Few-shot prompt examples for LLM
        few_shot_prompt = """
        You are a note parsing assistant. Given any input string return a JSON object with:
        - Title: A short 1-4 word title from the input if specified, else generate a descriptive one.
        - Note: The main content after removing title and duration (if present).
        - Duration: The duration in seconds if specified (e.g., " 30 days"), else None.
        Output requirements (CRITICAL):
        - Output **ONLY** a single valid JSON object (no surrounding text, no explanation, no backticks, no code fences).
        - Use ISO formatting for no special tokens. If you cannot determine duration, use null for Duration.
        - The Note should contain only the main content of the note remove the title and duration parts that is in the string like remove 'keep this note for x duration like in the first example' or remove the mention of the title from the note content.

        Examples:
        Input: "I am very happy today and had an amazing day keep this note for 3 months"
        Output: {"Title": "My amazing day", "Note": "I am very happy today and had an amazing day", "Duration": 7776000}

        Input: "I will try to buy groceries everyday for 2 hours"
        Output: {"Title": "grocery task", "Note": "I will try to buy groceries everyday for 2 hours", "Duration": null}

        Input: "plan my day everyday title daily plan note for 45 days"
        Output: {"Title": "daily plan", "Note": "plan my day everyday", "Duration": 3888000}
        """

        # Prepare the message for LLM
        messages = [
            {"role": "system", "content": few_shot_prompt},
            {"role": "user", "content": content_after_command}
        ]

        # Generate response using unified interface
        response = self.llm_interface.generate(messages, max_new_tokens=150)
        
        print(f"DEBUG: Raw LLM response: {response[:200]}...")
        # Get LLM response
        # response = client.chat.completions.create(
        #     model="llama3.2:1b",
        #     messages=messages,
        #     stream=False  # Non-streaming for simplicity in parsing
        # )
        


        print("[DEBUG] response before: ", response.strip())

        def _extract_first_json_object(text: str):
            """
            Find and return the first balanced JSON object substring from text.
            Returns the substring or None if not found.
            """
            start = None
            depth = 0
            for i, ch in enumerate(text):
                if ch == '{':
                    if start is None:
                        start = i
                    depth += 1
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start is not None:
                            return text[start:i+1]
            return None

        response = _extract_first_json_object(response)

        print("[DEBUG] response after: ", response.strip() if response else "None")
        try:
            if response is None:
                # No JSON object found in response
                raise ValueError("No JSON found in response")
            parsed_result = json.loads(response.strip())
            print(f"DEBUG: Parsed result: {parsed_result}")
            return parsed_result
        except (json.JSONDecodeError, ValueError) as e:
            print(f"DEBUG: Failed to parse LLM response as JSON: {e}")
            # Fallback: Generate default if LLM fails
            title = datetime.now().strftime("%Y%m%d_%H%M%S")
            return {"Title": title, "Note": content_after_command, "Duration": None}