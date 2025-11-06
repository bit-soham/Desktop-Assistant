import os
import json
import torch
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer # type: ignore
import re

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

class NoteManager:
    def __init__(self, notes_dir='notes', embeddings_dir='embeddings', embedding_model=None):
        self.notes_dir = notes_dir
        self.embeddings_dir = embeddings_dir
        self.embedding_model = embedding_model
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
    def manage_note(self, action, title, text, seconds=None):
        note_path = os.path.join(self.notes_dir, f"{title}.txt")
        emb_path = os.path.join(self.embeddings_dir, f"{title}.pt")

        if action == "create" or action == "edit":
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

        elif action == "delete":
            if os.path.exists(note_path):
                os.remove(note_path)
            if os.path.exists(emb_path):
                os.remove(emb_path)
            print(f"DEBUG: Deleted note '{title}'.")

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

    def parse_note_creation(self, content_after_command, llm_tokenizer, llm_model):
        print(f"DEBUG: Parsing note creation input: {content_after_command[:50]}... (truncated)")
        # Few-shot prompt examples for LLM
        few_shot_prompt = """
        You are a note parsing assistant. Given any input string return a JSON object with:
        - Title: A short 1-4 word title from the input if specified, else generate a descriptive one.
        - Note: The main content after removing title and duration (if present).
        - Duration: The duration in seconds if specified (e.g., " 30 days"), else None.
        Output requirements (CRITICAL):
        - Output **ONLY** a single valid JSON object (no surrounding text, no explanation, no backticks, no code fences).
        - Use ISO formatting for no special tokens. If you cannot determine duration, use null for Duration.

        Examples:
        Input: "I am very happy today and had an amazing day keep this note for 3 months"
        Output: {"Title": "My amazing day", "Note": "I am very happy today and had an amazing day", "Duration": 7776000}

        Input: "I will try to buy groceries everyday for 2 hours"
        Output: {"Title": "grocery task", "Note": "I will try to buy groceries everyday for 2 hours", "Duration": None}

        Input: "plan my day everyday title daily plan note for 45 days"
        Output: {"Title": "daily plan", "Note": "plan my day everyday", "Duration": 3888000}
        """

        # Prepare the message for LLM
        messages = [
            {"role": "system", "content": few_shot_prompt},
            {"role": "user", "content": content_after_command}
        ]

        inputs = llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(llm_model.device)
        
        # Ensure pad_token and attention_mask are set to avoid warnings
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        if 'attention_mask' not in inputs:
            import torch as _torch
            # Create attention mask: 1 for non-pad tokens, 0 for pad tokens
            inputs['attention_mask'] = (inputs['input_ids'] != llm_tokenizer.pad_token_id).long()
            
        outputs = llm_model.generate(**inputs, pad_token_id=llm_tokenizer.eos_token_id, max_new_tokens=100)
        response = llm_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]).strip()
        
        # Remove <|eot_id|> token from response
        response = response.replace("<|eot_id|>", "").strip()
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

        print("[DEBUG] response after: ", response.strip())
        try:
            parsed_result = json.loads(response.strip())
            print(f"DEBUG: Parsed result: {parsed_result}")
            return parsed_result
        except json.JSONDecodeError as e:
            print(f"DEBUG: Failed to parse LLM response as JSON: {e}")
            # Fallback: Generate default if LLM fails
            title = datetime.now().strftime("%Y%m%d_%H%M%S")
            return {"Title": title, "Note": content_after_command, "Duration": None}