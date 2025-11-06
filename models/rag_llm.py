import torch
import os
import json
from datetime import datetime
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModelForCausalLM

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

class RAGLLMProcessor:
    def __init__(self, llm_tokenizer, llm_model, embedding_model, notes_dir='notes', embeddings_dir='embeddings'):
        self.llm_tokenizer = llm_tokenizer
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.notes_dir = notes_dir
        self.embeddings_dir = embeddings_dir

    def get_relevant_context(self, user_input, threshold=0.70):
        """
        Return a list of note contents whose cosine similarity to the user_input
        embedding is >= threshold. Results are sorted by descending similarity.
        - user_input: string
        - threshold: float in [0,1]
        """
        print("DEBUG: Retrieving relevant context from stored embeddings (threshold {:.2f})...".format(threshold))

        # Load current notes + embeddings
        vault_content, vault_titles, vault_embeddings = self.load_notes_and_embeddings()

        if vault_embeddings.nelement() == 0:
            print("DEBUG: No embeddings available.")
            return []

        # Compute embedding for user input
        input_emb_np = self.embedding_model.encode([user_input])
        input_emb = torch.tensor(input_emb_np, dtype=torch.float32)  # shape (1, D)

        # Move to same device / dtype if necessary (we use CPU tensors here)
        # Compute cosine similarities
        cos_scores = util.cos_sim(input_emb, vault_embeddings)[0]  # shape (N,)

        # Get indices with similarity >= threshold
        above_mask = cos_scores >= float(threshold)
        if not above_mask.any():
            print("DEBUG: No contexts above similarity threshold.")
            return []

        idxs = torch.where(above_mask)[0].tolist()
        # sort indices by descending similarity
        idxs.sort(key=lambda i: float(cos_scores[i]), reverse=True)

        relevant_contexts = [vault_content[i].strip() for i in idxs]
        similarities = [float(cos_scores[i]) for i in idxs]
        print(f"DEBUG: Found {len(relevant_contexts)} contexts above threshold. Top similarity: {max(similarities):.3f}")

        return relevant_contexts

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
        import json
        from datetime import datetime
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('{"expiration":'):
                json_end = content.find('\n')
                metadata = json.loads(content[:json_end])
                if 'expiration' in metadata:
                    exp_time = datetime.fromisoformat(metadata['expiration'])
                    return datetime.now() > exp_time
        return False

    # Function to chat with streamed response
    def chatgpt_streamed(self, user_input, system_message, conversation_history, bot_name, threshold=0.70):
        print(f"DEBUG: Preparing to send query to LLM: {user_input[:50]}... (truncated)")
        # Get relevant context from the vault
        # threshold can be tuned (e.g. 0.65-0.80). using 0.70 by default here.
        relevant_context = self.get_relevant_context(user_input, threshold=threshold)
        # Concatenate the relevant context with the user's input
        if relevant_context:
            user_input_with_context = "\n".join(relevant_context) + "\n\n" + user_input
            print("DEBUG: Added relevant context to user input. New context:", user_input_with_context)
        else:
            user_input_with_context = user_input
            print("DEBUG: No relevant context found.")
        # Prepare the messages
        messages = [
            {"role": "system", "content": system_message}
        ] + conversation_history + [
            {"role": "user", "content": user_input_with_context}
        ]
        print("DEBUG: Sending request to LLM model (llama3.2:3b)... Waiting for response.")


        # Build inputs via chat template
        inputs = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.llm_model.device)

        # Ensure pad_token and attention_mask are set to avoid warnings
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        if 'attention_mask' not in inputs:
            import torch as _torch
            inputs['attention_mask'] = _torch.ones_like(inputs['input_ids'])

        outputs = self.llm_model.generate(
            **inputs,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            max_new_tokens=100,
        )
        response = self.llm_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]).strip()
        
        # Remove <|eot_id|> token from response
        response = response.replace("<|eot_id|>", "").strip()

        full_response = ""
        line_buffer = ""

        # streamed_completion = client.chat.completions.create(
        #     model="llama3.2:1b",
        #     messages=messages,
        #     stream=True,
        # )
        for chunk in response:
            delta_content = chunk
            if delta_content is not None:
                line_buffer += delta_content
                if '\n' in line_buffer:
                    lines = line_buffer.split('\n')
                    for line in lines[:-1]:
                        print(NEON_GREEN + line + RESET_COLOR)
                        full_response += line + "\n"
                    line_buffer = lines[-1]
        if line_buffer:
            print(NEON_GREEN + line_buffer + RESET_COLOR)
            full_response += line_buffer

        # if '</think>' in full_response:
        #     # Find the last occurrence of </think>
        #     last_think_end = full_response.rfind('</think>')
        #     if last_think_end != -1:
        #         full_response = full_response[last_think_end + len('</think>'):].strip()
        #         print("DEBUG: Extracted response after </think> tag.")
        #     else:
        #         print("DEBUG: No </think> tag found; returning full response.")
        # else:
        #     print("DEBUG: No <think> tags found; returning full response.")

        print(f"DEBUG: Received LLM response: {full_response[:50]}... (truncated)")
        return full_response