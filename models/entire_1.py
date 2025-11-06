import argparse
import os
import wave
import torch
import argparse
import os
import json
import time
from datetime import datetime, timedelta
import wave
import pyaudio #type: ignore
import soundfile as sf #type: ignore
from openai import OpenAI #type: ignore
from faster_whisper import WhisperModel #type: ignore
from sentence_transformers import SentenceTransformer, util #type: ignore
from TTS.tts.configs.xtts_config import XttsConfig #type: ignore
from TTS.tts.models.xtts import Xtts #type: ignore
from TTS.tts.utils.speakers import SpeakerManager #type: ignore
from TTS.tts.utils.text.tokenizer import TTSTokenizer #type: ignore
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer #type: ignore
from models.device_test import pick_device_with_both
import difflib
import re
from transformers import AutoTokenizer, AutoModelForCausalLM #type: ignore
# ANSI escape codes for colors


PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

CANONICAL_COMMANDS = [
    "create note",
    "delete note",
    "list notes",
    "exit",
    # add more commands you support, exact canonical phrases you want to force
]

print("DEBUG: Setting up the faster-whisper model...")
# Set up the faster-whisper model
model_size = "medium.en"
whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Function to open a file and return its contents as a string
def open_file(filepath):
    print(f"DEBUG: Opening file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Initialize the OpenAI client with the API key
# client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",
                                             device_map="cpu",         # <- ensures CPU usage
                                            torch_dtype="float32")


def _clean_text_for_compare(s: str) -> str:
    s = s.lower().strip()
    # remove punctuation except keep spaces and alphanumerics
    s = re.sub(r'[^0-9a-z\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# --- helper: find best matching canonical command using only the first N words ---
def find_best_command_match(transcript: str, commands=CANONICAL_COMMANDS):
    """
    Returns a tuple (best_command, similarity_float, command_word_count)
    or (None, 0.0, 0) if no good candidates.
    similarity is in [0.0, 1.0].
    """
    if not transcript or transcript.strip() == "":
        return None, 0.0, 0

    cleaned = _clean_text_for_compare(transcript)
    words = cleaned.split()
    if not words:
        return None, 0.0, 0

    best = (None, 0.0, 0)
    for cmd in commands:
        cmd_clean = _clean_text_for_compare(cmd)
        cmd_words = cmd_clean.split()
        if not cmd_words:
            continue
        n = len(cmd_words)
        # if transcript shorter than n words, still compare available portion
        prefix = " ".join(words[:n])
        # compute similarity ratio
        ratio = difflib.SequenceMatcher(None, prefix, cmd_clean).ratio()
        if ratio > best[1]:
            best = (cmd, ratio, n)
    return best  # (command, ratio, n)

# --- interactive disambiguation wrapper (call after you transcribe) ---
def confirm_and_apply_command_correction(transcript: str, threshold: float = 0.90):
    """
    If the start of `transcript` is similar to a canonical command >= threshold,
    prompt the user: "Did you mean '...'? (y/n)". If yes, replace the first N words
    in the transcript with the canonical command and return the new transcript.
    Otherwise return original transcript.
    """
    cmd, similarity, n = find_best_command_match(transcript)
    if cmd is None:
        return transcript

    # Prompt user (text prompt). Use full human-readable candidate.
    print(f"\nDid you mean the command: '{cmd}' ?  (similarity {similarity*100:.1f}%)")
    if similarity < threshold or similarity > 0.98:
        return transcript
    
    ans = input("Type 'y' or 'yes' to accept, anything else to keep original: ").strip().lower()
    if ans in ("y", "yes"):
        # replace first n words of original transcript (preserve rest)
        orig_words = transcript.split()
        # if original transcript has fewer than n words, just use command alone
        rest = orig_words[n:] if len(orig_words) > n else []
        new_transcript = " ".join([cmd] + rest).strip()
        print(f"[AUTOCORRECT] Using clarified command: {new_transcript}")
        return new_transcript
    else:
        print("[AUTOCORRECT] Keeping original transcript.")
        return transcript


# Function to play audio using PyAudio
def play_audio(file_path, output_device_index=None):
    print(f"DEBUG: Playing audio: {file_path}, output_device_index={output_device_index}")
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output_device_index=output_device_index,
                    output=True,
                    frames_per_buffer=1024)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.close()
    p.terminate()
    print("DEBUG: Audio playback completed.")

# Command Line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
output_dir = 'outputs'  # Adjust if needed
os.makedirs(output_dir, exist_ok=True)

# Notes and embeddings directories
notes_dir = 'notes'
embeddings_dir = 'embeddings'
os.makedirs(notes_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)

print("DEBUG: Loading XTTS configuration...")
# Load XTTS configuration
xtts_config = XttsConfig()
xtts_config.load_json("C:/Users/bitso/XTTS-v2/config.json")  # Replace with your actual path

print("DEBUG: Initializing XTTS model...")
# Initialize XTTS model
xtts_model = Xtts.init_from_config(xtts_config)
# Manually load the checkpoint with weights_only=False
checkpoint_path = os.path.join("C:/Users/bitso/XTTS-v2/", "model.pth")  # Adjust if your checkpoint file has a different name
print(f"DEBUG: Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

# Apply to the model (adapt based on TTS internals; this assumes the 'model' key has the state dict)
xtts_model.load_state_dict(checkpoint['model'], strict=False)  # Use strict=False if keys don't match perfectly
# xtts_model.load_checkpoint(xtts_config, checkpoint_dir="C:/Users/bitso/XTTS-v2/", eval=True)  # Replace with your actual path
# xtts_model.cuda()  # Move the model to GPU if available

checkpoint_dir = "C:/Users/bitso/XTTS-v2/"  # Your directory path

# Load tokenizer from vocab.json
print("DEBUG: Loading tokenizer...")
vocab_path = os.path.join(checkpoint_dir, "vocab.json")
xtts_model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
print("DEBUG: Tokenizer loaded successfully.")

# Load speaker manager from speakers_xtts.pth
print("DEBUG: Loading speaker manager...")
speaker_file = os.path.join(checkpoint_dir, "speakers_xtts.pth")
xtts_model.speaker_manager = SpeakerManager()
xtts_model.speaker_manager.speakers = torch.load(speaker_file, map_location=torch.device('cpu'))
print("DEBUG: Speaker manager loaded successfully.")

# Load DVAE from dvae.pth
# print("DEBUG: Loading DVAE...")
# dvae_path = os.path.join(checkpoint_dir, "dvae.pth")
# dvae_checkpoint = torch.load(dvae_path, map_location=torch.device("cpu"), weights_only=False)
# # Assume it's keyed under 'model'; adjust if traceback shows otherwise (e.g., to dvae_checkpoint directly)
# xtts_model.dvae.load_state_dict(dvae_checkpoint['model'] if 'model' in dvae_checkpoint else dvae_checkpoint, strict=True)
# print("DEBUG: DVAE loaded successfully.")

# Load mel stats from mel_stats.pth
print("DEBUG: Loading mel stats...")
mel_stats_path = os.path.join(checkpoint_dir, "mel_stats.pth")
xtts_model.mel_stats = torch.load(mel_stats_path, map_location=torch.device("cpu"))
print("DEBUG: Mel stats loaded successfully.")

# Set model to eval mode and move to device (equivalent to eval=True in load_checkpoint)
xtts_model.eval()
print("DEBUG: XTTS model loaded successfully.")

# Function to synthesize speech using XTTS
def process_and_play(prompt, audio_file_pth, output_device):
    print(f"DEBUG: Starting speech synthesis for prompt: {prompt[:50]}... (truncated)")
    try:
        # Use XTTS to synthesize speech
        outputs = xtts_model.synthesize(
            prompt,  # Pass the prompt as a string directly
            xtts_config,
            speaker_wav=audio_file_pth,  # Pass the file path directly
            gpt_cond_len=24,
            temperature=0.6,
            language='en',
            speed=1.2  # Specify the desired language
        )
        # Get the synthesized audio tensor from the dictionary
        synthesized_audio = outputs['wav']
        # Save the synthesized audio to the output path
        src_path = f'{output_dir}/output.wav'
        sample_rate = xtts_config.audio.sample_rate
        print(f"DEBUG: Saving synthesized audio to: {src_path}")
        sf.write(src_path, synthesized_audio, sample_rate)
        print("Audio generated successfully.")
        play_audio(src_path, output_device)
    except Exception as e:
        print(f"Error during audio generation: {e}")

def get_relevant_context(user_input, embedding_model, notes_dir, embeddings_dir, threshold=0.70):
    """
    Return a list of note contents whose cosine similarity to the user_input
    embedding is >= threshold. Results are sorted by descending similarity.
    - user_input: string
    - embedding_model: sentence-transformers model used to compute embedding
    - notes_dir / embeddings_dir: folders used by load_notes_and_embeddings()
    - threshold: float in [0,1]
    """
    print("DEBUG: Retrieving relevant context from stored embeddings (threshold {:.2f})...".format(threshold))

    # Load current notes + embeddings
    vault_content, vault_titles, vault_embeddings = load_notes_and_embeddings(embedding_model)

    if vault_embeddings.nelement() == 0:
        print("DEBUG: No embeddings available.")
        return []

    # Compute embedding for user input
    input_emb_np = embedding_model.encode([user_input])
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


# Function to chat with streamed response
def chatgpt_streamed(user_input, system_message, conversation_history, bot_name, vault_embeddings, vault_content, model):
    print(f"DEBUG: Preparing to send query to LLM: {user_input[:50]}... (truncated)")
    # Get relevant context from the vault
    # threshold can be tuned (e.g. 0.65-0.80). using 0.70 by default here.
    relevant_context = get_relevant_context(user_input, embedding_model, notes_dir, embeddings_dir, threshold=0.70)
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
    
    
    inputs = llm_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(llm_model.device)

    outputs = llm_model.generate(**inputs, max_new_tokens=100)
    response = llm_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]).strip()
    
    full_response = ""
    line_buffer = ""
    
    # streamed_completion = client.chat.completions.create(
    #     model="llama3.2:1b",
    #     messages=messages,
    #     stream=True,
    # )
    for chunk in response:
        delta_content = chunk.choices[0].delta.content
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

# Function to transcribe the recorded audio using faster-whisper
def transcribe_with_whisper(audio_file):
    print(f"DEBUG: Transcribing audio file: {audio_file}")
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    transcription = transcription.strip()
    print(f"DEBUG: Transcription result: {transcription}")
    return transcription

# Function to record audio from the microphone and save to a file
def record_audio(file_path, input_device_index=None):
    print(f"DEBUG: Starting recording -> {file_path} using input_device_index={input_device_index}")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, input_device_index=input_device_index, frames_per_buffer=1024)
    frames = []
    print("Recording...")
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    print("Recording stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    print(f"DEBUG: Audio recording completed and saved to: {file_path}")


def load_notes_and_embeddings(embedding_model):
    """
    Load note text files from notes_dir and their embeddings from embeddings_dir.
    If an embedding file is missing, compute it and save as .pt.
    Returns: vault_content (list[str]), vault_titles (list[str]), vault_embeddings (torch.Tensor)
    """
    print("DEBUG: Loading notes and embeddings from folders...")
    vault_content = []
    vault_titles = []
    vault_embeddings_list = []

    for filename in sorted(os.listdir(notes_dir)):
        if not filename.endswith('.txt'):
            continue
        title = filename[:-4]
        note_path = os.path.join(notes_dir, filename)
        emb_path = os.path.join(embeddings_dir, f"{title}.pt")

        # Skip expired notes
        if is_expired(note_path):
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
                emb_np = embedding_model.encode([content])
                emb = torch.tensor(emb_np, dtype=torch.float32)
                torch.save(emb, emb_path)
        else:
            emb_np = embedding_model.encode([content])
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


def is_expired(note_path):
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
def manage_note(action, title, text, seconds=None):
    note_path = os.path.join(notes_dir, f"{title}.txt")
    emb_path = os.path.join(embeddings_dir, f"{title}.pt")
    
    if action == "create" or action == "edit":
        metadata = {}
        if seconds:
            exp_time = datetime.now() + timedelta(seconds=seconds)
            metadata['expiration'] = exp_time.isoformat()
        
        content = json.dumps(metadata) + '\n' + text if metadata else text
        
        with open(note_path, 'w' if action == "create" else 'a', encoding='utf-8') as f:
            f.write(content)
        
        # Compute and save embedding
        emb = torch.tensor(embedding_model.encode([text]))
        torch.save(emb, emb_path)
        print(f"DEBUG: {action.capitalize()}d note '{title}'.")
    
    elif action == "delete":
        if os.path.exists(note_path):
            os.remove(note_path)
        if os.path.exists(emb_path):
            os.remove(emb_path)
        print(f"DEBUG: Deleted note '{title}'.")

# Function to list notes
def list_notes():
    active_notes = []
    for filename in os.listdir(notes_dir):
        if filename.endswith('.txt'):
            title = filename[:-4]
            note_path = os.path.join(notes_dir, filename)
            if not is_expired(note_path):
                active_notes.append(title)
    print(NEON_GREEN + "Active Notes: " + ", ".join(active_notes) + RESET_COLOR)

def parse_duration_response(duration_input):
    """
    Parse user's duration response (e.g., "3 days", "1 month") and convert to seconds.
    If no unit, assume seconds. If invalid, return None (never expire).
    """
    print(f"DEBUG: Parsing duration response: {duration_input}")
    # Simple regex or keyword-based parsing
    import re
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
    
def parse_note_creation(content_after_command):
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

    outputs = llm_model.generate(**inputs, max_new_tokens=100)
    response = llm_tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
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


if __name__ == "__main__":
    print("DEBUG: Entering main execution block...")
    # Load the sentence transformer model for embeddings
    print("DEBUG: Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("DEBUG: Sentence transformer model loaded.")
    
    # Load notes + embeddings from the notes/embeddings folders
    vault_content, vault_titles, vault_embeddings = load_notes_and_embeddings(embedding_model)

    # Conversation setup
    conversation_history = []
    system_message = "You are a helpful desktop assistant. You have a bitchy personality that complains about the work being given and you answer in short like having a real-time conversation "  # Customize as needed
    bot_name = "Emma"
    # Path to speaker sample for TTS (replace with your own)
    speaker_sample_path = "C:/Users/bitso/XTTS-v2/sample/en_sample.wav"
    
    input_device, output_device, device_name = pick_device_with_both()
    
    while True:
        print("DEBUG: Starting new loop iteration...")
        # Record user input
        audio_file = "temp_recording.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        # os.remove(audio_file)  # Clean up the temporary audio file
        # print(f"DEBUG: Cleaned up temporary audio file: {audio_file}")
        user_input_lower = user_input.lower()
        user_input_lower = confirm_and_apply_command_correction(user_input_lower, threshold=0.85)
        user_input_lower = "create note i will stay awake till tomorrow at 8 am keep this note for 9 hours"
        user_input = "create note i will stay awake till tomorrow at 8 am keep this note for 9 hours"
        m = re.match(r'^\s*create\s+note\b(.*)$', user_input_lower, flags=re.IGNORECASE)
        if user_input_lower == "exit":
            print("DEBUG: Exit command detected. Breaking loop.")
            break
        elif user_input_lower.startswith("list notes"):
            print("DEBUG: Processing 'list notes' command...")
            list_notes()
            continue
        elif m:
            print("DEBUG: Processing 'create note' command...")
            note_part = m.group(1).strip()
            # Assume text after "create note" is the note content
            # note_part = user_input.split("create note", 1)[1].strip()
            parsed_notes = parse_note_creation(note_part)
            title = parsed_notes['Title']
            note_text = parsed_notes['Note']
            duration_seconds = parsed_notes['Duration']
            
            # TODO Implementation: Handle duration if None
            if duration_seconds is None:
                print("DEBUG: No duration specified in initial parse. Asking user for duration.")
                # Synthesize and play prompt for duration
                duration_prompt = "Ugh, fine, how long do you want this note to last? Say something like '3 days' or 'forever' if no expiration."
                process_and_play(duration_prompt, speaker_sample_path, output_device)
                
                # Record and transcribe user response
                duration_audio = "duration_response.wav"
                record_audio(duration_audio, input_device)
                duration_response = transcribe_with_whisper(duration_audio)
                os.remove(duration_audio)
                
                # Parse the response
                duration_seconds = parse_duration_response(duration_response)
                if duration_seconds is None:
                    print("DEBUG: Duration parsing failed or 'forever' implied. Setting to never expire.")
                    duration_seconds = None
            
            # Now create the note with the resolved duration
            manage_note("create", title, note_text, seconds=duration_seconds)
            # Optional: Confirm creation
            
            confirm_msg = f"Note '{title}' created. {f'It expires in {duration_seconds} seconds.' if duration_seconds else 'It never expires.'}"
            print('DEBUG: Note Creation Confirm msg: ', confirm_msg)
            process_and_play(confirm_msg, speaker_sample_path, output_device)
            continue
        elif user_input_lower.startswith("delete note"):
            print("DEBUG: Processing 'delete note' command...")
            title = user_input.split("delete note", 1)[1].strip()
            manage_note("delete", title, None)
            continue
        
        # Process normal query with RAG and LLM
        print("DEBUG: Processing normal query with RAG and LLM...")
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + f"{bot_name}: " + RESET_COLOR)
        chat_response = chatgpt_streamed(user_input, system_message, conversation_history, bot_name, vault_embeddings, vault_content, embedding_model)
        conversation_history.append({"role": "assistant", "content": chat_response})
        process_and_play(chat_response, speaker_sample_path)
        # Keep only the last 20 messages for context
        conversation_history = conversation_history[-20:]
        print("DEBUG: Conversation history trimmed to last 20 messages.")