import argparse
import os
import re
from datetime import datetime

# Import our modules
from models.audio_processing import AudioProcessor
from models.text_processing import confirm_and_apply_command_correction
from models.note_management import NoteManager
from models.rag_llm import RAGLLMProcessor
from models.model_setup import setup_whisper_model, setup_xtts_model, setup_llm_model, setup_embedding_model
from models.device_test import pick_device_with_both

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Command Line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
device = 'cpu'
output_dir = 'outputs'  # Adjust if needed
os.makedirs(output_dir, exist_ok=True)

# Notes and embeddings directories
notes_dir = 'user_data/notes'
embeddings_dir = 'user_data/embeddings'
os.makedirs(notes_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)

def main():
    print("DEBUG: Entering main execution block...")

    # Setup all models
    whisper_model = setup_whisper_model()
    xtts_model, xtts_config = setup_xtts_model()
    llm_tokenizer, llm_model = setup_llm_model()
    embedding_model = setup_embedding_model()

    # Initialize processors
    audio_processor = AudioProcessor(whisper_model, xtts_model, xtts_config)
    note_manager = NoteManager(notes_dir, embeddings_dir, embedding_model)
    rag_llm_processor = RAGLLMProcessor(llm_tokenizer, llm_model, embedding_model, notes_dir, embeddings_dir)

    # Load notes + embeddings from the notes/embeddings folders
    # vault_content, vault_titles, vault_embeddings = note_manager.load_notes_and_embeddings()

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
        audio_processor.record_audio(audio_file, input_device)
        user_input = audio_processor.transcribe_with_whisper(audio_file)
        # os.remove(audio_file)  # Clean up the temporary audio file
        # print(f"DEBUG: Cleaned up temporary audio file: {audio_file}")
        user_input_lower = user_input.lower()
        user_input_lower = confirm_and_apply_command_correction(user_input_lower, threshold=0.85)
        # user_input_lower = "create note i will stay awake till tomorrow at 8 am keep this note for 9 hours"
        # user_input = "create note i will stay awake till tomorrow at 8 am keep this note for 9 hours"
        m = re.match(r'^\s*create\s+note\b(.*)$', user_input_lower, flags=re.IGNORECASE)
        if user_input_lower == "exit":
            print("DEBUG: Exit command detected. Breaking loop.")
            break
        elif user_input_lower.startswith("list notes"):
            print("DEBUG: Processing 'list notes' command...")
            note_manager.list_notes()
            continue
        elif m:
            print("DEBUG: Processing 'create note' command...")
            note_part = m.group(1).strip()
            # Assume text after "create note" is the note content
            # note_part = user_input.split("create note", 1)[1].strip()
            parsed_notes = note_manager.parse_note_creation(note_part, llm_tokenizer, llm_model)
            title = parsed_notes['Title']
            note_text = parsed_notes['Note']
            duration_seconds = parsed_notes['Duration']

            # TODO Implementation: Handle duration if None
            if duration_seconds is None:
                print("DEBUG: No duration specified in initial parse. Asking user for duration.")
                # Synthesize and play prompt for duration
                duration_prompt = "Ugh, fine, how long do you want this note to last? Say something like '3 days' or 'forever' if no expiration."
                audio_processor.process_and_play(duration_prompt, speaker_sample_path, output_device)

                # Record and transcribe user response
                duration_audio = "duration_response.wav"
                audio_processor.record_audio(duration_audio, input_device)
                duration_response = rag_llm_processor.transcribe_with_whisper(duration_audio)
                os.remove(duration_audio)

                # Parse the response
                duration_seconds = note_manager.parse_duration_response(duration_response)
                if duration_seconds is None:
                    print("DEBUG: Duration parsing failed or 'forever' implied. Setting to never expire.")
                    duration_seconds = None

            # Now create the note with the resolved duration
            note_manager.manage_note("create", title, note_text, seconds=duration_seconds)
            # Optional: Confirm creation

            confirm_msg = f"Note '{title}' created. {f'It expires in {duration_seconds} seconds.' if duration_seconds else 'It never expires.'}"
            print('DEBUG: Note Creation Confirm msg: ', confirm_msg)
            audio_processor.process_and_play(confirm_msg, speaker_sample_path, output_device)
            continue
        elif user_input_lower.startswith("delete note"):
            print("DEBUG: Processing 'delete note' command...")
            title = user_input.split("delete note", 1)[1].strip()
            note_manager.manage_note("delete", title, None)
            continue

        # Process normal query with RAG and LLM
        print("DEBUG: Processing normal query with RAG and LLM...")
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + f"{bot_name}: " + RESET_COLOR)
        chat_response = rag_llm_processor.chatgpt_streamed(user_input, system_message, conversation_history, bot_name, threshold=0.70)
        conversation_history.append({"role": "assistant", "content": chat_response})
        audio_processor.process_and_play(chat_response, speaker_sample_path, output_device)
        # Keep only the last 20 messages for context
        conversation_history = conversation_history[-20:]
        print("DEBUG: Conversation history trimmed to last 20 messages.")

if __name__ == "__main__":
    main()
