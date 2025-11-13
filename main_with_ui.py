import argparse
import os
import re
import threading
from datetime import datetime

# Import our modules
from models.audio_processing import AudioProcessor
from models.text_processing import confirm_and_apply_command_correction
from models.note_management import NoteManager
from models.rag_llm import RAGLLMProcessor
from models.model_setup import setup_whisper_model, setup_xtts_model, setup_llm_model, setup_embedding_model
from models.device_test import pick_device_with_both

# Import UI controller
from ui.orb_controller import OrbController, OrbThread

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Command Line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
parser.add_argument("--no-ui", action='store_true', default=False, help="run without UI")
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
    
    # Initialize UI controller if UI is enabled
    orb_controller = None
    orb_thread = None
    
    if not args.no_ui:
        print("DEBUG: Initializing UI...")
        orb_controller = OrbController()
        orb_thread = OrbThread(orb_controller)
        orb_thread.start()
        
        # Give UI time to initialize
        import time
        time.sleep(1)
        print("DEBUG: UI initialized and running in separate thread")

    # Setup all models
    if orb_controller:
        orb_controller.set_state("processing")
        
    print("DEBUG: Loading models...")
    whisper_model = setup_whisper_model()
    xtts_model, xtts_config = setup_xtts_model()
    llm_tokenizer, llm_model = setup_llm_model()
    embedding_model = setup_embedding_model()

    # Initialize processors
    audio_processor = AudioProcessor(whisper_model, xtts_model, xtts_config)
    note_manager = NoteManager(notes_dir, embeddings_dir, embedding_model)
    rag_llm_processor = RAGLLMProcessor(llm_tokenizer, llm_model, embedding_model, notes_dir, embeddings_dir)

    # Conversation setup
    conversation_history = []
    system_message = "You are a helpful desktop assistant. You have a bitchy personality that complains about the work being given and you answer in short like having a real-time conversation "
    bot_name = "Emma"
    speaker_sample_path = "C:/Users/bitso/XTTS-v2/sample/en_sample.wav"

    input_device, output_device, device_name = pick_device_with_both()
    
    if orb_controller:
        orb_controller.set_state("idle")
    
    print(NEON_GREEN + f"\n{'='*60}\n  Desktop Assistant Ready!\n  Device: {device_name}\n{'='*60}\n" + RESET_COLOR)

    while True:
        print("DEBUG: Starting new loop iteration...")
        
        # Set UI to listening state
        if orb_controller:
            orb_controller.set_state("listening")
            
        # Record user input
        audio_file = "temp_recording.wav"
        audio_processor.record_audio(audio_file, input_device)
        
        # Set UI to processing state while transcribing
        if orb_controller:
            orb_controller.set_state("processing")
            
        user_input = audio_processor.transcribe_with_whisper(audio_file)
        user_input_lower = user_input.lower()
        user_input_lower = confirm_and_apply_command_correction(user_input_lower, threshold=0.85)
        
        # Handle exit command
        m = re.match(r'^\s*create\s+note\b(.*)$', user_input_lower, flags=re.IGNORECASE)
        if user_input_lower == "exit":
            print("DEBUG: Exit command detected. Breaking loop.")
            if orb_controller:
                orb_controller.set_state("idle")
            break
            
        # Handle list notes command
        elif user_input_lower.startswith("list notes"):
            print("DEBUG: Processing 'list notes' command...")
            note_manager.list_notes()
            
            # Brief talking animation for confirmation
            if orb_controller:
                orb_controller.set_state("talking")
                import time
                time.sleep(1)
                orb_controller.set_state("idle")
            continue
            
        # Handle create note command
        elif m:
            print("DEBUG: Processing 'create note' command...")
            if orb_controller:
                orb_controller.set_state("processing")
                
            note_part = m.group(1).strip()
            parsed_notes = note_manager.parse_note_creation(note_part, llm_tokenizer, llm_model)
            title = parsed_notes['Title']
            note_text = parsed_notes['Note']
            duration_seconds = parsed_notes['Duration']

            # Handle duration if None
            if duration_seconds is None:
                print("DEBUG: No duration specified in initial parse. Asking user for duration.")
                
                # Set to talking state for TTS
                if orb_controller:
                    orb_controller.set_state("talking")
                    
                duration_prompt = "Ugh, fine, how long do you want this note to last? Say something like '3 days' or 'forever' if no expiration."
                audio_processor.process_and_play(duration_prompt, speaker_sample_path, output_device)

                # Set to listening state for recording
                if orb_controller:
                    orb_controller.set_state("listening")
                    
                duration_audio = "duration_response.wav"
                audio_processor.record_audio(duration_audio, input_device)
                
                # Set to processing while transcribing
                if orb_controller:
                    orb_controller.set_state("processing")
                    
                duration_response = rag_llm_processor.transcribe_with_whisper(duration_audio)
                os.remove(duration_audio)

                duration_seconds = note_manager.parse_duration_response(duration_response)
                if duration_seconds is None:
                    print("DEBUG: Duration parsing failed or 'forever' implied. Setting to never expire.")
                    duration_seconds = None

            # Create the note
            note_manager.manage_note("create", title, note_text, seconds=duration_seconds)
            
            # Confirm creation with talking animation
            confirm_msg = f"Note '{title}' created. {f'It expires in {duration_seconds} seconds.' if duration_seconds else 'It never expires.'}"
            print('DEBUG: Note Creation Confirm msg: ', confirm_msg)
            
            if orb_controller:
                orb_controller.set_state("talking")
                
            audio_processor.process_and_play(confirm_msg, speaker_sample_path, output_device)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue
            
        # Handle delete note command
        elif user_input_lower.startswith("delete note"):
            print("DEBUG: Processing 'delete note' command...")
            if orb_controller:
                orb_controller.set_state("processing")
                
            title = user_input.split("delete note", 1)[1].strip()
            note_manager.manage_note("delete", title, None)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue

        # Process normal query with RAG and LLM
        print("DEBUG: Processing normal query with RAG and LLM...")
        if orb_controller:
            orb_controller.set_state("processing")
            
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + f"{bot_name}: " + RESET_COLOR)
        
        chat_response = rag_llm_processor.chatgpt_streamed(user_input, system_message, conversation_history, bot_name, threshold=0.70)
        conversation_history.append({"role": "assistant", "content": chat_response})
        
        # Set to talking state during TTS
        if orb_controller:
            orb_controller.set_state("talking")
            
        audio_processor.process_and_play(chat_response, speaker_sample_path, output_device)
        
        # Return to idle
        if orb_controller:
            orb_controller.set_state("idle")
            
        # Keep only the last 20 messages for context
        conversation_history = conversation_history[-20:]
        print("DEBUG: Conversation history trimmed to last 20 messages.")
    
    # Cleanup
    print("DEBUG: Shutting down...")
    if orb_thread:
        orb_thread.quit()
        orb_thread.wait()

if __name__ == "__main__":
    main()
