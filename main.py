import argparse
import os
import re
from datetime import datetime
import time

# Import our modules
from models.audio_processing import AudioProcessor
from models.text_processing import confirm_and_apply_command_correction
from models.note_management import NoteManager
from models.rag_llm import RAGLLMProcessor
from models.model_setup import setup_whisper_model, setup_xtts_model, setup_llm_model, setup_embedding_model
from models.device_test import pick_device_with_both
from models.llm_interface import create_llm_interface, USE_LOCAL_MODEL

# Import services
from services.gmail_services import GmailService, EmailParser
from services.calendar_services import CalendarService, DurationParser, EventParser, TimeParser

# Import UI controller
from ui.orb_controller import OrbController

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
    
    if not args.no_ui:
        print("DEBUG: Initializing UI...")
        orb_controller = OrbController()
        orb_controller.start_ui()
        
        # Give UI time to initialize
        time.sleep(2)
        print("DEBUG: UI initialized and running in separate thread")

    # Setup all models
    if orb_controller:
        orb_controller.set_state("processing")
        
    print("DEBUG: Loading models...")
    whisper_model = setup_whisper_model()
    xtts_model, xtts_config = setup_xtts_model()
    
    # Set whether to use local or API model
    use_local = USE_LOCAL_MODEL  # Change this in models/llm_interface.py or set it here
    
    if use_local:
        print(CYAN + "Using LOCAL LLM model" + RESET_COLOR)
        llm_tokenizer, llm_model = setup_llm_model()
    else:
        print(CYAN + "Using HUGGING FACE API model" + RESET_COLOR)
        llm_tokenizer, llm_model = None, None  # Not needed for API
    
    embedding_model = setup_embedding_model()
    
    # Create unified LLM interface
    llm_interface = create_llm_interface(
        use_local=use_local,
        llm_tokenizer=llm_tokenizer,
        llm_model=llm_model
    )

    # Initialize processors with LLM interface
    audio_processor = AudioProcessor(whisper_model, xtts_model, xtts_config)
    note_manager = NoteManager(notes_dir, embeddings_dir, embedding_model, 
                                llm_interface=llm_interface, use_local_llm=use_local)
    rag_llm_processor = RAGLLMProcessor(llm_tokenizer, llm_model, embedding_model, notes_dir, embeddings_dir,
                                        llm_interface=llm_interface, use_local_llm=use_local)
    
    # Initialize Gmail services
    gmail_service = GmailService()
    email_parser = EmailParser(llm_interface)
    
    # Initialize Calendar services
    calendar_service = CalendarService()
    duration_parser = DurationParser()
    time_parser = TimeParser()
    event_parser = EventParser(llm_interface)

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
        corrected_transcript = confirm_and_apply_command_correction(user_input_lower, threshold=0.85)
        user_input = corrected_transcript
        user_input_lower = corrected_transcript.lower()
        
        # Handle exit command
        m_create_note = re.match(r'^\s*create\s+note\b(.*)$', user_input_lower, flags=re.IGNORECASE)
        m_send_email = re.match(r'^\s*send\s+email\b(.*)$', user_input_lower, flags=re.IGNORECASE)
        m_check_gmail = re.match(r'^\s*check\s+gmail\b(.*)$', user_input_lower, flags=re.IGNORECASE)
        m_create_event = re.match(r'^\s*create\s+event\b(.*)$', user_input_lower, flags=re.IGNORECASE)
        m_list_events = re.match(r'^\s*list\s+events\b(.*)$', user_input_lower, flags=re.IGNORECASE)
        m_search_event = re.match(r'^\s*search\s+event\b(.*)$', user_input_lower, flags=re.IGNORECASE)
        
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
                time.sleep(1)
                orb_controller.set_state("idle")
            continue
            
        # Handle send email command
        elif m_send_email:
            print("DEBUG: Processing 'send email' command...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            email_part = "send email " +  m_send_email.group(1).strip()
            
            # If no email content after command, ask for it
            if not email_part:
                prompt = "Ugh, fine. Tell me the recipient, subject, and what you want to say."
                audio_processor.process_and_play(prompt, speaker_sample_path, output_device, orb_controller)
                
                if orb_controller:
                    orb_controller.set_state("listening")
                
                email_audio = "email_input.wav"
                audio_processor.record_audio(email_audio, input_device)
                
                if orb_controller:
                    orb_controller.set_state("processing")
                
                email_part = audio_processor.transcribe_with_whisper(email_audio)
                os.remove(email_audio)
            
            # Parse email details using LLM
            parsed_email = email_parser.parse_email_creation(email_part)
            to_recipient = parsed_email['to']
            subject = parsed_email['subject']
            body = parsed_email['body']
            
            if not to_recipient or not body:
                error_msg = "Ugh, I couldn't understand the email details. Try again with recipient and content."
                audio_processor.process_and_play(error_msg, speaker_sample_path, output_device, orb_controller)
                if orb_controller:
                    orb_controller.set_state("idle")
                continue
            
            # Check if recipient is a name (convert to email)
            if '@' not in to_recipient:
                email_result = gmail_service.get_email_from_name(to_recipient, threshold=0.9)
                email_address, matched_name, similarity = email_result
                
                if email_address:
                    # Found a match (exact or fuzzy)
                    if similarity < 1.0:
                        # Fuzzy match - ask for terminal confirmation (y/n)
                        print(f"\n{CYAN}Fuzzy match found: {matched_name} (similarity: {similarity:.2%}){RESET_COLOR}")
                        print(f"{CYAN}Email: {email_address}{RESET_COLOR}")
                        terminal_confirm = input(f"Did you mean {matched_name}? (y/n): ").strip().lower()
                        
                        if terminal_confirm not in ['y', 'yes']:
                            # User rejected - ask for email address or name
                            print(f"{YELLOW}Please provide the correct email address or name:{RESET_COLOR}")
                            
                            while True:
                                ask_email = f"What's the email address or name for {to_recipient}?"
                                audio_processor.process_and_play(ask_email, speaker_sample_path, output_device, orb_controller)
                                
                                if orb_controller:
                                    orb_controller.set_state("listening")
                                
                                email_audio = "email_address.wav"
                                audio_processor.record_audio(email_audio, input_device)
                                
                                if orb_controller:
                                    orb_controller.set_state("processing")
                                
                                email_input = audio_processor.transcribe_with_whisper(email_audio)
                                os.remove(email_audio)
                                
                                # Check if it's an email or name
                                if '@' in email_input:
                                    # It's an email address
                                    email_address = email_input.strip().replace(' ', '').lower()
                                    # Save contact
                                    gmail_service.add_contact(to_recipient, email_address)
                                    break
                                else:
                                    # It's a name - try to find it again
                                    retry_result = gmail_service.get_email_from_name(email_input, threshold=0.9)
                                    retry_email, retry_name, retry_sim = retry_result
                                    
                                    if retry_email:
                                        if retry_sim < 1.0:
                                            # Another fuzzy match
                                            print(f"\n{CYAN}Found: {retry_name} (similarity: {retry_sim:.2%}){RESET_COLOR}")
                                            print(f"{CYAN}Email: {retry_email}{RESET_COLOR}")
                                            retry_confirm = input(f"Did you mean {retry_name}? (y/n): ").strip().lower()
                                            
                                            if retry_confirm in ['y', 'yes']:
                                                email_address = retry_email
                                                break
                                        else:
                                            # Exact match
                                            email_address = retry_email
                                            break
                        # else: user confirmed, use email_address
                    # else: exact match, use email_address
                else:
                    # No match found - ask for email address or name
                    print(f"{YELLOW}No contact found for '{to_recipient}'{RESET_COLOR}")
                    
                    while True:
                        ask_email = f"I don't have an email for {to_recipient}. What's their email address or name?"
                        audio_processor.process_and_play(ask_email, speaker_sample_path, output_device, orb_controller)
                        
                        if orb_controller:
                            orb_controller.set_state("listening")
                        
                        email_audio = "email_address.wav"
                        audio_processor.record_audio(email_audio, input_device)
                        
                        if orb_controller:
                            orb_controller.set_state("processing")
                        
                        email_input = audio_processor.transcribe_with_whisper(email_audio)
                        os.remove(email_audio)
                        
                        # Check if it's an email or name
                        if '@' in email_input:
                            # It's an email address
                            email_address = email_input.strip().replace(' ', '').lower()
                            # Save contact
                            gmail_service.add_contact(to_recipient, email_address)
                            break
                        else:
                            # It's a name - try to find it
                            retry_result = gmail_service.get_email_from_name(email_input, threshold=0.9)
                            retry_email, retry_name, retry_sim = retry_result
                            
                            if retry_email:
                                if retry_sim < 1.0:
                                    # Fuzzy match
                                    print(f"\n{CYAN}Found: {retry_name} (similarity: {retry_sim:.2%}){RESET_COLOR}")
                                    print(f"{CYAN}Email: {retry_email}{RESET_COLOR}")
                                    retry_confirm = input(f"Did you mean {retry_name}? (y/n): ").strip().lower()
                                    
                                    if retry_confirm in ['y', 'yes']:
                                        email_address = retry_email
                                        break
                                else:
                                    # Exact match
                                    email_address = retry_email
                                    break
                
                to_recipient = email_address
            
            # Send the email
            success = gmail_service.send_email(to_recipient, subject, body)
            
            if success:
                confirm_msg = f"Email sent to {to_recipient} with subject: {subject}"
            else:
                confirm_msg = "Ugh, something went wrong. The email wasn't sent."
            
            audio_processor.process_and_play(confirm_msg, speaker_sample_path, output_device, orb_controller)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue
        
        # Handle check gmail command
        elif m_check_gmail:
            print("DEBUG: Processing 'check gmail' command...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            search_query = m_check_gmail.group(1).strip()
            
            # If no search query, ask what to check for
            if not search_query:
                prompt = "What do you want me to check your emails for?"
                audio_processor.process_and_play(prompt, speaker_sample_path, output_device, orb_controller)
                
                if orb_controller:
                    orb_controller.set_state("listening")
                
                search_audio = "search_query.wav"
                audio_processor.record_audio(search_audio, input_device)
                
                if orb_controller:
                    orb_controller.set_state("processing")
                
                search_query = audio_processor.transcribe_with_whisper(search_audio)
                os.remove(search_audio)
            
            # Get recent emails
            print(f"DEBUG: Searching last 50 emails for: {search_query}")
            recent_emails = gmail_service.get_recent_emails(max_results=50)
            
            if not recent_emails:
                response_msg = "I couldn't fetch your emails. Maybe check your internet connection?"
                audio_processor.process_and_play(response_msg, speaker_sample_path, output_device, orb_controller)
                if orb_controller:
                    orb_controller.set_state("idle")
                continue
            
            # Search through emails using similarity (60% threshold)
            matching_emails = gmail_service.search_emails_for_content(recent_emails, search_query, similarity_threshold=0.6)
            
            # Generate LLM-based summary
            summary = gmail_service.generate_email_summary(llm_interface, matching_emails, search_query)
            
            print(f"DEBUG: Email search summary: {summary}")
            audio_processor.process_and_play(summary, speaker_sample_path, output_device, orb_controller)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue
        
        # Handle create event command
        elif m_create_event:
            print("DEBUG: Processing 'create event' command...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            event_part = m_create_event.group(1).strip()
            
            # If no event details after command, ask for them
            if not event_part:
                prompt = "What's the event? Tell me the title, description, start time, and end time."
                audio_processor.process_and_play(prompt, speaker_sample_path, output_device, orb_controller)
                
                if orb_controller:
                    orb_controller.set_state("listening")
                
                event_audio = "event_input.wav"
                audio_processor.record_audio(event_audio, input_device)
                
                if orb_controller:
                    orb_controller.set_state("processing")
                
                event_part = audio_processor.transcribe_with_whisper(event_audio)
                os.remove(event_audio)
            
            # Parse event details using LLM
            parsed_event = event_parser.parse_event_creation(event_part)
            title = parsed_event['title']
            description = parsed_event['description']
            date_str = parsed_event['date']
            start_time_str = parsed_event['start_time']
            end_time_str = parsed_event['end_time']
            
            print(f"DEBUG: Parsed - date:{date_str}, start:{start_time_str}, end:{end_time_str}")
            
            # Parse date from event (or use today as default)
            if date_str:
                event_date = time_parser.parse_date_string(date_str)
                if event_date is None:
                    print(f"DEBUG: Could not parse date '{date_str}', using today")
                    event_date = datetime.now().date()
                else:
                    print(f"DEBUG: Using parsed date: {event_date}")
            else:
                event_date = datetime.now().date()
                print(f"DEBUG: No date provided, using today: {event_date}")
            
            # Parse start time
            if start_time_str:
                # Extract time component
                time_obj = time_parser.parse_time_string(start_time_str)
                
                if time_obj:
                    start_time = datetime.combine(event_date, time_obj)
                else:
                    start_time = None
            else:
                start_time = None
            
            # Parse end time
            if end_time_str:
                # Extract time component
                time_obj = time_parser.parse_time_string(end_time_str)
                
                if time_obj:
                    end_time = datetime.combine(event_date, time_obj)
                else:
                    end_time = None
            else:
                end_time = None
            
            # If start time not provided, ask for it
            if start_time is None:
                prompt = "When should the event start? Tell me the date and time."
                audio_processor.process_and_play(prompt, speaker_sample_path, output_device, orb_controller)
                
                if orb_controller:
                    orb_controller.set_state("listening")
                
                start_audio = "event_start.wav"
                audio_processor.record_audio(start_audio, input_device)
                
                if orb_controller:
                    orb_controller.set_state("processing")
                
                start_text = audio_processor.transcribe_with_whisper(start_audio)
                os.remove(start_audio)
                
                # Parse start time
                date_obj = time_parser.parse_date_string(start_text)
                time_obj = time_parser.parse_time_string(start_text)
                
                if date_obj is None:
                    date_obj = datetime.now().date()
                
                if time_obj:
                    start_time = datetime.combine(date_obj, time_obj)
                else:
                    error_msg = "I couldn't understand the start time. Event not created."
                    audio_processor.process_and_play(error_msg, speaker_sample_path, output_device, orb_controller)
                    if orb_controller:
                        orb_controller.set_state("idle")
                    continue
            
            # If end time not provided, ask for it
            if end_time is None:
                prompt = "When should the event end? Tell me the date and time."
                audio_processor.process_and_play(prompt, speaker_sample_path, output_device, orb_controller)
                
                if orb_controller:
                    orb_controller.set_state("listening")
                
                end_audio = "event_end.wav"
                audio_processor.record_audio(end_audio, input_device)
                
                if orb_controller:
                    orb_controller.set_state("processing")
                
                end_text = audio_processor.transcribe_with_whisper(end_audio)
                os.remove(end_audio)
                
                # Parse end time
                date_obj = time_parser.parse_date_string(end_text)
                time_obj = time_parser.parse_time_string(end_text)
                
                if date_obj is None:
                    # Use start time's date
                    date_obj = start_time.date()
                
                if time_obj:
                    end_time = datetime.combine(date_obj, time_obj)
                else:
                    error_msg = "I couldn't understand the end time. Event not created."
                    audio_processor.process_and_play(error_msg, speaker_sample_path, output_device, orb_controller)
                    if orb_controller:
                        orb_controller.set_state("idle")
                    continue
            
            # Create the event
            event = calendar_service.create_event(title, description, start_time, end_time)
            
            if event:
                start_str = start_time.strftime('%I:%M %p')
                end_str = end_time.strftime('%I:%M %p')
                confirm_msg = f"Event '{title}' created from {start_str} to {end_str}."
            else:
                confirm_msg = "Ugh, something went wrong. The event wasn't created."
            
            audio_processor.process_and_play(confirm_msg, speaker_sample_path, output_device, orb_controller)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue
        
        # Handle list events command
        elif m_list_events:
            print("DEBUG: Processing 'list events' command...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            duration_text = m_list_events.group(1).strip()
            
            # If no duration specified, ask for it
            if not duration_text:
                prompt = "For how long? Tell me in days, hours, or minutes."
                audio_processor.process_and_play(prompt, speaker_sample_path, output_device, orb_controller)
                
                if orb_controller:
                    orb_controller.set_state("listening")
                
                duration_audio = "list_duration.wav"
                audio_processor.record_audio(duration_audio, input_device)
                
                if orb_controller:
                    orb_controller.set_state("processing")
                
                duration_text = audio_processor.transcribe_with_whisper(duration_audio)
                os.remove(duration_audio)
            
            # Parse duration with simple parser
            duration_hours = duration_parser.parse_duration(duration_text)
            
            if duration_hours is None:
                error_msg = "I couldn't understand the duration. Try again."
                audio_processor.process_and_play(error_msg, speaker_sample_path, output_device, orb_controller)
                if orb_controller:
                    orb_controller.set_state("idle")
                continue
            
            # Get events
            events = calendar_service.list_events(duration_hours)
            
            if events:
                summary = f"You have {len(events)} event{'s' if len(events) > 1 else ''} in the next {duration_hours} hours. "
                
                # List each event
                for i, event in enumerate(events[:5], 1):
                    event_title = event.get('summary', 'Untitled')
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    # Parse and format time
                    from datetime import datetime
                    try:
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        time_str = start_dt.strftime('%I:%M %p')
                    except:
                        time_str = start
                    
                    summary += f"Event {i}: {event_title} at {time_str}. "
                
                if len(events) > 5:
                    summary += f"And {len(events) - 5} more events."
            else:
                summary = f"You have no events in the next {duration_hours} hours."
            
            print(f"DEBUG: Events summary: {summary}")
            audio_processor.process_and_play(summary, speaker_sample_path, output_device, orb_controller)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue
        
        # Handle search event command
        elif m_search_event:
            print("DEBUG: Processing 'search event' command...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            search_text = m_search_event.group(1).strip()
            
            # If no search query, ask for it
            if not search_text:
                prompt = "What event are you looking for?"
                audio_processor.process_and_play(prompt, speaker_sample_path, output_device, orb_controller)
                
                if orb_controller:
                    orb_controller.set_state("listening")
                
                search_audio = "event_search.wav"
                audio_processor.record_audio(search_audio, input_device)
                
                if orb_controller:
                    orb_controller.set_state("processing")
                
                search_text = audio_processor.transcribe_with_whisper(search_audio)
                os.remove(search_audio)
            
            # Parse duration and start date from search query using LLM
            duration_hours, start_date_str = event_parser.parse_search_duration(search_text)
            
            # Default to 24 hours if no duration found
            if duration_hours is None:
                duration_hours = 24
                print(f"DEBUG: No duration in search query, defaulting to 24 hours")
            
            # Default to today if no start date provided
            if start_date_str is None:
                start_date_str = datetime.now().strftime('%d-%m-%Y')
                print(f"DEBUG: No start date in search query, defaulting to today: {start_date_str}")
            
            # Extract search keywords (remove time-related words)
            search_query = re.sub(r'\b(today|tomorrow|yesterday|next|this|week|day|hour|minute|on|at)\b', '', search_text, flags=re.IGNORECASE)
            search_query = search_query.strip()
            
            if not search_query:
                search_query = search_text
            
            print(f"DEBUG: Searching events for '{search_query}' starting from {start_date_str} for {duration_hours} hours")
            
            # Search events
            events = calendar_service.search_events(duration_hours, search_query, start_date_str)
            
            if events:
                summary = f"Found {len(events)} event{'s' if len(events) > 1 else ''} about {search_query}. "
                
                # Mention top 3 events
                for i, event in enumerate(events[:3], 1):
                    event_title = event.get('summary', 'Untitled')
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    # Parse and format time
                    from datetime import datetime
                    try:
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        time_str = start_dt.strftime('%I:%M %p')
                    except:
                        time_str = start
                    
                    summary += f"Event {i}: {event_title} at {time_str}. "
                
                if len(events) > 3:
                    summary += f"And {len(events) - 3} more."
            else:
                summary = f"I didn't find any events about {search_query}."
            
            print(f"DEBUG: Event search summary: {summary}")
            audio_processor.process_and_play(summary, speaker_sample_path, output_device, orb_controller)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue
        
        # Handle create note command
        elif m_create_note:
            print("DEBUG: Processing 'create note' command...")
            if orb_controller:
                orb_controller.set_state("processing")
                
            note_part = m_create_note.group(1).strip()
            # Note: llm_tokenizer and llm_model can be None if using API
            parsed_notes = note_manager.parse_note_creation(note_part, llm_tokenizer, llm_model)
            title = parsed_notes['Title']
            note_text = parsed_notes['Note']
            duration_seconds = parsed_notes['Duration']

            # Handle duration if None
            if duration_seconds is None:
                print("DEBUG: No duration specified in initial parse. Asking user for duration.")
                
                # Set to talking state for TTS
                # if orb_controller:
                #     orb_controller.set_state("talking")
                    
                duration_prompt = "Ugh, fine, how long do you want this note to last? Say something like '3 days' or 'forever' if no expiration."
                audio_processor.process_and_play(duration_prompt, speaker_sample_path, output_device, orb_controller)

                # Set to listening state for recording
                if orb_controller:
                    orb_controller.set_state("listening")
                    
                duration_audio = "duration_response.wav"
                audio_processor.record_audio(duration_audio, input_device)
                
                # Set to processing while transcribing
                if orb_controller:
                    orb_controller.set_state("processing")
                    
                duration_response = audio_processor.transcribe_with_whisper(duration_audio)
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
            
            # if orb_controller:
            #     orb_controller.set_state("talking")
                
            audio_processor.process_and_play(confirm_msg, speaker_sample_path, output_device, orb_controller)
            
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
        
        chat_response = rag_llm_processor.chatgpt_streamed(user_input, system_message, conversation_history, bot_name, threshold=0.50)
        conversation_history.append({"role": "assistant", "content": chat_response})
        
        # Set to talking state during TTS
        # if orb_controller:
        #     orb_controller.set_state("talking")

        audio_processor.process_and_play(chat_response, speaker_sample_path, output_device, orb_controller)

        # Return to idle
        if orb_controller:
            orb_controller.set_state("idle")
            
        # Keep only the last 20 messages for context
        conversation_history = conversation_history[-20:]
        print("DEBUG: Conversation history trimmed to last 20 messages.")
    
    # Cleanup
    print("DEBUG: Shutting down...")
    if orb_controller:
        orb_controller.stop_ui()

if __name__ == "__main__":
    main()
