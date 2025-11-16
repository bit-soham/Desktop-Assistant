"""
Desktop Assistant with LangChain-based Command Routing

This version uses LangChain and LLM for intelligent command classification instead of regex.
The CommandRouter analyzes user intent and routes to appropriate handlers.

Commands supported:
- create_note: Create a new note
- delete_note: Delete an existing note
- list_notes: List all notes
- send_email: Send an email
- check_gmail: Check Gmail for emails
- create_event: Create a calendar event
- list_events: List calendar events
- search_event: Search for calendar events
- conversation: General conversation with RAG/LLM
- exit: Exit the application

The system intelligently determines whether the user wants to execute a command
or have a conversation, without relying on exact command patterns.
"""
import argparse
import os
import re
from datetime import datetime, timedelta
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

# Import LangChain-based command router
from core.command_router import CommandRouter

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
    
    # Initialize LangChain-based command router
    command_router = CommandRouter(llm_interface)

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
        
        # Use LangChain-based command router for intent classification
        classification = command_router.classify_intent(user_input)
        command = classification['command']
        parameters = classification['parameters']
        
        print(f"DEBUG: Intent classification - Command: {command}, Confidence: {classification['confidence']:.2f}")
        
        if command == 'exit':
            print("DEBUG: Exit command detected. Breaking loop.")
            if orb_controller:
                orb_controller.set_state("idle")
            break
            
        # Handle list notes command
        elif command == 'list_notes':
            print("DEBUG: Processing 'list notes' command...")
            note_manager.list_notes()
            
            # Brief talking animation for confirmation
            if orb_controller:
                orb_controller.set_state("talking")
                time.sleep(1)
                orb_controller.set_state("idle")
            continue
            
        # Handle send email command
        elif command == 'send_email':
            print("DEBUG: Processing 'send email' command...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            # Use full user input - LLM parser handles natural language
            email_part = user_input
            
            # Basic check if input seems too short
            if len(email_part.strip()) < 5:
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
        elif command == 'check_gmail':
            print("DEBUG: Processing 'check gmail' command...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            # Use full user input - will extract query from natural language
            search_query = user_input
            
            # Basic check if input seems too short or generic
            if len(search_query.strip()) < 5 or search_query.lower().strip() in ['check gmail', 'check email', 'gmail', 'email']:
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
        elif command == 'create_event':
            print("DEBUG: Processing 'create event' command...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            # Use full user input - LLM parser handles natural language like "make an event for gym at 3pm"
            event_part = user_input
            
            # Basic check if input seems too short
            if len(event_part.strip()) < 5:
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
            
            # If start time not provided, ask for it in terminal
            if start_time is None:
                print(f"\n{YELLOW}Start time is required. Please enter start time (e.g., '19:00' or '7pm'):{RESET_COLOR}")
                start_input = input("Start time: ").strip()
                
                time_obj = time_parser.parse_time_string(start_input)
                if time_obj:
                    start_time = datetime.combine(event_date, time_obj)
                else:
                    error_msg = "I couldn't understand the start time. Event not created."
                    audio_processor.process_and_play(error_msg, speaker_sample_path, output_device, orb_controller)
                    if orb_controller:
                        orb_controller.set_state("idle")
                    continue
            
            # If end time not provided, ask for it in terminal
            if end_time is None:
                print(f"\n{YELLOW}End time is required. Please enter end time (e.g., '20:00' or '8pm'):{RESET_COLOR}")
                end_input = input("End time: ").strip()
                
                time_obj = time_parser.parse_time_string(end_input)
                if time_obj:
                    end_time = datetime.combine(event_date, time_obj)
                else:
                    error_msg = "I couldn't understand the end time. Event not created."
                    audio_processor.process_and_play(error_msg, speaker_sample_path, output_device, orb_controller)
                    if orb_controller:
                        orb_controller.set_state("idle")
                    continue
            
            # Create the event
            event = calendar_service.create_event(title, description, start_time, end_time)
            
            if event:
                confirm_msg = f"{title} event created successfully."
            else:
                confirm_msg = "Ugh, something went wrong. The event wasn't created."
            
            audio_processor.process_and_play(confirm_msg, speaker_sample_path, output_device, orb_controller)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue
        
        # Handle list events command
        elif command == 'list_events':
            print("DEBUG: Processing 'list events' command...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            # Use LLM to parse duration and start date from user input
            duration_hours, start_date_str = event_parser.parse_search_duration(user_input)
            
            # If no duration found, ask for it
            if duration_hours is None:
                # Try simple duration parser as fallback
                duration_hours = duration_parser.parse_duration(user_input)
                
                if duration_hours is None:
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
                    
                    duration_hours = duration_parser.parse_duration(duration_text)
                    
                    if duration_hours is None:
                        error_msg = "I couldn't understand the duration. Try again."
                        audio_processor.process_and_play(error_msg, speaker_sample_path, output_device, orb_controller)
                        if orb_controller:
                            orb_controller.set_state("idle")
                        continue
            
            # Parse start date (defaults to today if None)
            if start_date_str:
                start_date = time_parser.parse_date_string(start_date_str)
                if start_date is None:
                    start_date = datetime.now().date()
            else:
                start_date = datetime.now().date()
            
            print(f"DEBUG: Listing events from {start_date} for {duration_hours} hours")
            
            # Create start datetime at beginning of the specified date
            start_datetime = datetime.combine(start_date, datetime.min.time())
            from datetime import timezone
            start_datetime = start_datetime.replace(tzinfo=timezone.utc)
            end_datetime = start_datetime + timedelta(hours=duration_hours)
            
            # Get events from calendar
            events_result = calendar_service.service if calendar_service.service else calendar_service.authenticate()
            events_data = calendar_service.service.events().list(
                calendarId='primary',
                timeMin=start_datetime.isoformat(),
                timeMax=end_datetime.isoformat(),
                maxResults=50,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_data.get('items', [])
            
            if events:
                # Generate LLM response for events
                event_summaries = []
                for event in events[:10]:  # Limit to 10 events
                    event_title = event.get('summary', 'Untitled')
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    try:
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        time_str = start_dt.strftime('%I:%M %p on %B %d')
                    except:
                        time_str = start
                    event_summaries.append(f"{event_title} at {time_str}")
                
                # Use LLM to generate natural response
                llm_prompt = f"""The user asked: "{user_input}"

I found {len(events)} event(s):
{chr(10).join(f"{i+1}. {summary}" for i, summary in enumerate(event_summaries))}

Generate a natural, conversational response (2-3 sentences max) to answer the user's question. Be concise and friendly."""
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Provide brief, natural responses."},
                    {"role": "user", "content": llm_prompt}
                ]
                
                summary = llm_interface.generate(messages, max_new_tokens=150, temperature=0.7)
            else:
                summary = f"You have no events scheduled for that time period."
            
            print(f"DEBUG: Events summary: {summary}")
            audio_processor.process_and_play(summary, speaker_sample_path, output_device, orb_controller)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue
        
        # Handle search event command
        elif command == 'search_event':
            print("DEBUG: Processing 'search event' command...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            # Use full user input - will extract query from natural language
            search_text = user_input
            
            # Basic check if input seems too short
            if len(search_text.strip()) < 5:
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
            
            # Get extracted content from classifier (the search query without command keywords)
            search_query = parameters.get('content', search_text)
            
            # Clean up the search query further if needed
            search_query = re.sub(r'\b(today|tomorrow|yesterday|next|this|week|day|hour|minute|on|at|in|calendar)\b', '', search_query, flags=re.IGNORECASE)
            search_query = search_query.strip()
            
            if not search_query or len(search_query) < 2:
                search_query = search_text
            
            print(f"DEBUG: Searching events for '{search_query}' starting from {start_date_str} for {duration_hours} hours")
            
            # Search events using RAG similarity (60% threshold)
            events = calendar_service.search_events(duration_hours, search_query, start_date_str, similarity_threshold=0.6)
            
            if events:
                # Generate LLM response with event details
                event_summaries = []
                for event in events[:5]:  # Limit to top 5
                    event_title = event.get('summary', 'Untitled')
                    event_desc = event.get('description', '')
                    start = event['start'].get('dateTime', event['start'].get('date'))
                    try:
                        start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                        time_str = start_dt.strftime('%I:%M %p on %B %d')
                    except:
                        time_str = start
                    
                    event_info = f"{event_title} at {time_str}"
                    if event_desc:
                        event_info += f" - {event_desc[:50]}"
                    event_summaries.append(event_info)
                
                # Use LLM to generate natural response
                llm_prompt = f"""The user asked: "{user_input}"

I found {len(events)} matching event(s):
{chr(10).join(f"{i+1}. {summary}" for i, summary in enumerate(event_summaries))}

Generate a natural, conversational response (2-3 sentences max) to answer the user's question. Be concise and friendly."""
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Provide brief, natural responses."},
                    {"role": "user", "content": llm_prompt}
                ]
                
                summary = llm_interface.generate(messages, max_new_tokens=150, temperature=0.7)
            else:
                summary = f"I didn't find any events matching '{search_query}' for that time period."
            
            print(f"DEBUG: Event search summary: {summary}")
            audio_processor.process_and_play(summary, speaker_sample_path, output_device, orb_controller)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue
        
        # Handle create note command
        elif command == 'create_note':
            print("DEBUG: Processing 'create note' command...")
            if orb_controller:
                orb_controller.set_state("processing")
                
            # Use full user input - LLM parser handles natural language
            note_part = user_input
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
        elif command == 'delete_note':
            print("DEBUG: Processing 'delete note' command...")
            if orb_controller:
                orb_controller.set_state("processing")
                
            # Use full user input - extract title from natural language
            # For now, try to extract the title part
            title_match = re.search(r'(?:delete|remove|erase)\s+(?:note|the note)\s+(.+)', user_input, flags=re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
            else:
                # Fallback: use everything after common delete keywords
                title = re.sub(r'^.*?(?:delete|remove|erase)\s+', '', user_input, flags=re.IGNORECASE).strip()
            note_manager.manage_note("delete", title, None)
            
            if orb_controller:
                orb_controller.set_state("idle")
            continue

        # Handle conversation (normal query with RAG and LLM)
        elif command == 'conversation':
            print("DEBUG: Processing conversation with RAG and LLM...")
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
        
        # Unknown command - treat as conversation
        else:
            print(f"DEBUG: Unknown command '{command}', treating as conversation...")
            if orb_controller:
                orb_controller.set_state("processing")
            
            conversation_history.append({"role": "user", "content": user_input})
            print(PINK + f"{bot_name}: " + RESET_COLOR)
            
            chat_response = rag_llm_processor.chatgpt_streamed(user_input, system_message, conversation_history, bot_name, threshold=0.50)
            conversation_history.append({"role": "assistant", "content": chat_response})
            
            audio_processor.process_and_play(chat_response, speaker_sample_path, output_device, orb_controller)

            if orb_controller:
                orb_controller.set_state("idle")
                
            conversation_history = conversation_history[-20:]
    
    # Cleanup
    print("DEBUG: Shutting down...")
    if orb_controller:
        orb_controller.stop_ui()

if __name__ == "__main__":
    main()
