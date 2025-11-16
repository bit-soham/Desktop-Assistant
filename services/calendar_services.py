"""
Google Calendar Services for Desktop Assistant
Handles creating, listing, and searching calendar events using Google Calendar API
"""

import os
import json
import re
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Calendar API scopes
SCOPES = ['https://www.googleapis.com/auth/calendar']


class CalendarService:
    def __init__(self, credentials_file='client_secrets.json', token_file='calendar_token.json'):
        """
        Initialize Calendar service with OAuth2 authentication.
        
        Args:
            credentials_file: Path to OAuth2 client secrets
            token_file: Path to store/load access tokens
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        
    def authenticate(self):
        """Authenticate with Google Calendar API using OAuth2."""
        creds = None
        
        # Load existing token if available
        if os.path.exists(self.token_file):
            try:
                creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
            except Exception as e:
                print(f"{YELLOW}WARNING: Could not load token: {e}{RESET_COLOR}")
        
        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("DEBUG: Refreshing expired token...")
                creds.refresh(Request())
            else:
                print("DEBUG: Starting OAuth2 flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
            print(f"{NEON_GREEN}Authentication successful!{RESET_COLOR}")
        
        self.service = build('calendar', 'v3', credentials=creds)
        return self.service
    
    def create_event(self, title, description, start_time, end_time):
        """
        Create a calendar event.
        
        Args:
            title: Event title/summary
            description: Event description/content
            start_time: Event start time (datetime object)
            end_time: Event end time (datetime object)
            
        Returns:
            dict: Created event details or None if failed
        """
        if not self.service:
            self.authenticate()
        
        try:
            # Validate that start_time and end_time are provided
            if start_time is None:
                print(f"{YELLOW}ERROR: Start time is required{RESET_COLOR}")
                return None
            
            if end_time is None:
                print(f"{YELLOW}ERROR: End time is required{RESET_COLOR}")
                return None
            
            # Validate that end_time is after start_time
            if end_time <= start_time:
                print(f"{YELLOW}ERROR: End time must be after start time{RESET_COLOR}")
                print(f"{YELLOW}Start: {start_time}, End: {end_time}{RESET_COLOR}")
                return None
            
            # Convert timezone-naive datetimes to UTC
            if start_time.tzinfo is None:
                # Assume local time, convert to UTC
                from datetime import timezone
                start_time = start_time.replace(tzinfo=timezone.utc)
            
            if end_time.tzinfo is None:
                from datetime import timezone
                end_time = end_time.replace(tzinfo=timezone.utc)
            
            event = {
                'summary': title,
                'description': description,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                },
            }
            
            print(f"DEBUG: Creating event with start={start_time.isoformat()}, end={end_time.isoformat()}")
            
            created_event = self.service.events().insert(
                calendarId='primary', body=event).execute()
            
            print(f"{NEON_GREEN}Event created: {created_event.get('htmlLink')}{RESET_COLOR}")
            return created_event
            
        except HttpError as error:
            print(f"{YELLOW}ERROR: Failed to create event: {error}{RESET_COLOR}")
            return None
        except Exception as e:
            print(f"{YELLOW}ERROR: Unexpected error creating event: {e}{RESET_COLOR}")
            import traceback
            traceback.print_exc()
            return None
    
    def list_events(self, duration_hours):
        """
        List events for a specified duration from now.
        
        Args:
            duration_hours: Duration to check in hours from now
            
        Returns:
            list: List of event dictionaries
        """
        if not self.service:
            self.authenticate()
        
        try:
            # Calculate time range
            now = datetime.utcnow()
            time_max = now + timedelta(hours=duration_hours)
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now.isoformat() + 'Z',
                timeMax=time_max.isoformat() + 'Z',
                maxResults=50,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                print(f"{YELLOW}No events found in the next {duration_hours} hours.{RESET_COLOR}")
                return []
            
            print(f"{NEON_GREEN}Found {len(events)} event(s) in the next {duration_hours} hours.{RESET_COLOR}")
            return events
            
        except HttpError as error:
            print(f"{YELLOW}ERROR: Failed to list events: {error}{RESET_COLOR}")
            return []
        except Exception as e:
            print(f"{YELLOW}ERROR: Unexpected error listing events: {e}{RESET_COLOR}")
            return []
    
    def search_events(self, duration_hours, query, start_date_str=None, similarity_threshold=0.4):
        """
        Search for events matching a query within a duration using similarity search.
        
        Args:
            duration_hours: Duration to search in hours from start date
            query: Search query string
            start_date_str: Start date in DD-MM-YYYY format or None (defaults to today)
            similarity_threshold: Minimum similarity score (0.0 to 1.0), default 0.6
            
        Returns:
            list: List of matching event dictionaries sorted by similarity
        """
        if not self.service:
            self.authenticate()
        
        try:
            # Parse start date or use today
            if start_date_str:
                time_parser = TimeParser()
                start_date = time_parser.parse_date_string(start_date_str)
                if start_date is None:
                    print(f"{YELLOW}WARNING: Could not parse start date '{start_date_str}', using today{RESET_COLOR}")
                    start_date = datetime.utcnow().date()
            else:
                start_date = datetime.utcnow().date()
                print(f"DEBUG: No start date provided, using today: {start_date}")
            
            # Create start datetime at beginning of day (00:00)
            start_datetime = datetime.combine(start_date, datetime.min.time())
            # Make it UTC-aware
            from datetime import timezone
            start_datetime = start_datetime.replace(tzinfo=timezone.utc)
            
            # Calculate end datetime
            end_datetime = start_datetime + timedelta(hours=duration_hours)
            
            print(f"DEBUG: Searching events from {start_datetime.isoformat()} to {end_datetime.isoformat()}")
            
            # Get all events in the time range
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=start_datetime.isoformat(),
                timeMax=end_datetime.isoformat(),
                maxResults=50,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            all_events = events_result.get('items', [])
            
            if not all_events:
                print(f"{YELLOW}No events found in the next {duration_hours} hours.{RESET_COLOR}")
                return []
            
            # Import sentence transformers for similarity
            try:
                from sentence_transformers import SentenceTransformer, util
                import torch
            except ImportError:
                print(f"{YELLOW}ERROR: sentence-transformers not installed. Falling back to simple text search.{RESET_COLOR}")
                # Fallback to simple text search
                matching_events = [
                    event for event in all_events
                    if query.lower() in event.get('summary', '').lower() or 
                       query.lower() in event.get('description', '').lower()
                ]
                print(f"{NEON_GREEN}Found {len(matching_events)} event(s) matching '{query}'.{RESET_COLOR}")
                return matching_events
            
            # Load embedding model (use a lightweight model)
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode the query
            query_embedding = model.encode(query, convert_to_tensor=True)
            
            # Calculate similarity for each event
            event_scores = []
            for event in all_events:
                title = event.get('summary', '')
                description = event.get('description', '')
                
                # Combine title and description for matching
                event_text = f"{title} {description}".strip()
                
                if not event_text:
                    continue
                
                # Encode event text
                event_embedding = model.encode(event_text, convert_to_tensor=True)
                
                # Calculate cosine similarity
                similarity = util.cos_sim(query_embedding, event_embedding).item()
                
                if similarity >= similarity_threshold:
                    event_scores.append((event, similarity))
            
            # Sort by similarity (highest first)
            event_scores.sort(key=lambda x: x[1], reverse=True)
            matching_events = [event for event, score in event_scores]
            
            if not matching_events:
                print(f"{YELLOW}No events found matching '{query}' with similarity >= {similarity_threshold*100:.0f}% in the next {duration_hours} hours.{RESET_COLOR}")
                return []
            
            print(f"{NEON_GREEN}Found {len(matching_events)} event(s) matching '{query}' with similarity >= {similarity_threshold*100:.0f}%.{RESET_COLOR}")
            for event, score in event_scores[:3]:  # Show top 3 scores
                print(f"  - {event.get('summary', 'Untitled')}: {score*100:.1f}% similarity")
            
            return matching_events
            
        except HttpError as error:
            print(f"{YELLOW}ERROR: Failed to search events: {error}{RESET_COLOR}")
            return []
        except Exception as e:
            print(f"{YELLOW}ERROR: Unexpected error searching events: {e}{RESET_COLOR}")
            import traceback
            traceback.print_exc()
            return []


class DurationParser:
    """Parse duration from natural language without using LLM."""
    
    @staticmethod
    def parse_duration(text):
        """
        Parse duration from text like 'next 3 hours', '2 days', '30 minutes', etc.
        
        Args:
            text: Input text containing duration
            
        Returns:
            float: Duration in hours, or None if parsing failed
        """
        text = text.lower().strip()
        
        # Patterns for time units
        patterns = {
            'days': r'(\d+)\s*(?:day|days)',
            'hours': r'(\d+)\s*(?:hour|hours|hr|hrs|h)',
            'minutes': r'(\d+)\s*(?:minute|minutes|min|mins|m)',
            'weeks': r'(\d+)\s*(?:week|weeks)',
        }
        
        total_hours = 0.0
        found_any = False
        
        # Check for each time unit
        for unit, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                value = int(match.group(1))
                found_any = True
                
                # Convert to hours
                if unit == 'days':
                    total_hours += value * 24
                elif unit == 'hours':
                    total_hours += value
                elif unit == 'minutes':
                    total_hours += value / 60
                elif unit == 'weeks':
                    total_hours += value * 24 * 7
        
        # Special cases
        if 'today' in text:
            # From now until end of day
            now = datetime.now()
            end_of_day = datetime.combine(now.date(), datetime.max.time())
            hours_left = (end_of_day - now).total_seconds() / 3600
            return hours_left
        elif 'tomorrow' in text:
            return 24
        elif 'this week' in text:
            return 168  # 7 days
        elif 'next week' in text:
            return 168
        
        if found_any:
            return total_hours
        else:
            return None


class TimeParser:
    """Parse start and end times from natural language without using LLM."""
    
    @staticmethod
    def parse_time_string(time_str):
        """
        Parse time string like '3pm', '14:30', '3:30 PM', 'HH:MM', etc.
        
        Args:
            time_str: Time string
            
        Returns:
            datetime.time object or None if parsing failed
        """
        time_str = time_str.strip().lower()
        
        # Try different time formats
        time_formats = [
            r'(\d{1,2}):(\d{2})',  # HH:MM or H:MM (24-hour format, priority)
            r'(\d{1,2})\s*(?::|\.)\s*(\d{2})\s*(am|pm)?',  # 3:30 PM, 14:30
            r'(\d{1,2})\s*(am|pm)',  # 3pm, 3 PM
        ]
        
        for pattern in time_formats:
            match = re.search(pattern, time_str)
            if match:
                groups = match.groups()
                hour = int(groups[0])
                minute = int(groups[1]) if len(groups) > 1 and groups[1] and groups[1].isdigit() else 0
                meridiem = groups[-1] if len(groups) > 2 and groups[-1] in ['am', 'pm'] else None
                
                # Convert to 24-hour format if meridiem present
                if meridiem == 'pm' and hour != 12:
                    hour += 12
                elif meridiem == 'am' and hour == 12:
                    hour = 0
                
                try:
                    from datetime import time
                    return time(hour=hour, minute=minute)
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def parse_date_string(date_str):
        """
        Parse date string like 'today', 'tomorrow', '2025-11-20', 'DD-MM-YYYY', etc.
        
        Args:
            date_str: Date string
            
        Returns:
            datetime.date object or None if parsing failed
        """
        date_str = date_str.strip().lower()
        now = datetime.now()
        
        if 'today' in date_str:
            return now.date()
        elif 'tomorrow' in date_str:
            return (now + timedelta(days=1)).date()
        elif 'yesterday' in date_str:
            return (now - timedelta(days=1)).date()
        
        # Try to parse date formats
        date_formats = [
            '%d-%m-%Y',  # DD-MM-YYYY (priority for LLM output)
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%B %d',  # November 20
            '%b %d',  # Nov 20
        ]
        
        for fmt in date_formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                # If year not provided, use current year
                if fmt in ['%B %d', '%b %d']:
                    parsed = parsed.replace(year=now.year)
                return parsed.date()
            except ValueError:
                continue
        
        return None


class EventParser:
    """Parse event creation commands using LLM."""
    
    def __init__(self, llm_interface):
        """
        Initialize event parser.
        
        Args:
            llm_interface: LLMInterface instance for parsing
        """
        self.llm_interface = llm_interface
    
    def parse_event_creation(self, user_input):
        """
        Parse event creation command using LLM.
        
        Args:
            user_input: User's event command text
            
        Returns:
            dict: {'title': str, 'description': str, 'date': str or None, 'start_time': str or None, 'end_time': str or None}
        """
        print(f"DEBUG: Parsing event creation input: {user_input[:50]}... (truncated)")
        
        # Get current date for context
        today = datetime.now()
        today_str = datetime.now().strftime('%d-%m-%Y')  # e.g., "16-11-2025"
        
        system_prompt = f"""You are an event parser. Today is {today_str}.
Extract the event title, description/content, date, start time, and end time from the user's input.
Dates should be in DD-MM-YYYY format. If user says "today" use {today_str}, "tomorrow" use next day, etc.
Times MUST be in 24-hour format HH:MM (e.g., "14:30", "09:00", "18:45").
Convert any AM/PM times to 24-hour format: 3pm = "15:00", 10am = "10:00", etc.
If date, start time, or end time is not mentioned, set them to null.

Return ONLY a valid JSON object with this exact format:
{{"title": "event title", "description": "event description", "date": "DD-MM-YYYY", "start_time": "HH:MM", "end_time": "HH:MM"}}

Examples:
Input: "meeting with team from 2pm to 4pm about project updates"
Output: {{"title": "Meeting with Team", "description": "Discuss project updates", "date": "{today_str}", "start_time": "14:00", "end_time": "16:00"}}

Input: "doctor appointment tomorrow at 3pm for 1 hour"
Output: {{"title": "Doctor Appointment", "description": "Doctor appointment", "date": "{(today + timedelta(days=1)).strftime('%d-%m-%Y')}", "start_time": "15:00", "end_time": "16:00"}}

Input: "lunch with john on November 20 from 12:30 to 1:30"
Output: {{"title": "Lunch with John", "description": "Lunch meeting", "date": "20-11-2025", "start_time": "12:30", "end_time": "13:30"}}

Input: "team standup at 10am"
Output: {{"title": "Team Standup", "description": "Daily standup meeting", "date": "{today_str}", "start_time": "10:00", "end_time": null}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = self.llm_interface.generate(messages, max_new_tokens=150, temperature=0.3)
        print(f"DEBUG: Raw LLM response: {response[:100]}... (truncated)")
        
        # Extract JSON from response
        response = response.strip()
        
        # Remove thinking tags if present
        if "<think>" in response:
            think_end = response.find("</think>")
            if think_end != -1:
                response = response[think_end + 8:].strip()
        
        # Remove markdown code blocks if present
        response = response.replace("```json", "").replace("```", "").strip()
        
        print(f"[DEBUG] response before:  {response[:100]}...")
        
        # Find JSON object
        if '{' in response and '}' in response:
            start = response.find('{')
            end = response.rfind('}') + 1
            response = response[start:end]
        
        print(f"[DEBUG] response after:  {response[:100]}...")
        
        try:
            parsed = json.loads(response)
            result = {
                'title': parsed.get('title', 'Untitled Event'),
                'description': parsed.get('description', ''),
                'date': parsed.get('date'),  # Can be None
                'start_time': parsed.get('start_time'),  # Can be None
                'end_time': parsed.get('end_time')  # Can be None
            }
            print(f"DEBUG: Parsed event: {result}")
            return result
        except json.JSONDecodeError as e:
            print(f"{YELLOW}WARNING: JSON parse error: {e}{RESET_COLOR}")
            print(f"{YELLOW}Returning empty event structure.{RESET_COLOR}")
            return {'title': 'Untitled Event', 'description': '', 'date': None, 'start_time': None, 'end_time': None}
    
    def parse_search_duration(self, user_input):
        """
        Parse duration and start date from search query using LLM.
        
        Args:
            user_input: User's search query
            
        Returns:
            tuple: (duration_hours, start_date_str) or (None, None) if parsing failed
                   duration_hours: float or None
                   start_date_str: str in DD-MM-YYYY format or None
        """
        print(f"DEBUG: Parsing search duration and start: {user_input[:50]}...")
        
        # Get current date for context
        today = datetime.now()
        today_str = today.strftime('%d-%m-%Y')
        tomorrow_str = (today + timedelta(days=1)).strftime('%d-%m-%Y')
        yesterday_str = (today - timedelta(days=1)).strftime('%d-%m-%Y')
        
        system_prompt = f"""You are a time parser. Today is {today_str}.
Extract the time duration and start date from the search query.
Convert duration to hours.
Start date MUST be in DD-MM-YYYY format.
If user says "today" use {today_str}, "tomorrow" use {tomorrow_str}, "yesterday" use {yesterday_str}.
If not mentioned, use null.

Return ONLY a valid JSON object:
{{"duration": hours_as_number_or_null, "start_date": "DD-MM-YYYY_or_null"}}

Examples:
Input: "do I have any meetings today"
Output: {{"duration": 24, "start_date": "{today_str}"}}

Input: "what's on my calendar for the next 3 hours"
Output: {{"duration": 3, "start_date": "{today_str}"}}

Input: "any events tomorrow"
Output: {{"duration": 24, "start_date": "{tomorrow_str}"}}

Input: "meetings this week"
Output: {{"duration": 168, "start_date": "{today_str}"}}

Input: "events on November 20"
Output: {{"duration": 24, "start_date": "20-11-2025"}}

Input: "do I have any appointments"
Output: {{"duration": null, "start_date": null}}

Input: "meetings yesterday"
Output: {{"duration": 24, "start_date": "{yesterday_str}"}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = self.llm_interface.generate(messages, max_new_tokens=80, temperature=0.3)
        
        # Clean response
        response = response.strip()
        if "<think>" in response:
            think_end = response.find("</think>")
            if think_end != -1:
                response = response[think_end + 8:].strip()
        response = response.replace("```json", "").replace("```", "").strip()
        
        if '{' in response and '}' in response:
            start = response.find('{')
            end = response.rfind('}') + 1
            response = response[start:end]
        
        try:
            parsed = json.loads(response)
            duration = parsed.get('duration')
            start_date = parsed.get('start_date')
            
        
            if start_date is None:
                start_date = today_str
                print(f"DEBUG: Start date was null, setting to today ({today_str})")
            
            print(f"DEBUG: Parsed duration={duration} hours, start_date={start_date}")
            return (duration, start_date)
        except Exception as e:
            print(f"{YELLOW}WARNING: Failed to parse search parameters: {e}{RESET_COLOR}")
            return (None, None)
