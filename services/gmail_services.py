"""
Gmail Services for Desktop Assistant
Handles sending and reading emails using Gmail API
"""

import os
import base64
import json
import difflib
from datetime import datetime
from email.mime.text import MIMEText
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

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.send', 
          'https://www.googleapis.com/auth/gmail.readonly']

class GmailService:
    def __init__(self, credentials_file='client_secrets.json', token_file='gmail_token.json', contacts_file='user_data/contacts.json'):
        """
        Initialize Gmail service with OAuth2 authentication.
        
        Args:
            credentials_file: Path to OAuth2 client secrets
            token_file: Path to store/load access tokens
            contacts_file: Path to contacts JSON file
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.contacts_file = contacts_file
        self.service = None
        self.contacts = self.load_contacts()
        
    def load_contacts(self):
        """Load contacts from JSON file."""
        if os.path.exists(self.contacts_file):
            try:
                with open(self.contacts_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"{YELLOW}WARNING: Could not load contacts: {e}{RESET_COLOR}")
                return {}
        return {}
    
    def save_contacts(self):
        """Save contacts to JSON file."""
        os.makedirs(os.path.dirname(self.contacts_file), exist_ok=True)
        try:
            with open(self.contacts_file, 'w', encoding='utf-8') as f:
                json.dump(self.contacts, f, indent=2, ensure_ascii=False)
            print(f"{NEON_GREEN}Contacts saved successfully.{RESET_COLOR}")
        except Exception as e:
            print(f"{YELLOW}WARNING: Could not save contacts: {e}{RESET_COLOR}")
    
    def add_contact(self, name, email):
        """Add or update a contact."""
        self.contacts[name.lower()] = email
        self.save_contacts()
        print(f"{NEON_GREEN}Contact '{name}' added with email: {email}{RESET_COLOR}")
    
    def get_email_from_name(self, name, threshold=0.9):
        """Get email address from contact name with fuzzy matching."""
        name_lower = name.lower()
        
        # Exact match first
        if name_lower in self.contacts:
            return self.contacts[name_lower], name_lower, 1.0
        
        # Fuzzy match - find similar names
        best_match = None
        best_similarity = 0.0
        
        for contact_name in self.contacts.keys():
            similarity = difflib.SequenceMatcher(None, name_lower, contact_name).ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = contact_name
        
        # Return match if above threshold
        if best_match and best_similarity >= threshold:
            return self.contacts[best_match], best_match, best_similarity
        
        # No match found
        return None, None, 0.0
    
    def authenticate(self):
        """Authenticate with Gmail API using OAuth2."""
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
        
        self.service = build('gmail', 'v1', credentials=creds)
        return self.service
    
    def send_email(self, to_email, subject, body):
        """
        Send an email via Gmail API.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body (plain text)
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.service:
            self.authenticate()
        
        try:
            message = MIMEText(body)
            message['to'] = to_email
            message['subject'] = subject
            
            # Encode the message
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            send_message = {'raw': raw_message}
            
            # Send the message
            result = self.service.users().messages().send(
                userId='me', body=send_message).execute()
            
            print(f"{NEON_GREEN}Email sent successfully! Message ID: {result['id']}{RESET_COLOR}")
            return True
            
        except HttpError as error:
            print(f"{YELLOW}ERROR: Failed to send email: {error}{RESET_COLOR}")
            return False
        except Exception as e:
            print(f"{YELLOW}ERROR: Unexpected error sending email: {e}{RESET_COLOR}")
            return False
    
    def get_recent_emails(self, max_results=50, query=''):
        """
        Get recent emails from Gmail.
        
        Args:
            max_results: Maximum number of emails to retrieve
            query: Gmail search query (optional)
            
        Returns:
            list: List of email dictionaries with id, subject, from, date, snippet, body
        """
        if not self.service:
            self.authenticate()
        
        try:
            # Get list of messages
            results = self.service.users().messages().list(
                userId='me', maxResults=max_results, q=query).execute()
            messages = results.get('messages', [])
            
            if not messages:
                print(f"{YELLOW}No messages found.{RESET_COLOR}")
                return []
            
            emails = []
            print(f"DEBUG: Fetching {len(messages)} emails...")
            
            for msg in messages:
                # Get full message details
                message = self.service.users().messages().get(
                    userId='me', id=msg['id'], format='full').execute()
                
                # Extract headers
                headers = message['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
                from_email = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown')
                date = next((h['value'] for h in headers if h['name'].lower() == 'date'), 'Unknown')
                
                # Extract body
                body = self._get_email_body(message['payload'])
                snippet = message.get('snippet', '')
                
                emails.append({
                    'id': msg['id'],
                    'subject': subject,
                    'from': from_email,
                    'date': date,
                    'snippet': snippet,
                    'body': body
                })
            
            print(f"{NEON_GREEN}Retrieved {len(emails)} emails.{RESET_COLOR}")
            return emails
            
        except HttpError as error:
            print(f"{YELLOW}ERROR: Failed to get emails: {error}{RESET_COLOR}")
            return []
        except Exception as e:
            print(f"{YELLOW}ERROR: Unexpected error getting emails: {e}{RESET_COLOR}")
            return []
    
    def _get_email_body(self, payload):
        """Extract email body from payload."""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
                elif part['mimeType'] == 'multipart/alternative' and 'parts' in part:
                    # Recursive search in multipart
                    body = self._get_email_body(part)
                    if body:
                        break
        elif 'body' in payload and 'data' in payload['body']:
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        return body
    
    def search_emails_for_content(self, emails, search_query):
        """
        Search through emails for specific content.
        
        Args:
            emails: List of email dictionaries
            search_query: What to search for
            
        Returns:
            list: Filtered emails that match the query
        """
        search_query_lower = search_query.lower()
        matching_emails = []
        
        for email in emails:
            # Search in subject, body, and snippet
            if (search_query_lower in email['subject'].lower() or
                search_query_lower in email['body'].lower() or
                search_query_lower in email['snippet'].lower()):
                matching_emails.append(email)
        
        return matching_emails


class EmailParser:
    """Parse email creation commands using LLM."""
    
    def __init__(self, llm_interface):
        """
        Initialize email parser.
        
        Args:
            llm_interface: LLMInterface instance for parsing
        """
        self.llm_interface = llm_interface
    
    def parse_email_creation(self, user_input):
        """
        Parse email creation command using LLM.
        
        Args:
            user_input: User's email command text
            
        Returns:
            dict: {'to': email/name, 'subject': str, 'body': str}
        """
        print(f"DEBUG: Parsing email creation input: {user_input[:50]}... (truncated)")
        
        system_prompt = """You are an email parser. Extract the recipient (email or name), subject, and body from the user's input.
If the subject is not explicitly mentioned, generate a brief appropriate subject based on the body content.
Return ONLY a valid JSON object with this exact format:

Output requirements (CRITICAL):
    - Output **ONLY** a single valid JSON object (no surrounding text, no explanation, no backticks, no code fences).
    - This should be the JSON format: {"to": "recipient email or name", "subject": "email subject", "body": "email body text"}
    - Use ISO formatting for no special tokens.
    - If name is mentioned instead of email, return the name as it is in the "to" field.

Examples:
Input: "send email to john@example.com subject meeting reminder body don't forget our meeting tomorrow at 3pm"
Output: {"to": "john@example.com", "subject": "meeting reminder", "body": "don't forget our meeting tomorrow at 3pm"}

Input: "send email alice saying I'll be late to the office today"
Output: {"to": "alice", "subject": "Running Late", "body": "I'll be late to the office today"}

Input: "send email to soham that the project is on track and we'll finish by Friday"
Output: {"to": "soham", "subject": "Project Update", "body": "the project is on track and we'll finish by Friday"}"""

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
                'to': parsed.get('to', ''),
                'subject': parsed.get('subject', 'No Subject'),
                'body': parsed.get('body', '')
            }
            print(f"DEBUG: Parsed email: {result}")
            return result
        except json.JSONDecodeError as e:
            print(f"{YELLOW}WARNING: JSON parse error: {e}{RESET_COLOR}")
            print(f"{YELLOW}Returning empty email structure.{RESET_COLOR}")
            return {'to': '', 'subject': 'No Subject', 'body': ''}
