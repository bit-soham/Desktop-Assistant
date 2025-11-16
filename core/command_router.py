"""
LangChain-based command router for intelligent intent classification
"""
from typing import Dict, Any, Optional
import re
from datetime import datetime

class CommandRouter:
    """
    Routes user input to appropriate command handlers using LangChain for intent classification
    """
    
    def __init__(self, llm_interface):
        """
        Initialize the command router
        
        Args:
            llm_interface: LLM interface for intent classification
        """
        self.llm_interface = llm_interface
        
        # Define available commands
        self.commands = {
            'create_note': {
                'description': 'Create a new note with title and content',
                'examples': ['create note meeting notes', 'make a note about the project']
            },
            'delete_note': {
                'description': 'Delete an existing note by title',
                'examples': ['delete note meeting notes', 'remove the note about project']
            },
            'list_notes': {
                'description': 'List all existing notes',
                'examples': ['list notes', 'show me all notes', 'what notes do i have']
            },
            'send_email': {
                'description': 'Send an email to a recipient with subject and body',
                'examples': ['send email to john about meeting', 'email jane the report']
            },
            'check_gmail': {
                'description': 'Check Gmail for specific emails or meetings',
                'examples': ['check gmail for meetings', 'do i have any emails from john']
            },
            'create_event': {
                'description': 'Create a calendar event with title, description, and time',
                'examples': ['create event meeting tomorrow at 3pm', 'schedule a call with client']
            },
            'list_events': {
                'description': 'List upcoming calendar events',
                'examples': ['list events for next week', 'show my events today']
            },
            'search_event': {
                'description': 'Search for specific calendar events',
                'examples': ['search event meeting', 'find events about project']
            },
            'conversation': {
                'description': 'Have a general conversation or ask questions',
                'examples': ['how are you', 'tell me a joke', 'what is the weather']
            },
            'exit': {
                'description': 'Exit the application',
                'examples': ['exit', 'quit', 'goodbye', 'bye']
            }
        }
    
    def classify_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Classify user intent using LangChain/LLM
        
        Args:
            user_input: The user's transcribed input
            
        Returns:
            Dictionary with 'intent', 'command', 'parameters', and 'confidence'
        """
        user_input_lower = user_input.lower().strip()
        
        # Quick exit check (no need for LLM)
        if user_input_lower in ['exit', 'quit', 'goodbye', 'bye']:
            return {
                'intent': 'exit',
                'command': 'exit',
                'parameters': {},
                'confidence': 1.0
            }
        
        # Build a simpler, more reliable prompt
        current_date = datetime.now().strftime('%A, %B %d, %Y')
        
        classification_prompt = f"""Analyze this user request and identify the user's intent. Today is {current_date}.

User request: "{user_input}"

What does the user want to do? Choose the MOST LIKELY action from these options:

1. CREATE_NOTE - User wants to create/make/write a note
2. DELETE_NOTE - User wants to delete/remove a note
3. LIST_NOTES - User wants to see/list/show all notes
4. SEND_EMAIL - User wants to send/write/compose an email
5. CHECK_EMAIL - User wants to check/read/search emails or Gmail
6. CREATE_EVENT - User wants to create/schedule/add a calendar event or meeting
7. LIST_EVENTS - User wants to see/list upcoming calendar events
8. SEARCH_EVENT - User wants to search/find/look for a specific calendar event
9. CONVERSATION - User wants to chat, ask questions, or get general information (NOT a command)

IMPORTANT: Only classify as CONVERSATION if the user is asking general questions, chatting, or requesting information.
If the user wants to DO something (create, send, check, search, list, delete), it's a COMMAND.

Examples of CONVERSATION:
- "how are you"
- "what's the weather"
- "tell me a joke"
- "what time is it"
- "explain quantum physics"

Examples of COMMANDS:
- "make an event in calendar for gym at 3 to 4 pm" → CREATE_EVENT
- "check gmail for any meetings" → CHECK_EMAIL
- "find events about project" → SEARCH_EVENT

Respond with ONLY the action name (e.g., "SEARCH_EVENT" or "CONVERSATION"). Nothing else."""

        # Get LLM response
        try:
            messages = [
                {"role": "system", "content": "You are a precise intent classifier. Respond with only the action name."},
                {"role": "user", "content": classification_prompt}
            ]
            llm_response = self.llm_interface.generate(messages, max_new_tokens=50, temperature=0.1)
            
            # Clean and normalize response
            llm_response = llm_response.strip().upper().replace(' ', '_')
            
            # Map LLM response to command
            command_map = {
                'CREATE_NOTE': 'create_note',
                'DELETE_NOTE': 'delete_note',
                'LIST_NOTES': 'list_notes',
                'SEND_EMAIL': 'send_email',
                'CHECK_EMAIL': 'check_gmail',
                'CHECK_GMAIL': 'check_gmail',
                'CREATE_EVENT': 'create_event',
                'LIST_EVENTS': 'list_events',
                'SEARCH_EVENT': 'search_event',
                'CONVERSATION': 'conversation',
            }
            
            # Try to match response
            command = None
            for key in command_map:
                if key in llm_response:
                    command = command_map[key]
                    break
            
            if command:
                print(f"DEBUG: LLM classified as '{llm_response}' -> '{command}'")
                
                # Extract the actual content/parameters by removing command keywords
                content = self._extract_command_content(user_input, command)
                
                return {
                    'intent': 'command' if command != 'conversation' else 'conversation',
                    'command': command,
                    'parameters': {'raw_input': user_input, 'content': content},
                    'confidence': 0.9
                }
            else:
                # LLM response unclear, try keyword fallback
                print(f"DEBUG: LLM response unclear: '{llm_response}', using fallback")
                return self._fallback_classification(user_input)
            
        except Exception as e:
            print(f"DEBUG: LLM classification error: {e}, using fallback")
            return self._fallback_classification(user_input)
    
    def _fallback_classification(self, user_input: str) -> Dict[str, Any]:
        """Fallback keyword-based classification if LLM fails"""
        user_input_lower = user_input.lower()
        
        # Enhanced keyword detection
        keywords_map = {
            'create_note': ['create note', 'make note', 'write note', 'new note', 'add note'],
            'delete_note': ['delete note', 'remove note', 'erase note'],
            'list_notes': ['list note', 'show note', 'all note', 'my note', 'what note'],
            'send_email': ['send email', 'email to', 'write email', 'compose email', 'mail to'],
            'check_gmail': ['check email', 'check gmail', 'any email', 'new email', 'recent email', 'emails from', 'emails about'],
            'create_event': ['create event', 'schedule', 'new event', 'add event', 'make event', 'set up meeting', 'create meeting'],
            'list_events': ['list event', 'show event', 'my event', 'upcoming event', 'what event'],
            'search_event': ['search event', 'find event', 'look for event', 'event about', 'event for', 'events about'],
        }
        
        # Check for keyword matches
        for command, keywords in keywords_map.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    print(f"DEBUG: Fallback matched '{keyword}' -> '{command}'")
                    
                    # Extract content
                    content = self._extract_command_content(user_input, command)
                    
                    return {
                        'intent': 'command',
                        'command': command,
                        'parameters': {'raw_input': user_input, 'content': content},
                        'confidence': 0.7
                    }
        
        # Default to conversation
        print(f"DEBUG: No keyword match, defaulting to conversation")
        return {
            'intent': 'conversation',
            'command': 'conversation',
            'parameters': {'query': user_input, 'content': user_input},
            'confidence': 0.5
        }
    
    def _extract_command_content(self, user_input: str, command: str) -> str:
        """
        Extract the actual content from user input by removing command keywords
        
        Args:
            user_input: Full user input
            command: Detected command type
            
        Returns:
            Extracted content without command keywords
        """
        user_input_lower = user_input.lower()
        content = user_input
        
        # Define patterns to remove for each command type
        removal_patterns = {
            'create_note': [r'^\s*(?:create|make|write|add|new)\s+(?:a\s+)?note\s+(?:about\s+)?', r'^\s*note\s+'],
            'delete_note': [r'^\s*(?:delete|remove|erase)\s+(?:the\s+)?note\s+(?:about\s+|called\s+)?'],
            'list_notes': [r'^\s*(?:list|show|display)\s+(?:all\s+)?(?:my\s+)?notes?\s*'],
            'send_email': [r'^\s*(?:send|write|compose)\s+(?:an\s+)?email\s+', r'^\s*email\s+'],
            'check_gmail': [
                r'^\s*(?:search|check|read|look\s+(?:at|up|in|for))\s+(?:my\s+)?(?:gmail|email|inbox|emails)\s+(?:for\s+)?(?:messages\s+)?(?:about\s+)?',
                r'^\s*(?:do\s+i\s+have|any)\s+(?:any\s+)?(?:emails?|messages?)\s+(?:about\s+|from\s+)?'
            ],
            'create_event': [r'^\s*(?:create|make|schedule|add|set\s+up)\s+(?:an?\s+)?(?:event|meeting|appointment)\s+(?:in\s+calendar\s+)?(?:for\s+)?(?:called\s+)?'],
            'list_events': [r'^\s*(?:list|show|display)\s+(?:my\s+)?events?\s+(?:for\s+)?'],
            'search_event': [r'^\s*(?:search|find|look\s+for|locate)\s+(?:for\s+)?(?:event|events|calendar\s+entries|appointments)\s+(?:in\s+calendar\s+)?(?:about\s+)?(?:for\s+)?(?:related\s+to\s+)?(?:involving\s+)?'],
        }
        
        # Try to remove patterns
        if command in removal_patterns:
            original_content = content
            for pattern in removal_patterns[command]:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE).strip()
                if content and content != original_content:  # If something was removed, stop
                    print(f"DEBUG: Extracted content '{content}' using pattern: {pattern[:50]}...")
                    break
        
        # If nothing was removed or result is empty, return original
        if not content or content == user_input:
            print(f"DEBUG: No extraction, using full input")
            content = user_input
        
        return content
    
    def get_command_handler(self, classification: Dict[str, Any]) -> str:
        """
        Get the appropriate handler name for the classified command
        
        Args:
            classification: Result from classify_intent()
            
        Returns:
            Handler name as string
        """
        return classification['command']
