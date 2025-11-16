# LangChain Command Router Implementation

## Overview
The desktop assistant now uses **LangChain and LLM-based intelligent intent classification** instead of regex pattern matching for command routing.

## What Changed

### Before (Regex-based)
```python
m_create_note = re.match(r'^\s*create\s+note\b(.*)$', user_input_lower)
m_send_email = re.match(r'^\s*send\s+email\b(.*)$', user_input_lower)

if m_create_note:
    # handle create note
elif m_send_email:
    # handle send email
```

### After (LangChain-based)
```python
classification = command_router.classify_intent(user_input)
command = classification['command']

if command == 'create_note':
    # handle create note
elif command == 'send_email':
    # handle send email
elif command == 'conversation':
    # handle general conversation
```

## Benefits

1. **Flexible Input**: Users don't need to say exact command phrases
   - "make a note about the meeting" → `create_note`
   - "email john about the project" → `send_email`
   - "do i have any meetings" → `check_gmail`

2. **Intelligent Classification**: LLM understands intent contextually
   - Distinguishes between commands and conversations
   - Extracts parameters automatically
   - Provides confidence scores

3. **Natural Conversations**: The system knows when users want to chat vs execute commands
   - "how are you" → `conversation`
   - "tell me a joke" → `conversation`
   - "create meeting tomorrow" → `create_event`

## Architecture

### CommandRouter (`core/command_router.py`)
- Uses LLM to classify user intent
- Returns structured classification with:
  - `intent`: "command" or "conversation"
  - `command`: The specific command name
  - `parameters`: Extracted information
  - `confidence`: Classification confidence (0-1)
- Fallback to regex if LLM fails

### Main Loop (`main_with_langchain.py`)
- Calls `command_router.classify_intent(user_input)`
- Routes to appropriate handler based on `command`
- Handles both specific commands and general conversation

## Supported Commands

| Command | Description | Examples |
|---------|-------------|----------|
| `create_note` | Create a new note | "create note meeting notes", "make a note" |
| `delete_note` | Delete a note | "delete note old tasks", "remove the note" |
| `list_notes` | List all notes | "list notes", "show my notes" |
| `send_email` | Send an email | "send email to john", "email sarah" |
| `check_gmail` | Check Gmail | "check gmail for meetings", "any emails" |
| `create_event` | Create calendar event | "create event tomorrow at 3pm" |
| `list_events` | List events | "list events for next week" |
| `search_event` | Search events | "search event about project" |
| `conversation` | General chat | "how are you", "tell me a joke" |
| `exit` | Exit application | "exit", "quit", "goodbye" |

## Testing

Run the test script to verify the command router:
```bash
python test_command_router.py
```

This will test various inputs and show how they're classified.

## Dependencies

Added to `requirements.txt`:
- `langchain` - Core LangChain library
- `langchain-community` - Community integrations
- `sentence-transformers` - Already included for Gmail similarity search

Install with:
```bash
pip install -r requirements.txt
```

## Usage

Simply run the main file:
```bash
python "main_with_langchain .py"
```

The system will automatically classify your intent and route to the appropriate handler. No changes to voice commands needed - it's more flexible than before!

## Customization

To add new commands:

1. Add command to `CommandRouter.commands` dict in `core/command_router.py`
2. Add handler in main loop (`main_with_langchain.py`)
3. Update this documentation

Example:
```python
# In command_router.py
'open_app': {
    'description': 'Open an application',
    'examples': ['open chrome', 'launch spotify']
}

# In main_with_langchain.py
elif command == 'open_app':
    # Handle app opening
    pass
```
