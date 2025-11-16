"""
Test script for CommandRouter to verify LangChain-based intent classification
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.command_router import CommandRouter
from models.llm_interface import create_llm_interface, USE_LOCAL_MODEL

def test_command_router():
    """Test the command router with various inputs"""
    
    print("Initializing LLM interface...")
    llm_interface = create_llm_interface(use_local=USE_LOCAL_MODEL)
    
    print("Creating command router...")
    router = CommandRouter(llm_interface)
    
    # Test cases - comprehensive natural language variations
    test_inputs = [
        # Note commands - various phrasings
        ("create note meeting notes with john", "create_note"),
        ("make a note about the project deadline", "create_note"),
        ("write note reminder to call mom", "create_note"),
        ("delete note old tasks", "delete_note"),
        ("remove the note about groceries", "delete_note"),
        ("list notes", "list_notes"),
        ("show me all my notes", "list_notes"),
        
        # Email commands - natural variations
        ("send email to sarah about the project", "send_email"),
        ("email john saying we need to reschedule", "send_email"),
        ("compose an email to the team with the updates", "send_email"),
        ("check gmail for any meetings", "check_gmail"),
        ("do i have any emails about the proposal", "check_gmail"),
        ("check my email for messages from boss", "check_gmail"),
        ("look up emails from last month about quarterly review", "check_gmail"),
        ("search inbox for messages about the new feature", "check_gmail"),
        
        # Calendar commands - the key test cases
        ("create event tomorrow at 3pm with the team", "create_event"),
        ("make an event in calendar for gym at 3 to 4 pm", "create_event"),
        ("schedule a meeting for project review next monday at 2pm", "create_event"),
        ("set up event called standup from 9am to 10am tomorrow", "create_event"),
        ("list my events for next week", "list_events"),
        ("show events for today", "list_events"),
        ("search for event in calendar about project review", "search_event"),
        ("search event about project review", "search_event"),
        ("find events for tomorrow", "search_event"),
        ("look for the meeting with client", "search_event"),
        ("find calendar entries related to budget discussion", "search_event"),
        ("locate calendar appointments involving client meetings", "search_event"),
        
        # Conversation - should not be commands
        ("how are you today", "conversation"),
        ("what's the weather like", "conversation"),
        ("tell me a joke", "conversation"),
        ("what time is it", "conversation"),
        
        # Exit
        ("exit", "exit")
    ]
    
    print("\n" + "="*60)
    print("Testing Command Router - Natural Language Understanding")
    print("="*60 + "\n")
    
    correct = 0
    total = len(test_inputs)
    errors = []
    
    for user_input, expected_command in test_inputs:
        print(f"\nInput: '{user_input}'")
        print(f"Expected: {expected_command}")
        print("-" * 60)
        
        classification = router.classify_intent(user_input)
        
        print(f"Got: {classification['command']}")
        print(f"Confidence: {classification['confidence']:.2f}")
        
        # Show extracted content
        content = classification['parameters'].get('content', '')
        if content and content != user_input:
            print(f"Extracted content: '{content}'")
        
        # Check if correct
        if classification['command'] == expected_command:
            print("✓ CORRECT")
            correct += 1
        else:
            print("✗ WRONG")
            errors.append((user_input, expected_command, classification['command']))
        
        print()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    if errors:
        print(f"\n{len(errors)} Misclassifications:")
        for inp, expected, got in errors:
            print(f"  '{inp}'")
            print(f"    Expected: {expected}, Got: {got}")
    else:
        print("\n🎉 Perfect! All tests passed!")
    print("="*60)

if __name__ == "__main__":
    test_command_router()
