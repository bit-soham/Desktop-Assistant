"""
Test Calendar Services
Tests event parsing, time parsing, and Google Calendar API integration
"""

from services.calendar_services import CalendarService, DurationParser, EventParser, TimeParser
from models.llm_interface import create_llm_interface
from datetime import datetime, timedelta

def test_time_parser():
    print("=" * 60)
    print("Testing Time Parser")
    print("=" * 60)
    
    parser = TimeParser()
    
    print("\nTesting Time Parsing:")
    time_tests = [
        "3pm",
        "3:30 PM",
        "14:30",
        "10am",
        "12:00",
        "6:45 PM"
    ]
    
    for test in time_tests:
        time_obj = parser.parse_time_string(test)
        print(f"Input: '{test}' -> {time_obj}")
    
    print("\nTesting Date Parsing:")
    date_tests = [
        "today",
        "tomorrow",
        "yesterday",
        "2025-11-20",
        "November 20",
        "Nov 20"
    ]
    
    for test in date_tests:
        date_obj = parser.parse_date_string(test)
        print(f"Input: '{test}' -> {date_obj}")
    print("-" * 60)

def test_duration_parser():
    print("=" * 60)
    print("Testing Duration Parser (Simple Parser)")
    print("=" * 60)
    
    parser = DurationParser()
    
    test_cases = [
        "next 3 hours",
        "for 2 days",
        "30 minutes",
        "1 week",
        "5 hrs",
        "today",
        "tomorrow",
        "this week",
        "2 hours and 30 minutes",
        "3 days and 2 hours"
    ]
    
    for test in test_cases:
        hours = parser.parse_duration(test)
        print(f"Input: '{test}' -> {hours} hours")
    print("-" * 60)

def test_event_parser():
    print("\n" + "=" * 60)
    print("Testing Event Parser (LLM-based)")
    print("=" * 60)
    
    # Create LLM interface
    llm_interface = create_llm_interface(use_local=False)
    event_parser = EventParser(llm_interface)
    
    # Test event creation parsing
    print("\nTesting Event Creation with Date and Start/End Times:")
    test_inputs = [
        "meeting with team from 2pm to 4pm about project updates",
        "doctor appointment tomorrow at 3pm for 1 hour",
        "lunch with john today from 12:30 to 1:30",
        "team standup at 10am to 10:30am",
        "project review from 9am to 11am",
        "conference call on November 20 from 3pm to 5pm",
        "dentist appointment tomorrow at 2pm"
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        test_input += datetime.now().strftime('%d-%m-%Y')
        print(f"\nTest {i}: {test_input}")
        result = event_parser.parse_event_creation(test_input)
        print(f"Title: {result.get('title')}")
        print(f"Description: {result.get('description')}")
        print(f"Date: {result.get('date')}")
        print(f"Start Time: {result.get('start_time')}")
        print(f"End Time: {result.get('end_time')}")
        
        # Validate that all expected fields are present
        assert 'title' in result, "Missing 'title' field"
        assert 'description' in result, "Missing 'description' field"
        assert 'date' in result, "Missing 'date' field"
        assert 'start_time' in result, "Missing 'start_time' field"
        assert 'end_time' in result, "Missing 'end_time' field"
        
        print("✅ All fields present")
        print("-" * 60)
    
    # Test search duration and start date parsing
    print("\n" + "=" * 60)
    print("Testing Search Duration and Start Date Parsing:")
    print("=" * 60)
    
    search_queries = [
        "do I have any meetings today",
        "what's on my calendar for the next 3 hours",
        "any events tomorrow",
        "meetings this week",
        "do I have any appointments",
        "events on November 20",
        "meetings yesterday"
    ]
    
    for i, query in enumerate(search_queries, 1):
        print(f"\nTest {i}: {query}")
        duration, start_date = event_parser.parse_search_duration(query)
        print(f"Duration: {duration} hours")
        print(f"Start Date: {start_date}")
        
        # Validate that return is a tuple
        assert isinstance(duration, (int, float, type(None))), "Duration should be a number or None"
        assert isinstance(start_date, (str, type(None))), "Start date should be a string or None"
        
        print("✅ Valid return format")
        print("-" * 60)

def test_calendar_service():
    print("\n" + "=" * 60)
    print("Testing Calendar Service")
    print("=" * 60)
    
    calendar_service = CalendarService()
    
    # Test authentication
    print("\nTesting Calendar authentication...")
    print("NOTE: This will open a browser for OAuth authentication")
    choice = input("Do you want to test Calendar authentication? (y/n): ")
    
    if choice.lower() == 'y':
        try:
            calendar_service.authenticate()
            print("✅ Calendar authentication successful!")
            
            # Test creating an event
            print("\n" + "=" * 60)
            print("Testing Event Creation")
            print("=" * 60)
            create_choice = input("Do you want to create a test event? (y/n): ")
            
            if create_choice.lower() == 'y':
                title = "Test Event from Desktop Assistant"
                description = "This is a test event created by the calendar integration."
                
                # Create event starting in 1 hour, ending in 2 hours
                start_time = datetime.now() + timedelta(hours=1)
                end_time = datetime.utcnow() + timedelta(hours=2)
                
                print(f"\nCreating event: {title}")
                print(f"Start: {start_time.strftime('%Y-%m-%d %I:%M %p')}")
                print(f"End: {end_time.strftime('%Y-%m-%d %I:%M %p')}")
                
                event = calendar_service.create_event(title, description, start_time, end_time)
                
                if event:
                    print(f"✅ Event created successfully!")
                    print(f"   Title: {event.get('summary')}")
                    print(f"   Link: {event.get('htmlLink')}")
                else:
                    print("❌ Failed to create event")
            
            # Test listing events
            print("\n" + "=" * 60)
            print("Testing List Events")
            print("=" * 60)
            list_choice = input("Do you want to list upcoming events? (y/n): ")
            
            if list_choice.lower() == 'y':
                duration = 168  # 1 week
                print(f"\nListing events for the next {duration} hours (1 week)...")
                events = calendar_service.list_events(duration)
                
                if events:
                    print(f"\n✅ Found {len(events)} event(s):")
                    for i, event in enumerate(events[:5], 1):
                        title = event.get('summary', 'Untitled')
                        start = event['start'].get('dateTime', event['start'].get('date'))
                        print(f"   {i}. {title} at {start}")
                    
                    if len(events) > 5:
                        print(f"   ... and {len(events) - 5} more events")
                else:
                    print("ℹ️  No events found")
            
            # Test searching events
            print("\n" + "=" * 60)
            print("Testing Search Events")
            print("=" * 60)
            search_choice = input("Do you want to search for events? (y/n): ")
            
            if search_choice.lower() == 'y':
                query = input("Enter search query (e.g., 'meeting'): ")
                duration = 168  # 1 week
                
                print(f"\nSearching for '{query}' in the next {duration} hours...")
                events = calendar_service.search_events(duration, query)
                
                if events:
                    print(f"\n✅ Found {len(events)} matching event(s):")
                    for i, event in enumerate(events[:5], 1):
                        title = event.get('summary', 'Untitled')
                        start = event['start'].get('dateTime', event['start'].get('date'))
                        print(f"   {i}. {title} at {start}")
                else:
                    print(f"ℹ️  No events found matching '{query}'")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping Calendar authentication test")

if __name__ == "__main__":
    print("Calendar Services Test Suite\n")
    
    # Test time parser
    test_time_parser()
    
    # Test duration parser
    test_duration_parser()
    
    # Test event parser
    test_event_parser()
    
    # Test calendar service
    test_calendar_service()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)
