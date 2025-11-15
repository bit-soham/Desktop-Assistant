"""
Test Gmail Services
Tests email parsing and Gmail API integration
"""

from services.gmail_services import GmailService, EmailParser
from models.llm_interface import create_llm_interface

def test_email_parser():
    print("=" * 60)
    print("Testing Email Parser")
    print("=" * 60)
    
    # Create LLM interface
    llm_interface = create_llm_interface(use_local=False)
    email_parser = EmailParser(llm_interface)
    
    # Test cases
    test_inputs = [
        "send email to john@example.com subject meeting reminder body don't forget our meeting tomorrow at 3pm",
        "send email alice saying I'll be late to the office today",
        "send to bob about the late arrival",
        "to bob@company.com with subject Project Update tell him the project is on track and we'll finish by Friday"
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\nTest {i}: {test_input}")
        result = email_parser.parse_email_creation(test_input)
        print(f"Result: {result}")
        print("-" * 60)

def test_send_email():
    print("\n" + "=" * 60)
    print("Testing Send Email")
    print("=" * 60)
    
    gmail_service = GmailService()
    
    # Test sending email
    print("\nTesting email sending...")
    print("NOTE: This will send an actual email to ntirth005@gmail.com")
    choice = input("Do you want to send a test email to naumish? (y/n): ")
    
    if choice.lower() == 'y':
        try:
            gmail_service.authenticate()
            print("✅ Gmail authentication successful!")
            
            # Send test email to naumish
            to_email = "ntirth005@gmail.com"
            subject = "Test Email from Desktop Assistant"
            body = """Hi Naumish,

This is a test email sent from the Desktop Assistant application.
The Gmail integration is working successfully!

Best regards,
Desktop Assistant"""
            
            print(f"\nSending email to {to_email}...")
            success = gmail_service.send_email(to_email, subject, body)
            
            if success:
                print("✅ Test email sent successfully to naumish!")
            else:
                print("❌ Failed to send test email")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Skipping send email test")

def test_gmail_service():
    print("\n" + "=" * 60)
    print("Testing Gmail Service")
    print("=" * 60)
    
    gmail_service = GmailService()
    
    # Test contacts
    print("\nTesting contacts...")
    gmail_service.add_contact("test_user", "test@example.com")
    result = gmail_service.get_email_from_name("test_user")
    email, matched_name, similarity = result
    print(f"Retrieved email for 'test_user': {email} (match: {matched_name}, similarity: {similarity:.2%})")
    
    # Test naumish contact
    print("\nTesting naumish contact...")
    result = gmail_service.get_email_from_name("naumish")
    email, matched_name, similarity = result
    print(f"Retrieved email for 'naumish': {email} (match: {matched_name}, similarity: {similarity:.2%})")
    
    # Test fuzzy matching
    print("\nTesting fuzzy matching...")
    print("Testing with 'naumsh' (missing 'i')...")
    result = gmail_service.get_email_from_name("naumsh", threshold=0.9)
    email, matched_name, similarity = result
    if email:
        print(f"✅ Fuzzy match found! Email: {email} (matched: {matched_name}, similarity: {similarity:.2%})")
    else:
        print(f"❌ No fuzzy match found (similarity: {similarity:.2%})")
    
    print("\nTesting with 'naomish' (typo)...")
    result = gmail_service.get_email_from_name("naomish", threshold=0.9)
    email, matched_name, similarity = result
    if email:
        print(f"✅ Fuzzy match found! Email: {email} (matched: {matched_name}, similarity: {similarity:.2%})")
    else:
        print(f"❌ No fuzzy match found (similarity: {similarity:.2%})")
    
    print("\nTesting with 'john' (exact match)...")
    result = gmail_service.get_email_from_name("john", threshold=0.9)
    email, matched_name, similarity = result
    if email:
        print(f"✅ Exact match! Email: {email} (matched: {matched_name}, similarity: {similarity:.2%})")
    else:
        print(f"❌ No match found")
    
    # Test authentication (will require OAuth flow)
    print("\nTesting Gmail authentication...")
    print("NOTE: This will open a browser for OAuth authentication")
    choice = input("Do you want to test Gmail authentication? (y/n): ")
    
    if choice.lower() == 'y':
        try:
            gmail_service.authenticate()
            print("✅ Gmail authentication successful!")
            
            # Test getting recent emails
            print("\nFetching recent emails...")
            emails = gmail_service.get_recent_emails(max_results=5)
            print(f"Retrieved {len(emails)} emails")
            
            for i, email in enumerate(emails[:3], 1):
                print(f"\nEmail {i}:")
                print(f"  From: {email['from']}")
                print(f"  Subject: {email['subject']}")
                print(f"  Snippet: {email['snippet'][:100]}...")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("Skipping Gmail authentication test")

if __name__ == "__main__":
    print("Gmail Services Test Suite\n")
    
    # Test email parser
    test_email_parser()
    
    # Test send email
    test_send_email()
    
    # Test Gmail service
    test_gmail_service()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)
