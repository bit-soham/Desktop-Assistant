# Gmail Integration Guide

## Overview
The Desktop Assistant now supports Gmail integration for sending and checking emails using voice commands.

## Features

### 1. Send Email
Send emails by voice with automatic parsing of recipient, subject, and body.

**Command Format:**
```
send email to <recipient> subject <subject> body <message>
```

**Examples:**
- "send email to john@example.com subject meeting reminder body don't forget our meeting tomorrow at 3pm"
- "email alice saying I'll be late to the office today"
- "send email to bob about the project update"

**How it Works:**
1. Say "send email" followed by your message
2. The LLM will parse the recipient, subject, and body
3. If subject is not mentioned, it will be auto-generated
4. If recipient is a saved contact name, it will use their email
5. If contact doesn't exist, you'll be asked for their email address
6. Email is sent via Gmail API

### 2. Check Gmail
Search through your recent emails for specific content.

**Command Format:**
```
check gmail for <search query>
```

**Examples:**
- "check gmail for meetings"
- "check gmail for invoices"
- "check gmail for messages from john"

**How it Works:**
1. Say "check gmail for" followed by what you're looking for
2. Fetches your last 50 emails
3. Searches through subject, body, and snippets
4. Summarizes matching emails with sender and subject
5. Reads out the top 3 matches

### 3. Contact Management
Save contacts for easier email sending.

**Contacts File:** `user_data/contacts.json`

**Format:**
```json
{
  "john": "john@example.com",
  "alice": "alice.smith@company.com",
  "bob": "bob.jones@example.org"
}
```

When you send an email to a name (e.g., "email john"), the system:
1. Checks if "john" is in contacts
2. If found, uses the saved email
3. If not found, asks you for the email and saves it

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `google-auth-oauthlib`
- `google-auth-httplib2`
- `google-api-python-client`

### 2. Gmail API Setup

#### Enable Gmail API:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Gmail API for your project
4. Create OAuth 2.0 credentials (Desktop app)
5. Download the credentials as `client_secrets.json`
6. Place in project root directory

#### First Time Authentication:
1. Run the assistant: `python main.py`
2. Use a Gmail command (send email or check gmail)
3. Browser will open for OAuth consent
4. Sign in with your Google account
5. Grant permissions to send/read emails
6. Token is saved to `gmail_token.json` for future use

### 3. Configuration Files

**client_secrets.json** (OAuth credentials):
```json
{
  "installed": {
    "client_id": "your-client-id.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "your-client-secret",
    "redirect_uris": ["http://localhost"]
  }
}
```

**gmail_token.json** (Auto-generated after first auth):
- Stores access and refresh tokens
- Auto-refreshes when expired
- Delete to re-authenticate

## Usage Examples

### Sending Emails

**Basic Email:**
```
User: "send email to john@example.com subject Hello saying Hi John, how are you?"
Assistant: "Email sent to john@example.com with subject: Hello"
```

**Email to Contact:**
```
User: "send email to alice about tomorrow's meeting"
Assistant: [processes] "Email sent to alice@example.com with subject: Tomorrow's Meeting"
```

**Interactive Email (no details provided):**
```
User: "send email"
Assistant: "Ugh, fine. Tell me the recipient, subject, and what you want to say."
User: "to bob, subject project update, tell him everything is on track"
Assistant: "Email sent to bob@company.com with subject: project update"
```

**New Contact:**
```
User: "send email to sarah saying great job on the presentation"
Assistant: "I don't have an email for sarah. What's their email address?"
User: "sarah dot williams at company dot com"
Assistant: [saves contact] "Email sent to sarah.williams@company.com"
```

### Checking Emails

**Search for Meetings:**
```
User: "check gmail for meetings"
Assistant: "Found 3 emails about meetings. Email 1: From John Smith, subject: Team Meeting Tomorrow. Email 2: From Alice Johnson, subject: Q4 Planning Meeting. Email 3: From Bob Wilson, subject: Meeting Reschedule."
```

**Search for Invoices:**
```
User: "check gmail for invoices"
Assistant: "Found 2 emails about invoices. Email 1: From Billing Department, subject: Monthly Invoice. Email 2: From Vendor Services, subject: Invoice #12345."
```

**No Results:**
```
User: "check gmail for vacation requests"
Assistant: "I didn't find any emails about vacation requests in your last 50 messages."
```

## Command Autocorrect

The system includes autocorrect for Gmail commands:

- "send mail" → "send email"
- "check email" → "check gmail"
- "check mail" → "check gmail"

## LLM Email Parsing

The email parser uses the LLM to intelligently extract:

1. **Recipient:** Email address or contact name
2. **Subject:** Explicit or auto-generated from body
3. **Body:** Main message content

**Parsing Examples:**

Input: "to john subject meeting body let's meet at 3"
```json
{
  "to": "john",
  "subject": "meeting",
  "body": "let's meet at 3"
}
```

Input: "email alice saying I'm running late"
```json
{
  "to": "alice",
  "subject": "Running Late",
  "body": "I'm running late"
}
```

## Error Handling

**Authentication Issues:**
- Delete `gmail_token.json` and re-authenticate
- Ensure `client_secrets.json` is valid
- Check internet connection

**Email Sending Failures:**
- Verify recipient email is valid
- Check Gmail API quotas
- Ensure proper OAuth scopes

**Email Search Issues:**
- Verify internet connection
- Check OAuth token is valid
- Try broader search terms

## Security Notes

1. **OAuth Tokens:** Keep `gmail_token.json` secure (not in version control)
2. **Client Secrets:** Protect `client_secrets.json` (not in version control)
3. **Contacts:** `contacts.json` contains email addresses - keep private
4. **Permissions:** App only requests send/read permissions, not delete

## Testing

Run the test suite:
```bash
python test_gmail_services.py
```

This tests:
- Email parsing with LLM
- Contact management
- Gmail authentication
- Email retrieval

## Troubleshooting

**Browser doesn't open for OAuth:**
- Check if `client_secrets.json` exists
- Verify OAuth redirect URIs include `http://localhost`

**"Insufficient permissions" error:**
- Delete `gmail_token.json`
- Re-authenticate with proper scopes

**LLM parsing errors:**
- Ensure LLM interface is configured
- Check API key in `.env` file
- Try more explicit email commands

**Email not sending:**
- Verify internet connection
- Check recipient email format
- Review Gmail API quotas in Cloud Console

## File Structure

```
Desktop Assistant/
├── services/
│   └── gmail_services.py          # Gmail API integration
├── user_data/
│   └── contacts.json               # Email contacts
├── client_secrets.json             # OAuth credentials (not in git)
├── gmail_token.json                # Access token (auto-generated)
└── test_gmail_services.py          # Test suite
```

## Future Enhancements

Potential features to add:
- Reply to emails
- Forward emails
- Attach files
- Mark as read/unread
- Delete emails
- Search by date range
- Filter by sender
- Create email drafts
