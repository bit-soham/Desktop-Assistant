# create_token.py
# pip install google-auth-oauthlib google-auth
from google_auth_oauthlib.flow import InstalledAppFlow
import os

CLIENT_SECRETS = "client_secrets.json"
TOKEN_FILE = "token.json"

# Must include cloud-platform for Generative AI access
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    # "https://www.googleapis.com/auth/generative-language", 
    "openid",
    "https://www.googleapis.com/auth/userinfo.email"
]

def create_token():
    if not os.path.exists(CLIENT_SECRETS):
        raise SystemExit("Missing client_secrets.json (create a Desktop app credential in Cloud Console).")

    # Delete old token file to force consent (you can also delete it manually)
    if os.path.exists(TOKEN_FILE):
        print("Removing existing token file to force fresh consent...")
        os.remove(TOKEN_FILE)

    # Run local server flow — browser will open for consent
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, SCOPES)
    creds = flow.run_local_server(port=0)   # opens browser, ask user to consent

    # Save credentials (contains refresh_token if consent returned one)
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    print("Saved token.json with scopes:", SCOPES)

if __name__ == "__main__":
    create_token()
