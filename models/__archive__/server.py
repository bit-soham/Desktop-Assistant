# server_ephemeral.py -- using google-genai (new SDK)
# pip install google-genai flask
from flask import Flask, jsonify
import os, datetime
from google import genai

app = Flask(__name__)

# Option A (simple): use API key stored in env LONG_LIVED_API_KEY
API_KEY = os.getenv("LONG_LIVED_API_KEY")
if not API_KEY:
    raise RuntimeError("Set LONG_LIVED_API_KEY environment variable to your long-lived API key")

# create client with API key (google-genai client)
client = genai.Client(api_key=API_KEY, http_options={"api_version":"v1alpha"})

@app.route("/ephemeral_token")
def ephemeral_token():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    token = client.auth_tokens.create(
        config = {
        'uses': 1, # The ephemeral token can only be used to start a single session
        'expire_time': now + datetime.timedelta(minutes=30), # Default is 30 minutes in the future
        # 'expire_time': '2025-05-17T00:00:00Z',   # Accepts isoformat.
        'new_session_expire_time': now + datetime.timedelta(minutes=1), # Default 1 minute in the future
        'http_options': {'api_version': 'v1alpha'},
        } 
    )
    # return token.name (ephemeral token string)
    return jsonify({"token": token.name})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
