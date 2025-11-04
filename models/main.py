# client_live_oauth.py
# pip install google-auth-oauthlib google-auth websocket-client requests pyaudio numpy google-generativeai

import os
import json
import base64
import time
import queue
import threading
import traceback
import urllib.parse

import requests
import pyaudio
import websocket
import numpy as np

# Optional: if you want to use the google generative text API for classification locally
# (Only for development; do NOT embed long-lived API keys in distributed apps)
try:
    import google as genai
except Exception:
    genai = None

# OAuth libraries
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from oauthlib.oauth2.rfc6749.errors import InvalidScopeError
import os
import traceback
# -------------------------
# Config
# -------------------------
LIVE_WS_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"

CLIENT_SECRETS_FILE = os.getenv("OAUTH_CLIENT_SECRETS", "client_secrets.json")
TOKEN_FILE = os.getenv("OAUTH_TOKEN_FILE", "token.json")

# OAUTH_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]  # adjust if you need narrower scopes

# Optional few-shot key for classify_intent (dev only)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Audio params
RATE = 16000
CHANNELS = 1
CHUNK = 1024
CLIENT_SECRETS = "client_secrets.json"
TOKEN_FILE = "token.json"

# Must include cloud-platform for Generative AI access
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
]
# -------------------------
# OAuth helpers
# -------------------------


def get_oauth_creds(client_secrets=CLIENT_SECRETS_FILE, token_file=TOKEN_FILE, scopes=OAUTH_SCOPES):
    """
    Robust loader/refresh for OAuth credentials:
    - load token.json (without forcing scopes)
    - try refresh if expired
    - if refresh fails (invalid_scope or other), delete token and run fresh consent flow
    - saves valid credentials to token_file and returns them
    """
    creds = None

    # 1) Try loading existing credentials (don't pass scopes here to avoid mismatches)
    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file)
            print("[OAuth] Loaded saved credentials from", token_file)
        except Exception as e:
            print("[OAuth] Error loading token file:", e)
            creds = None

    # 2) Attempt refresh if needed
    if creds and creds.expired and creds.refresh_token:
        try:
            print("[OAuth] Credentials expired. Attempting refresh...")
            creds.refresh(Request())
            # save refreshed credentials
            with open(token_file, "w") as f:
                f.write(creds.to_json())
            print("[OAuth] Token refreshed and saved.")
        except Exception as e:
            # If refresh fails (invalid_scope or other), fall through to force re-login
            print("[OAuth] Refresh failed:", repr(e))
            traceback.print_exc()
            creds = None
            try:
                os.remove(token_file)
                print("[OAuth] Removed stale token file to force fresh consent.")
            except Exception:
                pass

    # 3) If no valid creds, run a fresh consent flow and save result
    if not creds or not creds.valid:
        print("[OAuth] Starting interactive login flow (will open browser)...")
        if not os.path.exists(client_secrets):
            raise RuntimeError(f"Missing client_secrets.json: {client_secrets}")
        # Force offline access + user consent so we receive a refresh_token
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets, scopes)
        creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
        # Persist credentials
        with open(token_file, "w") as f:
            f.write(creds.to_json())
        print(f"[OAuth] Saved new credentials to {token_file}")

    # Final sanity check
    if not creds or not creds.valid:
        raise RuntimeError("Failed to obtain valid OAuth credentials.")

    return creds


# -------------------------
# Intent few-shot prompt
# -------------------------
INTENT_FEWSHOT_PROMPT = """
You are an assistant that classifies short user utterances into one of two labels:
- CONVERSATION  (user wants a chat or follow-up response)
- TASK          (user requests an action on the desktop, e.g. open calculator, create note)

Respond only with the label and optionally a short JSON with "action" when TASK.

Examples:
User: "Open calculator"
Label: TASK
{"action": "open_calculator"}

User: "What's the weather today?"
Label: CONVERSATION

User: "Create a sticky note that says buy milk"
Label: TASK
{"action": "create_sticky", "text": "buy milk"}

Now classify this utterance:
User: "{utterance}"
"""

def classify_intent(utterance):
    """
    Try using the local Gemini text API if GEMINI_API_KEY is set and genai is installed.
    Otherwise fallback to a simple heuristic.
    """
    prompt = INTENT_FEWSHOT_PROMPT.format(utterance=utterance.replace('"', '\\"'))
    if GEMINI_API_KEY and genai is not None:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            resp = genai.generate_text(
                model="models/text-bison-001",
                prompt=prompt,
                max_output_tokens=120,
                temperature=0.0
            )
            text_out = resp.text.strip()
        except Exception as e:
            print("[Intent] genai error:", e)
            text_out = ""
    else:
        # simple heuristic fallback
        low = utterance.lower()
        if any(w in low for w in ("open", "start", "create", "run", "launch", "make", "delete", "remove")):
            if "calculator" in low:
                return {"label": "TASK", "meta": {"action": "open_calculator"}}
            if "sticky" in low or "note" in low:
                text = utterance.partition("that says")[-1].strip() or utterance
                return {"label": "TASK", "meta": {"action": "create_sticky", "text": text}}
            return {"label": "TASK", "meta": {"action": "generic", "raw": utterance}}
        return {"label": "CONVERSATION", "raw": utterance}

    # parse output
    if text_out.upper().startswith("TASK"):
        rest = text_out[len("TASK"):].strip()
        meta = {}
        try:
            jstart = rest.find("{")
            if jstart != -1:
                meta = json.loads(rest[jstart:])
        except Exception:
            meta = {"raw": rest}
        return {"label": "TASK", "meta": meta, "raw": text_out}
    else:
        return {"label": "CONVERSATION", "raw": text_out}

# -------------------------
# Task executor
# -------------------------
def execute_task(meta):
    try:
        action = meta.get("action", "")
        if action == "open_calculator":
            import subprocess
            subprocess.Popen("calc.exe")
            return "Opened Calculator."
        elif action == "create_sticky":
            text = meta.get("text", "note created")
            # platform-specific: placeholder
            return f"Created a sticky note with text: {text}"
        else:
            return f"Executing task: {action or meta.get('raw','unknown')}"
    except Exception as e:
        return f"Failed to run task: {e}"

# -------------------------
# Audio player thread
# -------------------------
class AudioPlayer(threading.Thread):
    def __init__(self, pa, output_device_index=None, out_rate=RATE, channels=1, chunk=CHUNK):
        super().__init__(daemon=True)
        self.pa = pa
        self.output_device_index = output_device_index
        self.out_rate = out_rate
        self.channels = channels
        self.chunk = chunk
        self.q = queue.Queue()
        self.running = True
        self._open_stream()

    def _open_stream(self):
        self.stream = self.pa.open(format=pyaudio.paInt16,
                                   channels=self.channels,
                                   rate=self.out_rate,
                                   output=True,
                                   output_device_index=self.output_device_index,
                                   frames_per_buffer=self.chunk)

    def enqueue_raw(self, pcm16_bytes):
        self.q.put(pcm16_bytes)

    def run(self):
        while self.running:
            try:
                pcm = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.channels == 2:
                arr = np.frombuffer(pcm, dtype=np.int16)
                stereo = np.empty((arr.size * 2,), dtype=np.int16)
                stereo[0::2] = arr
                stereo[1::2] = arr
                self.stream.write(stereo.tobytes())
            else:
                self.stream.write(pcm)

    def stop(self):
        self.running = False
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass

# -------------------------
# Gemini Live client (uses OAuth credentials)
# -------------------------
class GeminiLiveClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self.connected = False
        self.playback_queue = queue.Queue()
        self.latest_final_transcript = None
        self._last_error = None
        self._last_close = None

    def _on_open(self, ws):
        print("[WS] on_open")
        self.connected = True
        init_msg = {
            "type": "session.start",
            "model": "multimodal-live",   # change if you have a specific live model name
            "audio": {"encoding": "pcm_s16le", "sample_rate": RATE, "channels": CHANNELS},
        }
        try:
            ws.send(json.dumps(init_msg))
        except Exception as e:
            print("[WS] send init error:", e)

    def _on_error(self, ws, err):
        print("[WS] on_error:", err)
        self._last_error = err

    def _on_close(self, ws, code, reason):
        print("[WS] on_close:", code, reason)
        self.connected = False
        self._last_close = {"code": code, "reason": reason}

    def _on_message(self, ws, raw_msg):
        print("hello")
        if isinstance(raw_msg, bytes):
            self.playback_queue.put(raw_msg)
            return
        try:
            msg = json.loads(raw_msg)
        except Exception:
            return
        etype = msg.get("type")
        if etype == "transcript.interim":
            print("[Interim]", msg.get("text"))
        elif etype == "transcript.final":
            final_text = msg.get("text", "")
            print("[Final]", final_text)
            self.latest_final_transcript = final_text
            classification = classify_intent(final_text)
            print("[Intent]", classification)

            if classification.get("label") == "CONVERSATION":
                request_reply = {
                    "type": "response.request",
                    "mode": "speech",
                    "text": final_text
                }
                try:
                    ws.send(json.dumps(request_reply))
                except Exception as e:
                    print("[WS] failed to send response.request:", e)
            else:
                meta = classification.get("meta", {})
                summary = execute_task(meta or {})
                print("[Task result]", summary)
                tts_request = {"type": "response.request", "mode": "speech", "text": summary}
                try:
                    ws.send(json.dumps(tts_request))
                except Exception as e:
                    print("[WS] failed to send tts_request:", e)

        elif etype == "audio.chunk":
            b64 = msg.get("data")
            if b64:
                try:
                    pcm = base64.b64decode(b64)
                    self.playback_queue.put(pcm)
                except Exception as e:
                    print("[WS] decode audio chunk error:", e)
        elif etype == "session.closed":
            print("[WS] session.closed:", msg.get("reason"))

    # connect using OAuth credentials (creds is google.oauth2.credentials.Credentials)
    def connect_with_oauth(self, creds, timeout=15):
        websocket.enableTrace(False)
        if not creds or not creds.valid:
            raise RuntimeError("Invalid OAuth credentials passed to connect_with_oauth()")

        # ensure token is fresh
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # save new token back to disk if token file path present in environment
                tf = os.getenv("OAUTH_TOKEN_FILE", TOKEN_FILE)
                with open(tf, "w") as f:
                    f.write(creds.to_json())
                print("[OAuth] token refreshed and saved.")
            except Exception as e:
                print("[OAuth] refresh failed:", e)

        access_token = creds.token
        if not access_token:
            raise RuntimeError("No access token available in credentials")

        # helper to start the websockets app
        def _start_ws(url, headers=None, wait_sec=timeout):
            try:
                if getattr(self, "ws", None):
                    try:
                        self.ws.close()
                    except Exception:
                        pass
            except Exception:
                pass

            self.ws = websocket.WebSocketApp(
                url,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            def _run():
                try:
                    self.ws.run_forever(ping_interval=30)
                except Exception as e:
                    print("[WS run_forever exception]", e)
            self.thread = threading.Thread(target=_run, daemon=True)
            self.thread.start()
            deadline = time.time() + wait_sec
            while not getattr(self, "connected", False) and time.time() < deadline:
                time.sleep(0.05)
            return getattr(self, "connected", False)

        # Try Authorization: Bearer header first
        headers = [f"Authorization: Bearer {access_token}"]
        print("[WS] trying Authorization: Bearer header...")
        ok = _start_ws(self.ws_url, headers=headers, wait_sec=timeout)

        # fallback to access_token query param
        if not ok:
            print("[WS] header auth failed; attempting fallback with access_token query param...")
            encoded = urllib.parse.quote(access_token, safe='')
            fallback_url = self.ws_url + ("&" if "?" in self.ws_url else "?") + "access_token=" + encoded
            print("[WS] fallback URL preview:", fallback_url[:200] + ("..." if len(fallback_url) > 200 else ""))
            ok = _start_ws(fallback_url, headers=None, wait_sec=timeout)

        if not ok:
            print("[WS] diagnostics -> thread_alive:", getattr(self.thread, "is_alive", lambda: False)(),
                  "last_error:", getattr(self, "_last_error", None), "last_close:", getattr(self, "_last_close", None))
            raise RuntimeError("WS connect failed: header and fallback both failed. Check project/API/billing and token.")

        print("[WS] connected OK")

    def send_audio_pcm16(self, pcm_bytes):
        if not self.connected:
            print("not connected")
            return
        try:
            b64 = base64.b64encode(pcm_bytes).decode("ascii")
            msg = {"type": "audio.buffer.append", "data": b64}
            print("sent")
            self.ws.send(json.dumps(msg))
        except Exception as e:
            print("[WS] send audio error:", e)

    def commit_audio(self):
        if not self.connected:
            return
        try:
            self.ws.send(json.dumps({"type": "audio.buffer.commit"}))
        except Exception:
            pass

    def get_next_playback_chunk(self, timeout=0.1):
        try:
            return self.playback_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        try:
            if self.ws:
                try:
                    self.ws.send(json.dumps({"type": "audio.buffer.commit"}))
                except Exception:
                    pass
                self.ws.close()
        except Exception:
            pass

# -------------------------
# Main
# -------------------------
def main():
    # get OAuth creds (will open browser once if token.json not present)
    creds = get_oauth_creds()

    # optional: configure genai for classify_intent if GEMINI_API_KEY set
    if GEMINI_API_KEY and genai is not None:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception as e:
            print("[genai configure] error:", e)

    p = pyaudio.PyAudio()

    # minimal device selection - adjust indices below if needed
    try:
        input_index = p.get_default_input_device_info()["index"]
    except Exception:
        input_index = 0
    try:
        out_info = p.get_default_output_device_info()
        out_index = int(out_info.get("index"))
        out_channels = int(out_info.get("maxOutputChannels", 1))
    except Exception:
        out_index = None
        out_channels = 1

    # override here if you want specific devices:
    # input_index = 1
    # out_index = 3

    mic_stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=input_index,
                        frames_per_buffer=CHUNK)

    player = AudioPlayer(p, output_device_index=out_index, out_rate=RATE, channels=out_channels)
    player.start()

    client = GeminiLiveClient(LIVE_WS_URL)
    client.connect_with_oauth(creds)  # connect using OAuth access token

    print("Ready. Press ENTER to START recording; press ENTER again to STOP (push-to-talk).")
    try:
        while True:
            input("Press ENTER to START recording...")
            print("Recording... press ENTER to STOP")
            stop_flag = False

            def stream_thread():
                nonlocal stop_flag
                chunk_count = 0
                while not stop_flag:
                    data = mic_stream.read(CHUNK, exception_on_overflow=False)
                    client.send_audio_pcm16(data)
                    chunk_count += 1

                    # Commit every 10 chunks (approx. every 640ms with CHUNK=1024)
                    if chunk_count % 10 == 0:
                        client.commit_audio()
                print("stream thread ended")

            t = threading.Thread(target=stream_thread, daemon=True)
            t.start()
            input()  # stop on ENTER
            stop_flag = True
            t.join()

            # wait shortly to receive final transcript and any reply audio
            wait_until = time.time() + 3.0
            while time.time() < wait_until:
                chunk = client.get_next_playback_chunk(timeout=0.2)
                if chunk:
                    player.enqueue_raw(chunk)
                else:
                    time.sleep(0.02)

    except KeyboardInterrupt:
        print("Shutting down...")

    finally:
        client.close()
        player.stop()
        mic_stream.stop_stream()
        mic_stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
