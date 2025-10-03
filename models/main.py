"""
Gemini Live push-to-talk template (Python)
- Streams mic -> Gemini Live (WebSocket streaming)
- Receives transcripts & audio chunks
- Uses a few-shot Gemini text call to classify intent (CONVERSATION / TASK)
- Plays back streamed audio; runs task confirmation TTS for TASK flow

IMPORTANT: Replace LIVE_WS_URL and adapt the send/recv message payloads according to your
Gemini Live WebSocket session API per the Google docs (see links in the chat).
"""

import os
import base64
import json
import threading
import time
import queue
import numpy as np
import pyaudio
import websocket                    # websocket-client
import google.generativeai as genai  # official google generative client (for few-shot text call)
import websocket
import threading
import time
import ssl
import json
import traceback

# -------------------------
# CONFIG / PLACEHOLDERS
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Official Live API docs describe how to open a session (you may need an ephemeral token or signed URL).
# TODO: Replace with the actual Gemini Live WebSocket endpoint / session creation flow.
LIVE_WS_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"  

# Audio params
RATE = 16000
CHANNELS = 1
CHUNK = 1024      # how many frames we read per pyaudio.read

# Few-shot intent prompt: keep short and focused
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

# -------------------------
# Helper: simple audio player
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
        # PCM16 little-endian bytes expected
        self.q.put(pcm16_bytes)

    def run(self):
        while self.running:
            try:
                pcm = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            # ---IGNORE-- #
            # if device expects stereo but input is mono, duplicate channels
            if self.channels == 2:
                arr = np.frombuffer(pcm, dtype=np.int16)
                stereo = np.empty((arr.size*2,), dtype=np.int16)
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
# Helper: classify intent via Gemini text API (few-shot)
# -------------------------
def classify_intent(utterance):
    """
    Uses the standard Gemini text API to classify utterance via few-shot prompt.
    Returns a dict: {"label": "CONVERSATION"|"TASK", "meta": {...}}
    """
    prompt = INTENT_FEWSHOT_PROMPT.format(utterance=utterance.replace('"', '\\"'))

    # configure client
    genai.configure(api_key=GEMINI_API_KEY)

    # Call a text model (choose model name available in your account; e.g. "models/text-bison-001" or "gpt-like")
    try:
        resp = genai.generate_text(
            model="models/text-bison-001",           # replace with available model if needed
            prompt=prompt,
            max_output_tokens=200,
            temperature=0.0
        )
        text_out = resp.text.strip()
    except Exception as e:
        print("Intent classification error:", e)
        return {"label": "CONVERSATION", "raw": ""}

    # Parse label heuristically: prefer the first token
    if text_out.upper().startswith("TASK"):
        # try to extract JSON part
        rest = text_out[len("TASK"):].strip()
        meta = {}
        try:
            # if a JSON object follows, try parse
            jstart = rest.find("{")
            if jstart != -1:
                jsonpart = rest[jstart:]
                meta = json.loads(jsonpart)
        except Exception:
            meta = {"raw": rest}
        return {"label": "TASK", "meta": meta, "raw": text_out}
    else:
        return {"label": "CONVERSATION", "raw": text_out}

# -------------------------
# Placeholder task executor
# -------------------------
def execute_task(meta):
    """
    Implement actual OS-specific actions here.
    meta may contain action and arguments as parsed from classifier.
    Return a short summary string to speak back to the user.
    """
    try:
        action = meta.get("action", "")
        if action == "open_calculator":
            # Example: Windows calculator
            import subprocess
            subprocess.Popen("calc.exe")
            return "Opened Calculator."
        elif action == "create_sticky":
            text = meta.get("text", "note created")
            # Implement your sticky note creation logic (Windows, macOS, etc.)
            return f"Created a sticky note with text: {text}"
        else:
            return f"Running task: {action}"
    except Exception as e:
        return f"Failed to run task: {e}"

# -------------------------
# WebSocket client skeleton to Gemini Live
# -------------------------
class GeminiLiveClient:
    def __init__(self, ws_url, api_key, audio_sample_rate=RATE, channels=1):
        self.ws_url = ws_url
        self.api_key = api_key
        self.audio_sample_rate = audio_sample_rate
        self.channels = channels
        self.ws = None
        self.connected = False
        self.playback_queue = queue.Queue()
        self.latest_final_transcript = None

   
    def _on_open(self, ws):
        try:
            print("[WS] on_open called")
            self.connected = True
            # If server needs initial JSON, send it here:
            # ws.send(json.dumps({"type":"session.start", ...}))
        except Exception as e:
            print("[WS] on_open exception:", e)
            traceback.print_exc()

    def _on_error(self, ws, err):
        # websocket-client sometimes gives Exception or WebSocketException
        print("[WS] on_error called:", repr(err))
        # store last error for later inspection
        self._last_error = err
        try:
            traceback.print_exc()
        except Exception:
            pass

    def _on_close(self, ws, code, reason):
        print(f"[WS] on_close called: code={code}, reason={reason}")
        self.connected =  False
        # save last close reason
        self._last_close = {"code": code, "reason": reason}

    def connect(self, timeout=15):
        websocket.enableTrace(True)
        print("[WS] connecting to:", self.ws_url)
        print("[WS] api_key present?:", bool(self.api_key))

        # only set header if we actually have a token
        headers = []
        if self.api_key:
            headers = [f"Authorization: Bearer {self.api_key}"]

        self.ws = websocket.WebSocketApp(
            self.ws_url,
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
                print("[WS run_forever exception]", repr(e))

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

        deadline = time.time() + timeout
        while not getattr(self, "connected", False) and time.time() < deadline:
            time.sleep(0.05)

        if not getattr(self, "connected", False):
            print("[WS] diagnostics -> thread_alive:", getattr(self.thread, "is_alive", lambda: False)(),
                "ws.sock:", getattr(self.ws, "sock", None))
            raise RuntimeError("WS connect failed: check websocket trace output and on_error/on_close logs")
        print("[WS] connected OK")




    def _on_open(self, ws):
        print("[WS] Connected")
        self.connected = True
        # TODO: Send any "session init" JSON required by the Live API here
        # e.g. {"type":"session.start","model":"gpt-...","audio": {"sample_rate":16000,...}}
        init_msg = {
            "type": "session.start",
            "model": "multimodal-live",   # placeholder model name
            "audio": {"encoding": "pcm_s16le", "sample_rate": self.audio_sample_rate, "channels": self.channels},
            # other fields: voice, initial system prompt, etc.
        }
        self.ws.send(json.dumps(init_msg))

    def _on_message(self, ws, raw_msg):
        """
        Called when server sends an event. The Live API emits structured events: transcripts, audio chunks, session events.
        You must read the Gemini Live docs and adjust parsing accordingly.
        """
        # Many servers send JSON for events, and audio may be sent as base64 in a JSON field.
        try:
            msg = json.loads(raw_msg)
        except Exception:
            print("[WS] Received non-JSON message (ignoring)")
            return

        # Example event types (adjust to real fields from docs):
        etype = msg.get("type")
        if etype == "transcript.interim":
            interim_text = msg.get("text", "")
            print("[Interim]", interim_text)
        elif etype == "transcript.final":
            final_text = msg.get("text", "")
            print("[Final]", final_text)
            self.latest_final_transcript = final_text
            # Immediately run classification on final transcript (you can offload to thread if needed)
            classification = classify_intent(final_text)
            print("Intent classification:", classification)
            if classification["label"] == "CONVERSATION":
                # In conversation mode, ask Gemini to generate a spoken reply.
                # TODO: send a message to the Live session to request TTS reply
                request_reply = {
                    "type": "response.request",
                    "mode": "speech",   # placeholder
                    "text": final_text
                }
                self.ws.send(json.dumps(request_reply))
            else:
                # If TASK: execute and  then reply confirmation via TTS
                meta = classification.get("meta", {})
                summary = execute_task(meta)
                # You can ask Gemini to synthesize 'summary' or use local TTS
                tts_request = {
                    "type": "response.request",
                    "mode": "speech",
                    "text": summary
                }
                self.ws.send(json.dumps(tts_request))
        elif etype == "audio.chunk":
            # Example: server sends base64-encoded PCM16 bytes in msg["data"]
            b64 = msg.get("data")
            if b64:
                pcm = base64.b64decode(b64)
                # enqueue for playback
                self.playback_queue.put(pcm)
        elif etype == "session.closed":
            print("[WS] Session closed:", msg.get("reason"))
            # handle reconnect if desired
        else:
            # debug print event
            print("[WS EVENT]", etype, msg.get("detail") or msg)

    def _on_error(self, ws, err):
        print("[WS] Error:", err)

    def _on_close(self, ws, close_status_code, close_msg):
        print("[WS] Closed", close_status_code, close_msg)
        self.connected = False

    def send_audio_pcm16(self, pcm_bytes):
        """
        Sends raw PCM16 bytes to the Live API. The Live API message envelope depends on the service.
        Typically you base64 encode the chunk and send a JSON event: {"type":"audio.buffer.append", "data":"<b64>"}
        Adjust to the exact schema in the Gemini docs.
        """
        if not self.connected:
            return
        b64 = base64.b64encode(pcm_bytes).decode("ascii")
        msg = {"type": "audio.buffer.append", "data": b64}
        self.ws.send(json.dumps(msg))

    def get_next_playback_chunk(self, timeout=0.1):
        try:
            return self.playback_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

# -------------------------
# Main: audio capture + playback + glue
# -------------------------
def main():
    # configure Gemini text client for few-shot classification
    genai.configure(api_key=GEMINI_API_KEY)

    # start PyAudio
    p = pyaudio.PyAudio()

    # select input device (you can adapt this to your earlier helpers)
    input_index = None
    try:
        input_index = p.get_default_input_device_info()["index"]
    except Exception:
        input_index = 0
    input_index = 1
    out_index = 3
    mic_stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=input_index,
                        frames_per_buffer=CHUNK)

    # open Gemini Live connection
    client = GeminiLiveClient(ws_url=LIVE_WS_URL, api_key=GEMINI_API_KEY, audio_sample_rate=RATE, channels=CHANNELS)
    client.connect()

    # audio player: picks default output device
    out_info = p.get_default_output_device_info()
    out_index = int(out_info.get("index", None))
    out_channels = int(out_info.get("maxOutputChannels", 1))
    player = AudioPlayer(p, output_device_index=out_index, out_rate=int(out_info.get("defaultSampleRate", RATE)), channels=out_channels)
    player.start()

    # push-to-talk control: simple loop where user types ENTER to start/stop
    print("Push-to-talk template running. Press ENTER to start recording; ENTER again to stop.")
    try:
        while True:
            input("Press ENTER to START recording (push-to-talk)...")
            print("Recording... press ENTER to STOP")
            # start streaming until user presses ENTER again
            stop_flag = False

            # stream loop in a thread
            def stream():
                while not stop_flag:
                    data = mic_stream.read(CHUNK, exception_on_overflow=False)
                    # send PCM16 chunk directly (adjust encoding if Live expects Opus/Opus frames)
                    client.send_audio_pcm16(data)
                print("stream thread ended")

            t = threading.Thread(target=stream, daemon=True)
            t.start()
            input()  # wait for user to press ENTER again to stop
            stop_flag = True
            t.join()
            print("Recording stopped. Waiting for final transcript / response...")

            # wait briefly for the server to emit final transcript and/or response audio
            wait_until = time.time() + 5.0
            while time.time() < wait_until:
                chunk = client.get_next_playback_chunk(timeout=0.2)
                if chunk:
                    player.enqueue_raw(chunk)
                else:
                    time.sleep(0.05)

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
