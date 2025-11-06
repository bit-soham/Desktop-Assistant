# minimal_live_ws.py
# pip install websocket-client requests
import json, time, threading, urllib.parse
import websocket
import requests

EPHEMERAL_ENDPOINT = "http://127.0.0.1:5000/ephemeral_token"
WS_URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"

def fetch_ephemeral_token(ephemeral_url=EPHEMERAL_ENDPOINT):
    r = requests.get(ephemeral_url, timeout=5)
    r.raise_for_status()
    data = r.json()
    # token could be in "token" or "name" etc. Use full returned string.
    token = data.get("token") or data.get("name") or data.get("ephemeral_token") or data.get("access_token")
    if not token:
        raise RuntimeError("No token in response: " + repr(data))
    return token

def on_open(ws):
    print("[WS] on_open: sending session.start")
    init = {
        "type": "session.start",
        "model": "gemini-2.0-flash-live-001",   # use the live model name you have access to
        "audio": {"encoding": "pcm_s16le", "sample_rate": 16000, "channels": 1}
    }
    ws.send(json.dumps(init))

def on_message(ws, msg):
    # server events arrive here (JSON text or base64 audio).
    try:
        j = json.loads(msg)
        print("[MSG]", j.get("type"), j.get("text") or "")
    except Exception:
        print("[MSG raw]", repr(msg)[:200])

def on_error(ws, err):
    print("[WS ERROR]", err)

def on_close(ws, code, reason):
    print("[WS CLOSED]", code, reason)

def connect_with_token(ws_url, token):
    # try header method first (Authorization: Token <token>)
    headers = [f"Authorization: Token {token}"]
    ws_app = websocket.WebSocketApp(ws_url,
                                    header=headers,
                                    on_open=on_open,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
    thread = threading.Thread(target=lambda: ws_app.run_forever(ping_interval=30), daemon=True)
    thread.start()

    # wait briefly to see if it connected or was closed
    t0 = time.time()
    while t0 + 5 > time.time():
        if getattr(ws_app, "sock", None) and ws_app.sock.connected:
            print("[WS] connected via header")
            return ws_app
        time.sleep(0.05)

    # header method failed — try fallback access_token query param
    try:
        ws_app.close()
    except Exception:
        pass

    encoded = urllib.parse.quote(token, safe='')
    fallback = ws_url + ("&" if "?" in ws_url else "?") + "access_token=" + encoded
    print("[WS] trying fallback URL auth...")
    ws_app2 = websocket.WebSocketApp(fallback,
                                     on_open=on_open,
                                     on_message=on_message,
                                     on_error=on_error,
                                     on_close=on_close)
    thread2 = threading.Thread(target=lambda: ws_app2.run_forever(ping_interval=30), daemon=True)
    thread2.start()

    t0 = time.time()
    while t0 + 5 > time.time():
        if getattr(ws_app2, "sock", None) and ws_app2.sock.connected:
            print("[WS] connected via fallback (access_token)")
            return ws_app2
        time.sleep(0.05)

    raise RuntimeError("WebSocket connect failed (header and fallback). Check token, project permissions, and that the token is fresh.")

if __name__ == "__main__":
    token = fetch_ephemeral_token()
    print("ephemeral token preview:", token[:80] + ("..." if len(token) > 80 else ""))
    ws = connect_with_token(WS_URL, token)
    print("Connected — now you can send audio events (audio.buffer.append) or text requests.")
    # leave running; ctrl-c to exit
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Closing")
        try:
            ws.close()
        except Exception:
            pass
