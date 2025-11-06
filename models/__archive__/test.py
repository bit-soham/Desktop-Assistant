import websocket
import threading
import time
import ssl
import json
import traceback

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
    self.connected = False
    # save last close reason
    self._last_close = {"code": code, "reason": reason}

def connect(self, timeout=20):
    websocket.enableTrace(True)
    print("[WS] connecting to:", self.ws_url)
    print("[WS] using API key prefix:", (self.api_key or "")[:8] + ("..." if self.api_key else ""))

    # If API requires a subprotocol, include it here. Remove if unknown.
    subprotocols = ["live"]

    headers = [f"Authorization: Bearer {self.api_key}"]

    # create app
    self.ws = websocket.WebSocketApp(
        self.ws_url,
        header=headers,
        subprotocols=subprotocols,
        on_open=self._on_open,
        on_message=self._on_message,
        on_error=self._on_error,
        on_close=self._on_close
    )

    def _run():
        try:
            # sslopt default is fine; increase debug logs from the lib
            self.ws.run_forever(ping_interval=30)
        except Exception as e:
            print("[WS] run_forever exception:", repr(e))

    self.thread = threading.Thread(target=_run, daemon=True)
    self.thread.start()

    # wait
    deadline = time.time() + timeout
    while not getattr(self, "connected", False) and time.time() < deadline:
        time.sleep(0.05)

    if not getattr(self, "connected", False):
        alive = getattr(self.thread, "is_alive", lambda: False)()
        sock = getattr(self.ws, "sock", None)
        print("[WS] diagnostics -> thread_alive:", alive, "ws.sock:", type(sock).__name__ if sock else None)
        print("[WS] last_error:", getattr(self, "_last_error", None))
        print("[WS] last_close:", getattr(self, "_last_close", None))
        raise RuntimeError("WS connect failed: check websocket trace output and on_error/on_close logs")
    print("[WS] connected OK")


