"""
push_to_talk_genai.py
Adds verbose logging so you can see what's recorded, sent, and received.
"""

import asyncio
import io
import json
import os
import queue
import sys
import threading
import time
import wave
import base64
import tempfile

import asyncio
import math
from google.genai import types

import numpy as np
import pyaudio
import soundfile as sf
import librosa

# google generative client (async)
from google import genai
from google.genai import types

# --- waiting tracking ---
awaiting_event = None        # will be set to asyncio.Event() inside run_push_to_talk()
last_send_ts = 0.0           # timestamp of last send
last_send_desc = ""          # human description of last send
import tempfile
import math
# RESPONSE_WAIT_TIMEOUT seconds to wait for model reply after send
RESPONSE_WAIT_TIMEOUT = 20.0

# --- globals (near top) ---
# single asyncio.Event used to signal "server replied" for the last send
response_event = None
last_send_ts = None
last_send_desc = None

# --- helper: save PCM16 raw bytes to WAV ---
def save_pcm16_to_wav(pcm_bytes: bytes, path: str, sr: int = 16000, channels: int = 1):
    import wave
    wf = wave.open(path, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm_bytes)
    wf.close()
    
    
async def response_watchdog(start_ts: float, desc: str, timeout: float = RESPONSE_WAIT_TIMEOUT):
    """Notify if no response arrives within `timeout` seconds for the send that started at start_ts."""
    await asyncio.sleep(timeout)
    global awaiting_event, last_send_ts, last_send_desc
    # Only report if still waiting and the send we are tracking is still the last send
    if awaiting_event is not None and awaiting_event.is_set() and last_send_ts == start_ts:
        waited = time.time() - start_ts
        print(f"[WAIT] Still waiting for response for '{desc}' after {waited:.1f}s (watchdog {timeout}s triggered).")

# ----------------- CONFIG -----------------
MODEL_NAME = "gemini-2.5-flash-native-audio-preview-09-2025"   # change if needed
CLIENT_CONFIG = {}   # if you need to pass configuration (or environment variable)
RESPONSE_MODALITIES = ["AUDIO"]   # we want audio responses
AUDIO_IN_SR = 16000
AUDIO_OUT_SR = 24000   # server replies are at 24kHz per docs
SILENCE_SECONDS = 1.0  # stop recording after this many seconds of silence
SILENCE_THRESHOLD = 500  # RMS threshold (tweak to your mic)
CHUNK_FRAMES = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ----------------- Helpers: intent classifier & task executor -----------------
def classify_intent(utterance: str):
    low = utterance.lower()
    if any(w in low for w in ("open", "start", "create", "launch", "run")):
        if "calculator" in low:
            return {"label": "TASK", "meta": {"action": "open_calculator"}}
        if "note" in low or "sticky" in low:
            return {"label": "TASK", "meta": {"action": "create_sticky", "text": utterance}}
        return {"label": "TASK", "meta": {"action": "generic", "raw": utterance}}
    return {"label": "CONVERSATION", "raw": utterance}

def execute_task(meta: dict):
    try:
        action = meta.get("action", "")
        if action == "open_calculator":
            import subprocess
            subprocess.Popen("calc.exe")
            return "Opened Calculator."
        elif action == "create_sticky":
            return f"Created note: {meta.get('text','')}"
        else:
            return f"Executing: {action or meta.get('raw','')}"
    except Exception as e:
        return f"Task failed: {e}"

# # ----------------- Utility: save PCM16 bytes to WAV -----------------
# def save_pcm16_to_wav(pcm_bytes: bytes, filepath: str, sr: int, channels: int = 1):
#     with wave.open(filepath, "wb") as wf:
#         wf.setnchannels(channels)
#         wf.setsampwidth(2)  # int16
#         wf.setframerate(sr)
#         wf.writeframes(pcm_bytes)

# ----------------- Microphone recorder (stop on silence) -----------------
def record_until_silence(p: pyaudio.PyAudio, device_index=None):
    """
    Record from mic until SILENCE_SECONDS of quiet. Return raw PCM16 bytes at AUDIO_IN_SR mono.
    Also prints summary stats and saves a temporary WAV file.
    """
    # try to open at target rate
    try:
        if device_index is None:
            device_index = p.get_default_input_device_info()["index"]
    except Exception:
        device_index = None

    # Get device default rate so we can resample if needed
    src_rate = None
    try:
        if device_index is not None:
            info = p.get_device_info_by_index(device_index)
            src_rate = int(info.get("defaultSampleRate", AUDIO_IN_SR))
    except Exception:
        src_rate = AUDIO_IN_SR

    want_rate = AUDIO_IN_SR
    use_rate = want_rate

    # Try open at want_rate; if it fails, open at device default and resample later
    stream = None
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=want_rate,
                        input=True, input_device_index=device_index,
                        frames_per_buffer=CHUNK_FRAMES)
        use_rate = want_rate
    except Exception:
        # fallback
        try:
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=src_rate,
                            input=True, input_device_index=device_index,
                            frames_per_buffer=CHUNK_FRAMES)
            use_rate = src_rate
            print(f"[REC] Recording at device rate {use_rate} Hz and will resample to {AUDIO_IN_SR} Hz.")
        except Exception as e:
            raise RuntimeError("Unable to open microphone stream: " + str(e))

    print("[REC] Recording... Speak now (will auto-stop after silence).")
    frames = []
    silence_start = None
    start_time = time.time()
    while True:
        data = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
        frames.append(data)
        # compute RMS on this chunk
        arr = np.frombuffer(data, dtype=np.int16).astype(np.int32)
        if arr.size == 0:
            rms = 0
        else:
            rms = int(np.sqrt(np.mean(arr * arr)))
        if rms < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif (time.time() - silence_start) >= SILENCE_SECONDS:
                print("\n[REC] Silence detected -> stopping.")
                break
        else:
            silence_start = None

    stream.stop_stream()
    stream.close()

    raw_bytes = b"".join(frames)
    total_time = time.time() - start_time

    # if use_rate != AUDIO_IN_SR -> resample using librosa
    if use_rate != AUDIO_IN_SR:
        print("[REC] Resampling to 16 kHz...")
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        y_res = librosa.resample(samples, orig_sr=use_rate, target_sr=AUDIO_IN_SR)
        y_out = (y_res * 32767.0).astype(np.int16)
        out_bytes = y_out.tobytes()
        sr = AUDIO_IN_SR
        samples_arr = y_out
    else:
        out_bytes = raw_bytes
        sr = use_rate
        samples_arr = np.frombuffer(out_bytes, dtype=np.int16)

    # Print summary stats
    num_samples = samples_arr.size
    duration = num_samples / sr if sr > 0 else 0.0
    if num_samples > 0:
        rms_all = int(np.sqrt(np.mean(samples_arr.astype(np.int64) ** 2)))
        peak = int(np.max(np.abs(samples_arr)))
        first_samples = samples_arr[:10].tolist()
    else:
        rms_all = 0
        peak = 0
        first_samples = []

    print(f"[REC] summary -> duration: {duration:.3f}s (wall clock {total_time:.3f}s), samples: {num_samples}, sr: {sr}")
    print(f"[REC] sample stats -> RMS: {rms_all}, peak: {peak}, first_samples: {first_samples}")

    # save a temporary WAV for inspection
    try:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        save_pcm16_to_wav(out_bytes, tf.name, sr, CHANNELS)
        print(f"[REC] saved recorded WAV to: {tf.name}")
    except Exception as e:
        print("[REC] failed to save temp WAV:", e)

    return out_bytes

# ----------------- Playback thread -----------------
class PlaybackThread(threading.Thread):
    def __init__(self, pyaudio_instance, out_sr=AUDIO_OUT_SR, channels=1):
        super().__init__(daemon=True)
        self.p = pyaudio_instance
        self.q = queue.Queue()
        self.rate = out_sr
        self.channels = channels
        self.running = True
        self.stream = None

    def run(self):
        try:
            self.stream = self.p.open(format=FORMAT, channels=self.channels, rate=self.rate, output=True, frames_per_buffer=CHUNK_FRAMES)
        except Exception as e:
            print("[PLAY] Failed to open output stream:", e)
            return

        while self.running:
            try:
                chunk = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.stream.write(chunk)
            except Exception as e:
                print("[PLAY] write error:", e)

    def enqueue(self, pcm_bytes):
        self.q.put(pcm_bytes)

    def stop(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        except Exception:
            pass

# ----------------- Main async logic (GenAI live native audio) -----------------
async def run_push_to_talk():
    # make sure API key is configured before creating client (your existing code does that)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing GOOGLE_API_KEY. Set it in PowerShell:\n"
            '$env:GOOGLE_API_KEY="YOUR_KEY"\n'
            "or with setx for persistent storage."
        )
    client = genai.Client(api_key=api_key)

    config = {
        "response_modalities": RESPONSE_MODALITIES,
        "system_instruction": "You are a helpful assistant. Reply verbally when asked."
    }

    p = pyaudio.PyAudio()
    player = PlaybackThread(p, out_sr=AUDIO_OUT_SR)
    player.start()

    print("[LIVE] connecting to model:", MODEL_NAME)
    async with client.aio.live.connect(model=MODEL_NAME, config=config) as session:
        print("[LIVE] session opened. Ready for push-to-talk.")

        # create the local asyncio.Event BEFORE defining the receive loop
        response_event = asyncio.Event()
        last_send_ts = None
        last_send_desc = None

        async def receive_loop(session, player, response_event):
            """Read incoming server responses and set response_event when a meaningful reply arrives."""
            try:
                async for response in session.receive():
                    now = time.time()
                    # debugging line
                    print("\n[RECV] raw response repr:", repr(response)[:1000])

                    # audio binary frames
                    try:
                        if getattr(response, "data", None):
                            data = response.data
                            print(f"[RECV] received audio bytes: {len(data)} bytes")
                            player.enqueue(data)
                            # signal sender that we've got audio back
                            if not response_event.is_set():
                                response_event.set()
                            continue
                    except Exception:
                        pass

                    sc = getattr(response, "server_content", None)
                    if sc is None:
                        print("[RECV] server_content is None; full response:", response)
                        continue

                    # parse transcripts / model_turn parts
                    try:
                        model_turn = getattr(sc, "model_turn", None)
                        if model_turn:
                            for part in getattr(model_turn, "parts", []) or []:
                                inline = getattr(part, "inline_data", None)
                                if inline:
                                    text = getattr(inline, "text", None)
                                    if text:
                                        print("[TRANSCRIPT]", text)
                                        classification = classify_intent(text)
                                        print("[INTENT]", classification)
                                        if classification.get("label") == "CONVERSATION":
                                            ask = f"Answer this: {text}"
                                            try:
                                                await session.send_realtime_input(text=types.Text(content=ask))
                                            except Exception as e:
                                                print("[SEND] failed to send ask:", e)
                                        else:
                                            meta = classification.get("meta", {})
                                            summary = execute_task(meta)
                                            try:
                                                await session.send_realtime_input(text=types.Text(content=summary))
                                            except Exception as e:
                                                print("[SEND] failed to send tts summary:", e)
                                        # signal sender that we saw a text reply
                                        if not response_event.is_set():
                                            response_event.set()
                    except Exception as e:
                        print("[RECV] server_content parsing error:", e)
            except Exception:
                print("[RECV] receive_loop crashed; exception:")
                import traceback; traceback.print_exc()

        # start receive loop task using the local response_event
        recv_task = asyncio.create_task(receive_loop(session, player, response_event))

        # --- send loop: single-blob send + wait for reply --- #
        try:
            while True:
                _ = await asyncio.get_event_loop().run_in_executor(
                    None, input, "\nPress ENTER to start recording (push-to-talk), or Ctrl-C to quit: "
                )

                pcm16 = record_until_silence(p)
                blob = types.Blob(data=pcm16, mime_type=f"audio/pcm;rate={AUDIO_IN_SR}")

                try:
                    num_samples = len(pcm16) // 2
                    duration = num_samples / AUDIO_IN_SR
                except Exception:
                    duration = None
                print(f"[SEND] sending recorded audio -> bytes: {len(pcm16)}, approx duration: {duration:.3f}s, mime: {blob.mime_type}")

                # save copy for debugging
                try:
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tf.close()
                    save_pcm16_to_wav(pcm16, tf.name, sr=AUDIO_IN_SR, channels=1)
                    print(f"[SEND] saved sent WAV to: {tf.name}")
                except Exception as e:
                    print("[SEND] failed to save sent WAV:", e)

                # clear the event, record timestamp & description
                response_event.clear()
                last_send_ts = time.time()
                last_send_desc = f"audio {len(pcm16)} bytes ~ {duration:.2f}s"

                # send the audio blob
                try:
                    await session.send_realtime_input(audio=blob)
                    print(f"[SEND] audio blob sent at {time.strftime('%H:%M:%S', time.localtime(last_send_ts))}")
                except Exception as e:
                    print("[SEND] failed to send audio blob:", e)
                    import traceback; traceback.print_exc()
                    continue

                # optional commit
                try:
                    await session.send_realtime_input(control=types.Control(name="commit"))
                except Exception:
                    pass

                # wait for model reply or timeout
                try:
                    await asyncio.wait_for(response_event.wait(), timeout=RESPONSE_WAIT_TIMEOUT)
                    elapsed = time.time() - last_send_ts if last_send_ts else 0.0
                    print(f"[SEND] reply received for '{last_send_desc}' after {elapsed:.3f}s")
                except asyncio.TimeoutError:
                    print(f"[WATCHDOG] timeout waiting for reply for '{last_send_desc}' after {RESPONSE_WAIT_TIMEOUT}s")

                await asyncio.sleep(0.1)

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("Interrupted — shutting down")
        finally:
            recv_task.cancel()
            player.stop()
            p.terminate()

    print("[LIVE] session closed.")

if __name__ == "__main__":
    try:
        asyncio.run(run_push_to_talk())
    except Exception as e:
        print("Fatal error:", e)
        raise
