from sesame_ai import SesameAI, SesameWebSocket, TokenManager
import pyaudio
import threading
import time
import numpy as np

# Get authentication token using TokenManager
api_client = SesameAI()
token_manager = TokenManager(api_client, token_file="token.json")
id_token = token_manager.get_valid_token()
id_token = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjA1NTc3MjZmYWIxMjMxZmEyZGNjNTcyMWExMDgzZGE2ODBjNGE3M2YiLCJ0eXAiOiJKV1QifQ.eyJuYW1lIjoic29oYW0gYml0IiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0pRVGFuMWVtZ0pOUHpBUWFTOWhvMVZkQy11c2lPWExpVmtCMkFPdUZXT0FYT3NIZz1zOTYtYyIsImlzcyI6Imh0dHBzOi8vc2VjdXJldG9rZW4uZ29vZ2xlLmNvbS9zZXNhbWUtYWktZGVtbyIsImF1ZCI6InNlc2FtZS1haS1kZW1vIiwiYXV0aF90aW1lIjoxNzU5MzU1MTA2LCJ1c2VyX2lkIjoiVTV1WmtNSHl5RlVEWFpmV1NiOEFjWjhLMTBKMyIsInN1YiI6IlU1dVprTUh5eUZVRFhaZldTYjhBY1o4SzEwSjMiLCJpYXQiOjE3NTkzNTUxMDYsImV4cCI6MTc1OTM1ODcwNiwiZW1haWwiOiJiaXRzb2hhbS5nb29kQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJmaXJlYmFzZSI6eyJpZGVudGl0aWVzIjp7Imdvb2dsZS5jb20iOlsiMTA3NTEwNzA3NTA0MDMzMTkwNDMxIl0sImVtYWlsIjpbImJpdHNvaGFtLmdvb2RAZ21haWwuY29tIl19LCJzaWduX2luX3Byb3ZpZGVyIjoiZ29vZ2xlLmNvbSJ9fQ.aZ-sg8jA2XJgPahLKhP2NzAxJmOZTSm8wVtqmirMuEgxCN6Igl_ifQ-0QdV5xwOsiT71XNejw2qKQAL5ZyFHCcBgLPsa-10F5OXrWXjG36UWrGTGMgqgiOmrbdeotpbjWvOaLW5yqVXxB2jlXAZQYHJJP2RmGU7EY2r0uggJOqN6DDYaFG07LAh0p60MVaOwp7BIJ6OhdUXSZdYDEo26cdjN5LFfSDH7qjqCIEZ9mNeCTTvJhh0yvTQHIgRIl1Ues452MphmJ2b9ghz8pxFPn38IJtsOxKCuasDAsKhwx3dAKfMdUlbZLeLjK1GbjET2yRtwuc1DsVidMwNYwc7KBQ"

# Connect to WebSocket (choose character: "Miles" or "Maya")
ws = SesameWebSocket(id_token=id_token, character="Maya")

# Set up connection callbacks
def on_connect():
    print("Connected to SesameAI!")

def on_disconnect():
    print("Disconnected from SesameAI")

ws.set_connect_callback(on_connect)
ws.set_disconnect_callback(on_disconnect)

# Connect to the server
ws.connect()

# Audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Initialize PyAudio
p = pyaudio.PyAudio()

# --- add these helper functions before you open streams ---
def list_input_devices(pa):
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get('maxInputChannels', 0)) > 0:
            devices.append({
                "index": i,
                "name": info.get('name'),
                "maxInputChannels": int(info.get('maxInputChannels')),
                "defaultSampleRate": int(info.get('defaultSampleRate', 16000))
            })
    return devices

def find_device_index_by_name(pa, name_substr):
    name_substr = name_substr.lower()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get('maxInputChannels', 0)) > 0 and name_substr in info.get('name', '').lower():
            return i
    return None

def get_default_input_index(pa):
    try:
        return pa.get_default_input_device_info().get('index')
    except Exception:
        return None

# --- before opening mic_stream, pick device ---
# Example choices:
# 1) pick by substring:
preferred_name = "microphone"   # change to a distinctive part of your device name
selected_index = find_device_index_by_name(p, preferred_name)

# 2) or pick by exact index (uncomment to use):
selected_index = 2
speaker_output_index = 2
# 3) fallback to default input device if not found:
if selected_index is None:
    selected_index = get_default_input_index(p)

# Print available devices (helpful for debugging)
print("Available input devices:")
for d in list_input_devices(p):
    print(f"  {d['index']}: {d['name']} (ch:{d['maxInputChannels']} rate:{d['defaultSampleRate']})")

if selected_index is None:
    raise RuntimeError("No input device found. Check microphone permissions / drivers.")
else:
    info = p.get_device_info_by_index(selected_index)
    device_rate = int(info.get('defaultSampleRate', RATE))
    print(f"Using input device {selected_index}: {info['name']} (default rate {device_rate})")

# Use the chosen device index when opening the microphone stream
mic_stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=device_rate,                 # use device default sample rate if needed
                    input=True,
                    input_device_index=selected_index,
                    frames_per_buffer=CHUNK)

# --- Modified speaker/output section (only changed this part) ---
# This forces output to the device at `speaker_output_index`. It validates the device,
# reads its default sample rate and max output channels, opens the stream, and
# resamples/duplicates channels when writing.

def resample_int16(mono_int16, src_rate, dst_rate):
    if src_rate == dst_rate or mono_int16.size == 0:
        return mono_int16
    src_len = mono_int16.shape[0]
    dst_len = int(round(src_len * (dst_rate / src_rate)))
    if dst_len <= 0:
        return np.array([], dtype=np.int16)
    x_old = np.arange(src_len)
    x_new = np.linspace(0, src_len - 1, dst_len)
    resampled = np.interp(x_new, x_old, mono_int16).astype(np.int16)
    return resampled

def mono_to_n_channels(mono_int16, n_channels):
    if n_channels == 1:
        return mono_int16
    return np.tile(mono_int16.reshape(-1,1), (1, n_channels)).reshape(-1)

# pick forced output device (speaker_output_index variable already set in your code)
def choose_output_device(pa, forced_index=None):
    if forced_index is not None:
        try:
            info = pa.get_device_info_by_index(forced_index)
            if int(info.get('maxOutputChannels', 0)) <= 0:
                raise RuntimeError(f"Device {forced_index} has no output channels")
            return forced_index, info
        except Exception as e:
            raise RuntimeError(f"Cannot use forced output device {forced_index}: {e}")
    # fallback (shouldn't be used since you set forced index)
    try:
        info = pa.get_default_output_device_info()
        return info.get('index'), info
    except Exception:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get('maxOutputChannels', 0)) > 0:
                return i, info
    raise RuntimeError("No output device available")

# Choose and open output device
out_index, out_info = choose_output_device(p, forced_index=speaker_output_index)
out_rate = int(out_info.get('defaultSampleRate', ws.server_sample_rate if hasattr(ws, 'server_sample_rate') else RATE))
out_max_channels = int(out_info.get('maxOutputChannels', 2))
print(f"Using output device {out_index}: {out_info.get('name')} (rate={out_rate}, channels={out_max_channels})")

try:
    speaker_stream = p.open(format=FORMAT,
                            channels=out_max_channels,
                            rate=out_rate,
                            output=True,
                            output_device_index=out_index,
                            frames_per_buffer=CHUNK)
except Exception as e:
    raise RuntimeError(f"Failed to open speaker stream on device {out_index}: {e}")

# Function to play received audio (modified to resample/expand to device channels)
def play_audio():
    print("Audio playback started (forced device).")
    src_rate = int(getattr(ws, "server_sample_rate", RATE))
    try:
        while True:
            audio_chunk = ws.get_next_audio_chunk(timeout=0.01)
            if audio_chunk:
                # assume ws sends mono int16 PCM bytes
                arr = np.frombuffer(audio_chunk, dtype=np.int16)
                # resample if needed
                if src_rate != out_rate:
                    arr = resample_int16(arr, src_rate, out_rate)
                # duplicate to device channels if device expects >1 channels
                if out_max_channels > 1:
                    arr = mono_to_n_channels(arr, out_max_channels)
                # write
                speaker_stream.write(arr.tobytes())
            else:
                time.sleep(0.005)
    except KeyboardInterrupt:
        print("Audio playback stopped")
    except Exception as e:
        print("Playback error:", repr(e))

# --- end modified speaker section ---

# Function to capture and send microphone audio
def capture_microphone():
    print("Microphone capture ssstarted...")
    try:
        print("hellooo")
        while True:
            if ws.is_connected():
                data = mic_stream.read(CHUNK, exception_on_overflow=False)
                print(data)
                ws.send_audio_data(data)
            else:
                # print('hellllo')
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Microphone capture stopped")

# Start audio threads
mic_thread = threading.Thread(target=capture_microphone)
mic_thread.daemon = True
mic_thread.start()

playback_thread = threading.Thread(target=play_audio)
playback_thread.daemon = True
playback_thread.start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Disconnecting...")
    ws.disconnect()
    mic_stream.stop_stream()
    mic_stream.close()
    speaker_stream.stop_stream()
    speaker_stream.close()
    p.terminate()