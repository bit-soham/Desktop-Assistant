
from sesame_ai import SesameAI, SesameWebSocket, TokenManager
import pyaudio
import threading
import time
import numpy as np

# Get authentication token using TokenManager
# api_client = SesameAI()
client = SesameAI()

# Create an anonymous account
signup_response = client.create_anonymous_account()
print(f"ID Token: {signup_response.id_token}")

# Look up account information
lookup_response = client.get_account_info(signup_response.id_token)
print(f"User ID: {lookup_response.local_id}")

# For easier token management, use TokenManager
token_manager = TokenManager(client, token_file="token.json")
id_token = token_manager.get_valid_token()
id_token = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjA1NTc3MjZmYWIxMjMxZmEyZGNjNTcyMWExMDgzZGE2ODBjNGE3M2YiLCJ0eXAiOiJKV1QifQ.eyJwcm92aWRlcl9pZCI6ImFub255bW91cyIsImlzcyI6Imh0dHBzOi8vc2VjdXJldG9rZW4uZ29vZ2xlLmNvbS9zZXNhbWUtYWktZGVtbyIsImF1ZCI6InNlc2FtZS1haS1kZW1vIiwiYXV0aF90aW1lIjoxNzU5MzU3OTU1LCJ1c2VyX2lkIjoiRVRIN3lUck5GS2RTVWY1dkV1MkRIQXpGOHdmMiIsInN1YiI6IkVUSDd5VHJORktkU1VmNXZFdTJESEF6Rjh3ZjIiLCJpYXQiOjE3NTkzNTc5NTUsImV4cCI6MTc1OTM2MTU1NSwiZmlyZWJhc2UiOnsiaWRlbnRpdGllcyI6e30sInNpZ25faW5fcHJvdmlkZXIiOiJhbm9ueW1vdXMifX0.CKFcMFKfKJONrCHZWcd3elM8h4BgE2r3zHBWm3tOtvxlAGAbtqj2BZ6zGKPUzRLtv7GlphRU2Noama9qhdmyFagcarJ-Rw5_-Afwj4iuIUcfnI1fZwdxWE5U7f-cSDxOwzo5sSH_aGaZTMm5F28HQwClcNS0D5jhunzktC_wEmRVWY27xkisTjNSEXPsRe-0SnEPwbUBHQzbnLj5io0XwkXbwCQCNyNmdP-fg_W5TWbKMUPB8XFE2CxAbxc7ayUyqV0boJkhSCHL2cv6TCYKgHCEQGrNCCT0x-2ynVuKWMpaVZvohMy5_BFfgQqr8sXD6uqgOicgE3z8RUKs7YOibQ"

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
preferred_name = "Noise"   # change to a distinctive part of your device name
selected_index = find_device_index_by_name(p, preferred_name)

# 2) or pick by exact index (uncomment to use):
selected_index = 2
speaker_output_index = 2
# 3) fallback to default input device if not found:
if selected_index is None:
    selected_index = get_default_input_index(p)

# Print available devices (helpful for debugging)
# print("Available input devices:")
# for d in list_input_devices(p):
#     print(f"  {d['index']}: {d['name']} (ch:{d['maxInputChannels']} rate:{d['defaultSampleRate']})")

if selected_index is None:
    raise RuntimeError("No input device found. Check microphone permissions / drivers.")
else:
    info = p.get_device_info_by_index(selected_index)
    device_rate = int(info.get('defaultSampleRate', RATE))
#     print(f"Using input device {selected_index}: {info['name']} (default rate {device_rate})")

# Use the chosen device index when opening the microphone stream
mic_stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=device_rate,                 # use device default sample rate if needed
                    input=True,
                    input_device_index=selected_index,
                    frames_per_buffer=CHUNK)

# If you also want to select the output device for speaker_stream:
# speaker_output_index = None  # set to a device index or find similarly
# speaker_stream = p.open(format=FORMAT,
#                         channels=CHANNELS,
#                         rate=ws.server_sample_rate,
#                         output=True,
#                         output_device_index=speaker_output_index)

# # Open speaker stream (using server's sample rate)
# # speaker_stream = p.open(format=FORMAT,
# #                         channels=CHANNELS,
# #                         rate=ws.server_sample_rate,
# #                         output=True)


# --- open-safe output stream + playback conversion ---
def get_output_device_info(pa, preferred_index=None):
    # print available outputs for debugging
    # for i in range(pa.get_device_count()):
    #     info = pa.get_device_info_by_index(i)
        # if int(info.get('maxOutputChannels', 0)) > 0:
        #     print(f"OUT idx={info['index']}: {info['name']} (maxOut={info['maxOutputChannels']}, rate={int(info['defaultSampleRate'])})")
    if preferred_index is not None:
        return pa.get_device_info_by_index(preferred_index)
    try:
        return pa.get_default_output_device_info()
    except Exception:
        # fallback: first output device
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get('maxOutputChannels', 0)) > 0:
                return info
    raise RuntimeError("No output device found")

# Decide correct settings from device
out_info = get_output_device_info(p, preferred_index=4)  # or pass index
out_index = out_info['index']
out_max_channels = int(out_info.get('maxOutputChannels', 2))
out_rate = int(out_info.get('defaultSampleRate', ws.server_sample_rate))

# Choose channels: if device supports stereo, use 2; else use 1
OUT_CHANNELS = 2 if out_max_channels >= 2 else 1

print(f"Using output device {out_index}: {out_info['name']} (channels={OUT_CHANNELS}, rate={out_rate})")

# Open output stream with device-supported params
try:
    speaker_stream = p.open(format=FORMAT,
                            channels=OUT_CHANNELS,
                            rate=out_rate,
                            output=True,
                            output_device_index=out_index,
                            frames_per_buffer=CHUNK)
except Exception as e:
    print("Failed to open output stream:", repr(e))
    raise

# Playback loop: convert mono->stereo if needed
def play_audio():
    print("Audio playback started...")
    try:
        while True:
            audio_chunk = ws.get_next_audio_chunk(timeout=0.01)  # raw bytes
            
            if not audio_chunk:
                
                time.sleep(0.005)
                continue
            print("speaking")
            # If output expects stereo but chunk is mono, duplicate channels
            if OUT_CHANNELS == 2:
                # assume incoming is int16 mono
                arr = np.frombuffer(audio_chunk, dtype=np.int16)
                if arr.size == 0:
                    continue
                stereo = np.empty((arr.size * 2,), dtype=np.int16)
                stereo[0::2] = arr    # left
                stereo[1::2] = arr    # right (duplicate)
                speaker_stream.write(stereo.tobytes())
            else:
                # output is mono - write directly
                speaker_stream.write(audio_chunk)
    except KeyboardInterrupt:
        print("Audio playback stopped")


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

# Function to play received audio
# def play_audio():
#     print("Audio playback started...")
#     try:
#         while True:
#             audio_chunk = ws.get_next_audio_chunk(timeout=0.01)
#             if audio_chunk:
#                 speaker_stream.write(audio_chunk)
#     except KeyboardInterrupt:
#         print("Audio playback stopped")

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