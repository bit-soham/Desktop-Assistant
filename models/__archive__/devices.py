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
# selected_index = 1

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

# If you also want to select the output device for speaker_stream:
# speaker_output_index = None  # set to a device index or find similarly
# speaker_stream = p.open(format=FORMAT,
#                         channels=CHANNELS,
#                         rate=ws.server_sample_rate,
#                         output=True,
#                         output_device_index=speaker_output_index)
