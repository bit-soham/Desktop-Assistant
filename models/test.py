# # # list_devices.py
# # import pyaudio
# # p = pyaudio.PyAudio()
# # print("Device count:", p.get_device_count())
# # for i in range(p.get_device_count()):
# #     info = p.get_device_info_by_index(i)
# #     print(f"[{i}] {info['name']!s}  in:{info.get('maxInputChannels',0)}  out:{info.get('maxOutputChannels',0)}  defaultSampleRate:{info.get('defaultSampleRate')}")
# # p.terminate()


# # test_rec_play.py
# import wave, time
# import pyaudio

# IDX = None  # set to an integer device index from list_devices.py, or None to use default

# RATE = 16000
# CHUNK = 1024
# DUR = 3  # seconds
# OUT_FILE = "mic_test.wav"

# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
#                 input_device_index=IDX, frames_per_buffer=CHUNK)
# print("Recording for", DUR, "seconds...")
# frames = []
# for _ in range(int(RATE/CHUNK * DUR)):
#     data = stream.read(CHUNK, exception_on_overflow=False)
#     frames.append(data)
# stream.stop_stream()
# stream.close()
# p.terminate()
# wf = wave.open(OUT_FILE, "wb")
# wf.setnchannels(1)
# wf.setsampwidth(2)
# wf.setframerate(RATE)
# wf.writeframes(b"".join(frames))
# wf.close()
# print("Saved", OUT_FILE, "-- now playing it back...")

# # play it back
# p2 = pyaudio.PyAudio()
# wf = wave.open(OUT_FILE, 'rb')
# stream = p2.open(format=p2.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
# data = wf.readframes(1024)
# while data:
#     stream.write(data)
#     data = wf.readframes(1024)
# stream.stop_stream(); stream.close(); p2.terminate()
# print("Playback done.")
# import difflib
# prefix = 'create not'
# cmd_clean = 'create note'
# ratio = difflib.SequenceMatcher(None, prefix, cmd_clean).ratio()
# print(ratio)

import requests, time
url = "http://localhost:11434/v1/chat/completions"
payload = {"model":"llama3.2:3b","messages":[{"role":"user","content":"ping"}],"max_tokens":10}
start = time.time()
r = requests.post(url, json=payload, headers={"Authorization":"Bearer ollama"}, timeout=20)
print("status", r.status_code, "time", time.time()-start)
print(r.text[:1000])
