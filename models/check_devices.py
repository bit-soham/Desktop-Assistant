import pyaudio
import numpy as np
import threading
import queue
import time
import sys

# ====== CONFIG ======
CHUNK = 1024            # read frames per read from mic
FORMAT = pyaudio.paInt16
DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1

# ====== HELPERS ======
def list_input_devices(pa):
    out = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxInputChannels", 0)) > 0:
            out.append((i, info))
    return out

def list_output_devices(pa):
    out = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if int(info.get("maxOutputChannels", 0)) > 0:
            out.append((i, info))
    return out

def resample_int16(mono_int16, src_rate, dst_rate):
    # simple linear interpolation resampler for int16 arrays
    if mono_int16.size == 0 or src_rate == dst_rate:
        return mono_int16
    src_len = mono_int16.shape[0]
    dst_len = int(round(src_len * (dst_rate / float(src_rate))))
    if dst_len <= 0:
        return np.array([], dtype=np.int16)
    x_old = np.arange(src_len)
    x_new = np.linspace(0, src_len - 1, dst_len)
    resampled = np.interp(x_new, x_old, mono_int16).astype(np.int16)
    return resampled

def mono_to_n_channels(mono_int16, n):
    if n == 1:
        return mono_int16
    return np.tile(mono_int16.reshape(-1,1), (1, n)).reshape(-1)

# ====== MAIN LOOPBACK ======
def main():
    p = pyaudio.PyAudio()

    # list devices
    print("Input devices:")
    for idx, info in list_input_devices(p):
        print(f"  [{idx}] {info.get('name')}  channels={info.get('maxInputChannels')} rate={int(info.get('defaultSampleRate', DEFAULT_RATE))}")
    print("\nOutput devices:")
    for idx, info in list_output_devices(p):
        print(f"  [{idx}] {info.get('name')}  channels={info.get('maxOutputChannels')} rate={int(info.get('defaultSampleRate', DEFAULT_RATE))}")

    # choose devices (you can change these indices)
    try:
        input_index = int(input("\nEnter input device index (or press Enter for default): ") or -1)
    except Exception:
        input_index = -1
    try:
        output_index = int(input("Enter output device index (or press Enter for default): ") or -1)
    except Exception:
        output_index = -1

    if input_index == -1:
        try:
            input_index = p.get_default_input_device_info()['index']
        except Exception:
            print("No default input, using first input device.")
            input_index = list_input_devices(p)[0][0]

    if output_index == -1:
        try:
            output_index = p.get_default_output_device_info()['index']
        except Exception:
            print("No default output, using first output device.")
            output_index = list_output_devices(p)[0][0]

    in_info = p.get_device_info_by_index(input_index)
    out_info = p.get_device_info_by_index(output_index)

    in_rate = int(in_info.get('defaultSampleRate', DEFAULT_RATE))
    out_rate = int(out_info.get('defaultSampleRate', DEFAULT_RATE))
    out_channels = int(out_info.get('maxOutputChannels', 1))

    print(f"\nUsing input [{input_index}] {in_info.get('name')} @ {in_rate} Hz")
    print(f"Using output [{output_index}] {out_info.get('name')} @ {out_rate} Hz, channels={out_channels}")
    print("Tip: use headphones to avoid speaker->mic feedback. Press Ctrl+C to quit.\n")

    # Open streams
    try:
        mic_stream = p.open(format=FORMAT,
                            channels=1,                # force mono capture for simplicity
                            rate=in_rate,
                            input=True,
                            input_device_index=input_index,
                            frames_per_buffer=CHUNK)
    except Exception as e:
        print("Failed to open input stream:", e)
        p.terminate()
        sys.exit(1)

    try:
        speaker_stream = p.open(format=FORMAT,
                                channels=out_channels,
                                rate=out_rate,
                                output=True,
                                output_device_index=output_index,
                                frames_per_buffer=CHUNK)
    except Exception as e:
        print("Failed to open output stream:", e)
        mic_stream.close()
        p.terminate()
        sys.exit(1)

    q = queue.Queue(maxsize=50)
    running = threading.Event()
    running.set()

    # reader thread: read mic, push to queue (raw PCM16 bytes)
    def reader():
        while running.is_set():
            try:
                data = mic_stream.read(CHUNK, exception_on_overflow=False)
                # print(data)
            except Exception as e:
                # on some systems read might fail briefly
                # sleep a bit and continue
                print("mic read err:", e)
                time.sleep(0.01)
                continue
            # push raw PCM into queue (as bytes)
            try:
                q.put_nowait(data)
            except queue.Full:
                # drop oldest to keep up if slow
                try:
                    q.get_nowait()
                    q.put_nowait(data)
                except Exception:
                    pass

    # writer thread: pop from queue, resample/expand, write to speaker
    def writer():
        while running.is_set():
            try:
                data = q.get(timeout=0.1)
            except queue.Empty:
                continue
            arr = np.frombuffer(data, dtype=np.int16)

            # DEBUG
            print("Writer got chunk:", arr[:5], "len:", len(arr))

            if in_rate != out_rate and arr.size > 0:
                arr = resample_int16(arr, in_rate, out_rate)
            if out_channels > 1 and arr.size > 0:
                arr = mono_to_n_channels(arr, out_channels)

            try:
                speaker_stream.write(arr.tobytes())
                print("Wrote", len(arr), "samples to speaker")  # <--- debug
            except Exception as e:
                print("Write error:", e)

    t_reader = threading.Thread(target=reader, daemon=True)
    t_writer = threading.Thread(target=writer, daemon=True)
    t_reader.start()
    t_writer.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        running.clear()
        time.sleep(0.1)
        try:
            mic_stream.stop_stream(); mic_stream.close()
            speaker_stream.stop_stream(); speaker_stream.close()
        except Exception:
            pass
        p.terminate()
        print("Exited cleanly.")

if __name__ == "__main__":
    main()
