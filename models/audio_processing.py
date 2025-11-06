import wave
import pyaudio # type: ignore
import soundfile as sf # type: ignore
import os 
import re
import numpy as np # type: ignore
from TTS.tts.configs.xtts_config import XttsConfig # type: ignore
from TTS.tts.models.xtts import Xtts # type: ignore

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

class AudioProcessor:
    def __init__(self, whisper_model, xtts_model, xtts_config, output_dir='outputs'):
        self.whisper_model = whisper_model
        self.xtts_model = xtts_model
        self.xtts_config = xtts_config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # Function to play audio using PyAudio
    def play_audio(self, file_path, output_device_index=None):
        print(f"DEBUG: Playing audio: {file_path}, output_device_index={output_device_index}")
        wf = wave.open(file_path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output_device_index=output_device_index,
                        output=True,
                        frames_per_buffer=1024)
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.close()
        p.terminate()
        print("DEBUG: Audio playback completed.")

    # Function to record audio from the microphone and save to a file
    def record_audio(self, file_path, input_device_index=None):
        print(f"DEBUG: Starting recording -> {file_path} using input_device_index={input_device_index}")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, input_device_index=input_device_index, frames_per_buffer=1024)
        frames = []
        print("Recording...")
        try:
            while True:
                data = stream.read(1024)
                frames.append(data)
        except KeyboardInterrupt:
            pass
        print("Recording stopped.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        print(f"DEBUG: Audio recording completed and saved to: {file_path}")

    # Function to transcribe the recorded audio using faster-whisper
    def transcribe_with_whisper(self, audio_file):
        print(f"DEBUG: Transcribing audio file: {audio_file}")
        segments, info = self.whisper_model.transcribe(audio_file, beam_size=5)
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        transcription = transcription.strip()
        print(f"DEBUG: Transcription result: {transcription}")
        return transcription

    # Function to synthesize speech using XTTS (with safe chunking <= 230 chars)
    def process_and_play(self, prompt, audio_file_pth, output_device):
        print(f"DEBUG: Starting speech synthesis for prompt: {prompt[:50]}... (truncated)")
        
        def _split_into_chunks(text: str, max_len: int = 230):
            # Prefer sentence boundaries, then fallback to word-safe chunks
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            chunks = []
            current = ""
            for s in sentences:
                if not s:
                    continue
                # If sentence itself too long, break by words
                if len(s) > max_len:
                    words = s.split()
                    buf = ""
                    for w in words:
                        if len(buf) + 1 + len(w) > max_len:
                            if buf:
                                chunks.append(buf)
                            buf = w
                        else:
                            buf = w if not buf else f"{buf} {w}"
                    if buf:
                        chunks.append(buf)
                else:
                    if len(current) + 1 + len(s) <= max_len:
                        current = s if not current else f"{current} {s}"
                    else:
                        if current:
                            chunks.append(current)
                        current = s
            if current:
                chunks.append(current)
            return chunks if chunks else [text]

        try:
            text_chunks = _split_into_chunks(prompt, max_len=230)
            if len(text_chunks) > 1:
                print(YELLOW + f"[XTTS] Splitting long text into {len(text_chunks)} chunks to avoid truncation." + RESET_COLOR)

            audio_segments = []
            for idx, chunk in enumerate(text_chunks, 1):
                print(f"DEBUG: Synthesizing chunk {idx}/{len(text_chunks)} (len={len(chunk)})")
                outputs = self.xtts_model.synthesize(
                    chunk,
                    self.xtts_config,
                    speaker_wav=audio_file_pth,
                    gpt_cond_len=24,
                    temperature=0.6,
                    language='en',
                    speed=1.2,
                )
                wav = outputs['wav']
                # Ensure numpy array for concatenation
                wav_np = np.asarray(wav, dtype=np.float32)
                audio_segments.append(wav_np)

            # Concatenate all segments
            full_audio = audio_segments[0] if len(audio_segments) == 1 else np.concatenate(audio_segments)

            # Save and play
            src_path = f'{self.output_dir}/output.wav'
            sample_rate = self.xtts_config.audio.sample_rate
            print(f"DEBUG: Saving synthesized audio to: {src_path}")
            sf.write(src_path, full_audio, sample_rate)
            print("Audio generated successfully.")
            self.play_audio(src_path, output_device)
        except Exception as e:
            error_msg = str(e)
            if "libtorchcodec" in error_msg or "FFmpeg" in error_msg:
                print(YELLOW + "\n⚠️  TTS Audio Generation Error (FFmpeg/TorchCodec issue)" + RESET_COLOR)
                print(CYAN + f"Response text: {prompt}" + RESET_COLOR)
                print(YELLOW + "\nℹ️  The assistant's response is shown above as text." + RESET_COLOR)
                print(YELLOW + "ℹ️  To fix audio output, try these solutions:" + RESET_COLOR)
                print("   1. Install FFmpeg: winget install Gyan.FFmpeg")
                print("   2. Restart your terminal/VS Code after installing")
                print("   3. Ensure torch/torchaudio match TTS requirements (see TROUBLESHOOTING.md)")
            else:
                print(f"{YELLOW}Error during audio generation: {e}{RESET_COLOR}")
                print(CYAN + f"Response text: {prompt}" + RESET_COLOR)