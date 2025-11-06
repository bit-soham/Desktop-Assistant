import argparse
import os
import wave
import torch
import pyaudio #type: ignore
import soundfile as sf #type: ignore
from openai import OpenAI #type: ignore
from faster_whisper import WhisperModel #type: ignore
from sentence_transformers import SentenceTransformer, util #type: ignore
from TTS.tts.configs.xtts_config import XttsConfig #type: ignore
from TTS.tts.models.xtts import Xtts #type: ignore
from TTS.tts.utils.speakers import SpeakerManager #type: ignore
from TTS.tts.utils.text.tokenizer import TTSTokenizer #type: ignore
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer #type: ignore
# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

print("DEBUG: Setting up the faster-whisper model...")
# Set up the faster-whisper model
model_size = "medium.en"
# whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")

# # Function to open a file and return its contents as a string
# def open_file(filepath):
#     print(f"DEBUG: Opening file: {filepath}")
#     with open(filepath, 'r', encoding='utf-8') as infile:
#         return infile.read()

# # Initialize the OpenAI client with the API key
# client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# # Function to play audio using PyAudio
# def play_audio(file_path):
#     print(f"DEBUG: Playing audio from file: {file_path}")
#     wf = wave.open(file_path, 'rb')
#     p = pyaudio.PyAudio()
#     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                     channels=wf.getnchannels(),
#                     rate=wf.getframerate(),
#                     output=True)
#     data = wf.readframes(1024)
#     while data:
#         stream.write(data)
#         data = wf.readframes(1024)
#     stream.close()
#     p.terminate()
#     print("DEBUG: Audio playback completed.")

# Command Line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
output_dir = 'outputs'  # Adjust if needed
os.makedirs(output_dir, exist_ok=True)

print("DEBUG: Loading XTTS configuration...")
# Load XTTS configuration
xtts_config = XttsConfig()
xtts_config.load_json("C:/Users/bitso/XTTS-v2/config.json")  # Replace with your actual path

print("DEBUG: Initializing XTTS model...")
# Initialize XTTS model
xtts_model = Xtts.init_from_config(xtts_config)
# Manually load the checkpoint with weights_only=False
checkpoint_path = os.path.join("C:/Users/bitso/XTTS-v2/", "model.pth")  # Adjust if your checkpoint file has a different name
print(f"DEBUG: Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

# Apply to the model (adapt based on TTS internals; this assumes the 'model' key has the state dict)
xtts_model.load_state_dict(checkpoint['model'], strict=False)  # Use strict=False if keys don't match perfectly
# xtts_model.load_checkpoint(xtts_config, checkpoint_dir="C:/Users/bitso/XTTS-v2/", eval=True)  # Replace with your actual path
# xtts_model.cuda()  # Move the model to GPU if available

checkpoint_dir = "C:/Users/bitso/XTTS-v2/"  # Your directory path

# Load tokenizer from vocab.json
print("DEBUG: Loading tokenizer...")
vocab_path = os.path.join(checkpoint_dir, "vocab.json")
xtts_model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
print("DEBUG: Tokenizer loaded successfully.")

# Load speaker manager from speakers_xtts.pth
print("DEBUG: Loading speaker manager...")
speaker_file = os.path.join(checkpoint_dir, "speakers_xtts.pth")
xtts_model.speaker_manager = SpeakerManager()
xtts_model.speaker_manager.speakers = torch.load(speaker_file, map_location=torch.device('cpu'))
print("DEBUG: Speaker manager loaded successfully.")

# Load DVAE from dvae.pth
# print("DEBUG: Loading DVAE...")
# dvae_path = os.path.join(checkpoint_dir, "dvae.pth")
# dvae_checkpoint = torch.load(dvae_path, map_location=torch.device("cpu"), weights_only=False)
# # Assume it's keyed under 'model'; adjust if traceback shows otherwise (e.g., to dvae_checkpoint directly)
# xtts_model.dvae.load_state_dict(dvae_checkpoint['model'] if 'model' in dvae_checkpoint else dvae_checkpoint, strict=True)
# print("DEBUG: DVAE loaded successfully.")

# Load mel stats from mel_stats.pth
print("DEBUG: Loading mel stats...")
mel_stats_path = os.path.join(checkpoint_dir, "mel_stats.pth")
xtts_model.mel_stats = torch.load(mel_stats_path, map_location=torch.device("cpu"))
print("DEBUG: Mel stats loaded successfully.")

# Set model to eval mode and move to device (equivalent to eval=True in load_checkpoint)
xtts_model.eval()
print("DEBUG: XTTS model loaded successfully.")
