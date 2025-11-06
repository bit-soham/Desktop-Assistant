import torch
import os
from faster_whisper import WhisperModel # type: ignore
from TTS.tts.configs.xtts_config import XttsConfig # type: ignore
from TTS.tts.models.xtts import Xtts # type: ignore
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer # type: ignore
from TTS.tts.utils.speakers import SpeakerManager # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore

def setup_whisper_model(model_size="medium.en", device="cpu", compute_type="int8"):
    """Set up the faster-whisper model"""
    print("DEBUG: Setting up the faster-whisper model...")
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def setup_xtts_model(checkpoint_dir="C:/Users/bitso/XTTS-v2/"):
    """Load XTTS configuration and model"""
    print("DEBUG: Loading XTTS configuration...")
    xtts_config = XttsConfig()
    xtts_config.load_json(os.path.join(checkpoint_dir, "config.json"))

    print("DEBUG: Initializing XTTS model...")
    xtts_model = Xtts.init_from_config(xtts_config)

    # Manually load the checkpoint with weights_only=False
    checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
    print(f"DEBUG: Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)

    # Apply to the model (adapt based on TTS internals; this assumes the 'model' key has the state dict)
    xtts_model.load_state_dict(checkpoint['model'], strict=False)

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

    # Load mel stats from mel_stats.pth
    print("DEBUG: Loading mel stats...")
    mel_stats_path = os.path.join(checkpoint_dir, "mel_stats.pth")
    xtts_model.mel_stats = torch.load(mel_stats_path, map_location=torch.device("cpu"))
    print("DEBUG: Mel stats loaded successfully.")

    # Set model to eval mode and move to device (equivalent to eval=True in load_checkpoint)
    xtts_model.eval()
    print("DEBUG: XTTS model loaded successfully.")

    return xtts_model, xtts_config

def setup_llm_model():
    """Initialize the LLM model and tokenizer"""
    llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",
                                             device_map="cpu",
                                            torch_dtype="float32")
    return llm_tokenizer, llm_model

def setup_embedding_model():
    """Load the sentence transformer model for embeddings"""
    print("DEBUG: Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("DEBUG: Sentence transformer model loaded.")
    return embedding_model