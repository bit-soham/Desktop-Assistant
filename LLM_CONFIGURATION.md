# LLM Configuration Guide

## Overview
Your Desktop Assistant now supports both **local LLM models** (like llama3.2:1b) and **Hugging Face Inference API** models. You can easily switch between them using a simple boolean flag.

## Quick Start

### Switch Between Local and API

Edit `models/llm_interface.py` and change the `USE_LOCAL_MODEL` variable:

```python
# Use LOCAL model (llama3.2:1b)
USE_LOCAL_MODEL = True

# Use HUGGING FACE API
USE_LOCAL_MODEL = False
```

That's it! The change will apply to:
- ✅ Note parsing (when creating notes)
- ✅ Real-time conversations (RAG-enhanced chat)

## Configuration Details

### Local Model Configuration
**File**: `models/llm_interface.py`

When `USE_LOCAL_MODEL = True`:
- Uses your downloaded llama3.2:1b model
- Requires `llm_tokenizer` and `llm_model` from `setup_llm_model()`
- Runs entirely on your machine (no internet needed after download)
- Slower but private

### API Model Configuration
**File**: `models/llm_interface.py`

When `USE_LOCAL_MODEL = False`:
- Uses Hugging Face Inference API
- Requires API key (currently hardcoded)
- Default model: `HuggingFaceTB/SmolLM3-3B`
- Faster and doesn't require local GPU/RAM

**To change API key or model**:
```python
HF_API_KEY = "your_api_key_here"
HF_MODEL = "HuggingFaceTB/SmolLM3-3B"  # or any other model
```

## Architecture

### Before (Old System)
```
main.py
  ├─→ note_manager.parse_note_creation(tokenizer, model)
  │     └─→ Direct local model calls
  │
  └─→ rag_llm_processor.chatgpt_streamed(...)
        └─→ Direct local model calls
```

### After (New System)
```
main.py
  ├─→ LLMInterface (created once)
  │     ├─→ use_local=True  → Local model
  │     └─→ use_local=False → HF API
  │
  ├─→ note_manager.parse_note_creation()
  │     └─→ llm_interface.generate()
  │
  └─→ rag_llm_processor.chatgpt_streamed()
        └─→ llm_interface.generate_streaming()
```

## Usage Examples

### Example 1: Using Local Model
```python
# In models/llm_interface.py
USE_LOCAL_MODEL = True

# Then run
python main.py
```

Output:
```
DEBUG: LLM Interface initialized with LOCAL model
DEBUG: Generating with LOCAL model (max_tokens=150)
```

### Example 2: Using API Model
```python
# In models/llm_interface.py
USE_LOCAL_MODEL = False

# Then run
python main.py
```

Output:
```
DEBUG: LLM Interface initialized with HF API (model: HuggingFaceTB/SmolLM3-3B)
DEBUG: Generating with HF API (model=HuggingFaceTB/SmolLM3-3B, max_tokens=150)
```

## Advanced Configuration

### Change Model at Runtime

You can also change the model when creating the assistant:

```python
# In main.py, change this line:
use_local = False  # or True

# This overrides the default in llm_interface.py
llm_interface = create_llm_interface(
    use_local=use_local,
    llm_tokenizer=llm_tokenizer if use_local else None,
    llm_model=llm_model if use_local else None
)
```

### Use Different API Models

Edit `models/llm_interface.py`:

```python
# Try different models
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # If available
# OR
HF_MODEL = "HuggingFaceTB/SmolLM3-3B"
# OR
HF_MODEL = "microsoft/phi-3-mini-4k-instruct"
```

## Testing

### Test API Connection
```bash
python models/hugface.py
```

This should output a response to "What is the capital of France?"

### Test in Assistant
```bash
python main.py

# Try creating a note:
> "create note buy groceries tomorrow for 3 days"

# The LLM will parse this and create a structured note
```

## Troubleshooting

### API Not Working
1. Check your API key in `models/llm_interface.py`
2. Ensure `huggingface-hub` is installed: `pip install huggingface-hub`
3. Check internet connection

### Local Model Not Working
1. Ensure models are downloaded (run `setup_llm_model()`)
2. Check GPU/CPU memory
3. Set `USE_LOCAL_MODEL = True`

### Import Errors
```bash
pip install huggingface-hub
```

## Performance Comparison

| Feature | Local Model | API Model |
|---------|------------|-----------|
| **Speed** | Slower (depends on hardware) | Faster |
| **Privacy** | ✅ Fully private | ❌ Sent to HF servers |
| **Internet** | ❌ Not needed (after download) | ✅ Required |
| **Cost** | Free | Free tier with limits |
| **Quality** | llama3.2:1b (smaller) | SmolLM3-3B (larger) |

## Files Modified

- ✅ `models/llm_interface.py` - New unified LLM interface
- ✅ `models/note_management.py` - Updated to use LLM interface
- ✅ `models/rag_llm.py` - Updated to use LLM interface
- ✅ `main.py` - Updated to initialize LLM interface
- ✅ `requirements.txt` - Added `huggingface-hub`

## Next Steps

1. **Choose your mode**: Edit `USE_LOCAL_MODEL` in `models/llm_interface.py`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the assistant**: `python main.py`
4. **Test both modes**: Try switching between local and API to see which works better for you!

## Security Note

⚠️ **Important**: The API key is currently hardcoded in the file. For production use, consider:
- Using environment variables: `os.getenv("HF_API_KEY")`
- Storing in a config file that's gitignored
- Using a secrets management system
