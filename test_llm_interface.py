"""
Test script to verify LLM interface works with both local and API modes.
Run this before using main.py to ensure everything is configured correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llm_interface import LLMInterface, USE_LOCAL_MODEL

def test_api_mode():
    """Test API mode"""
    print("\n" + "="*60)
    print("Testing HUGGING FACE API MODE")
    print("="*60)
    
    try:
        llm = LLMInterface(use_local=False)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer in one sentence."}
        ]
        
        print("\nSending test query...")
        response = llm.generate(messages, max_new_tokens=50)
        
        print(f"\n✅ API Response: {response}")
        print("\n✅ API MODE WORKS!")
        return True
        
    except Exception as e:
        print(f"\n❌ API MODE FAILED: {e}")
        return False

def test_local_mode():
    """Test local mode"""
    print("\n" + "="*60)
    print("Testing LOCAL MODEL MODE")
    print("="*60)
    
    try:
        from models.model_setup import setup_llm_model
        
        print("\nLoading local models (this may take a while)...")
        llm_tokenizer, llm_model = setup_llm_model()
        
        llm = LLMInterface(
            use_local=True,
            llm_tokenizer=llm_tokenizer,
            llm_model=llm_model
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer in one sentence."}
        ]
        
        print("\nSending test query...")
        response = llm.generate(messages, max_new_tokens=50)
        
        print(f"\n✅ Local Response: {response}")
        print("\n✅ LOCAL MODE WORKS!")
        return True
        
    except Exception as e:
        print(f"\n❌ LOCAL MODE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "#"*60)
    print("# LLM INTERFACE TEST")
    print("#"*60)
    print(f"\nCurrent setting: USE_LOCAL_MODEL = {USE_LOCAL_MODEL}")
    print("\nThis will test both modes regardless of the setting.")
    
    api_works = test_api_mode()
    local_works = test_local_mode()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"API Mode:   {'✅ Working' if api_works else '❌ Failed'}")
    print(f"Local Mode: {'✅ Working' if local_works else '❌ Failed'}")
    
    if USE_LOCAL_MODEL:
        print(f"\n📌 Currently configured to use: LOCAL MODEL")
        if not local_works:
            print("⚠️  WARNING: Local mode is configured but not working!")
    else:
        print(f"\n📌 Currently configured to use: API MODEL")
        if not api_works:
            print("⚠️  WARNING: API mode is configured but not working!")
    
    print("\n💡 To change mode, edit models/llm_interface.py")
    print("   Set USE_LOCAL_MODEL = True for local")
    print("   Set USE_LOCAL_MODEL = False for API")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
