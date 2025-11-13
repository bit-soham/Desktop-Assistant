"""
Unified LLM Interface
Supports both local models (llama3.2:1b) and Hugging Face Inference API
"""

import os
import yaml
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env
load_dotenv()

# Load configuration from config.yaml
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Configuration from files
USE_LOCAL_MODEL = config['llm']['use_local_model']
HF_API_KEY = os.getenv('HF_TOKEN')  # Assuming HF_TOKEN is the key in .env
HF_MODEL = config['llm']['hf_model']

# ANSI escape codes for colors
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET_COLOR = '\033[0m'


class LLMInterface:
    """
    Unified interface for LLM generation.
    Supports both local models and Hugging Face Inference API.
    """
    
    def __init__(self, use_local=USE_LOCAL_MODEL, llm_tokenizer=None, llm_model=None, 
                 hf_api_key=HF_API_KEY, hf_model=HF_MODEL):
        """
        Initialize the LLM interface.
        
        Args:
            use_local (bool): If True, use local model. If False, use HF API.
            llm_tokenizer: Local model tokenizer (required if use_local=True)
            llm_model: Local model (required if use_local=True)
            hf_api_key (str): Hugging Face API key (required if use_local=False)
            hf_model (str): Hugging Face model name (required if use_local=False)
        """
        self.use_local = use_local
        
        if self.use_local:
            if llm_tokenizer is None or llm_model is None:
                raise ValueError("llm_tokenizer and llm_model must be provided when use_local=True")
            self.tokenizer = llm_tokenizer
            self.model = llm_model
            print(CYAN + "DEBUG: LLM Interface initialized with LOCAL model" + RESET_COLOR)
        else:
            if not hf_api_key:
                raise ValueError("hf_api_key must be provided when use_local=False")
            self.client = InferenceClient(
                provider="hf-inference",
                api_key=hf_api_key,
            )
            self.hf_model = hf_model
            print(CYAN + f"DEBUG: LLM Interface initialized with HF API (model: {hf_model})" + RESET_COLOR)
    
    def generate(self, messages, max_new_tokens=100, temperature=0.7):
        """
        Generate a response from the LLM.
        
        Args:
            messages (list): List of message dicts with 'role' and 'content'
                            Example: [{"role": "system", "content": "..."}, 
                                     {"role": "user", "content": "..."}]
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature (0.0 to 1.0)
        
        Returns:
            str: Generated text response
        """
        if self.use_local:
            return self._generate_local(messages, max_new_tokens)
        else:
            return self._generate_api(messages, max_new_tokens, temperature)
    
    def _generate_local(self, messages, max_new_tokens):
        """Generate using local model."""
        print(f"DEBUG: Generating with LOCAL model (max_tokens={max_new_tokens})")
        
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Ensure pad_token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create attention mask if not present
        if 'attention_mask' not in inputs:
            import torch
            inputs['attention_mask'] = (inputs['input_ids'] != self.tokenizer.pad_token_id).long()
        
        # Generate
        outputs = self.model.generate(
            **inputs, 
            pad_token_id=self.tokenizer.eos_token_id, 
            max_new_tokens=max_new_tokens
        )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=False
        ).strip()
        
        # Clean up special tokens
        response = response.replace("<|eot_id|>", "").strip()
        response = response.replace("<|end_of_text|>", "").strip()
        
        return response
    
    def _generate_api(self, messages, max_new_tokens, temperature):
        """Generate using Hugging Face Inference API."""
        print(f"DEBUG: Generating with HF API (model={self.hf_model}, max_tokens={max_new_tokens})")
        
        try:
            completion = self.client.chat.completions.create(
                model=self.hf_model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            response = completion.choices[0].message.content
            
            # Remove thinking tags if present (some models use this)
            if "<think>" in response:
                # Extract content after </think>
                think_end = response.find("</think>")
                if think_end != -1:
                    response = response[think_end + 8:].strip()
            
            return response
            
        except Exception as e:
            print(YELLOW + f"WARNING: API generation failed: {e}" + RESET_COLOR)
            print(YELLOW + "Returning empty response." + RESET_COLOR)
            return ""
    
    def generate_streaming(self, messages, max_new_tokens=512, temperature=0.7):
        """
        Generate a response with streaming (yields tokens as they're generated).
        
        Args:
            messages (list): List of message dicts
            max_new_tokens (int): Maximum number of new tokens
            temperature (float): Sampling temperature
        
        Yields:
            str: Generated tokens
        """
        if self.use_local:
            # Local streaming not implemented yet, return full response
            response = self._generate_local(messages, max_new_tokens)
            yield response
        else:
            # API streaming
            print(f"DEBUG: Streaming generation with HF API (model={self.hf_model})")
            try:
                stream = self.client.chat.completions.create(
                    model=self.hf_model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stream=True,
                )
                
                in_thinking = False
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        
                        # Filter out thinking tags
                        if "<think>" in token:
                            in_thinking = True
                            continue
                        if "</think>" in token:
                            in_thinking = False
                            continue
                        if in_thinking:
                            continue
                            
                        yield token
                        
            except Exception as e:
                print(YELLOW + f"WARNING: API streaming failed: {e}" + RESET_COLOR)
                yield ""


# Convenience function for quick usage
def create_llm_interface(use_local=USE_LOCAL_MODEL, llm_tokenizer=None, llm_model=None):
    """
    Create an LLM interface instance.
    
    Args:
        use_local (bool): If True, use local model. If False, use HF API.
        llm_tokenizer: Local model tokenizer (required if use_local=True)
        llm_model: Local model (required if use_local=True)
    
    Returns:
        LLMInterface: Configured LLM interface
    """
    return LLMInterface(
        use_local=use_local,
        llm_tokenizer=llm_tokenizer,
        llm_model=llm_model,
    )
