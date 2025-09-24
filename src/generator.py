import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# Removed pipeline import to avoid torchvision dependency

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False

DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

class LocalGenerator:
    """Fixed generator without pipeline dependency"""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device_map: str | None = None,
        torch_dtype=None,
        load_in_4bit: bool = False,
    ):
        print(f"[LocalGenerator] Initializing {model_name}...")
        
        # Device + dtype setup
        if torch.cuda.is_available():
            device_map = device_map or "cuda"
            if torch_dtype is None:
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = device_map or "mps"
            torch_dtype = torch_dtype or torch.float16
        else:
            device_map = device_map or "cpu"
            torch_dtype = torch_dtype or torch.float32

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": False,
        }

        # Add quantization if requested
        if load_in_4bit and HAS_BNB:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )

        # Load model with fallback
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self.device = self.model.device
            print(f"[LocalGenerator] Model loaded on {self.device}")
        except Exception as e:
            print(f"[LocalGenerator] Failed optimal loading: {e}")
            # Fallback to CPU
            model_kwargs = {"torch_dtype": torch.float32, "device_map": "cpu"}
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self.device = "cpu"
            print(f"[LocalGenerator] Fallback: Model loaded on CPU")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        pad_token_id: int | None = None,
    ) -> str:
        """Generate text completion"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set pad token
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # Generate text
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
                
                # Decode only new tokens
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                return response.strip()
                
            except Exception as e:
                print(f"[LocalGenerator] Generation failed: {e}")
                return f"Error generating response: {str(e)}"
