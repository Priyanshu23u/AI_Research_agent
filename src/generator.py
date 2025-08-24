from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False

DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

class LocalGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device_map: str | None = None,
        torch_dtype=None,
        load_in_4bit: bool = False,
    ):
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

        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "trust_remote_code": False,
        }

        # Optional 4-bit quantization
        if load_in_4bit:
            if not HAS_BNB:
                raise RuntimeError("4-bit quantization requested but bitsandbytes is not installed. pip install bitsandbytes")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_config

        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load local model '{model_name}': {e}")

        # Setup pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device_map == "cuda" else -1  # cuda:0 or cpu
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1000,
        temperature: float = 0.5,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> str:
        # Truncate overly long prompt
        max_prompt_chars = 12000
        if len(prompt) > max_prompt_chars:
            prompt = prompt[-max_prompt_chars:]

        # Generate
        out = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # Remove prompt from output (to get only new text)
        if out.startswith(prompt):
            out = out[len(prompt):]
        return out.strip()
