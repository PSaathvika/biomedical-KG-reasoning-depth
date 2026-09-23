import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import MODEL_NAME


class BioMistral:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.device = self.get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def generate(self, prompt, max_new_tokens=128):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
