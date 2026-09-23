import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from shared.config import MODEL_NAME


class BioMistral:

    def __init__(self):
        self.device = self.get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        dtype = torch.float16 if self.device != "cpu" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype
        )

        self.model.to(self.device)
        self.model.eval()

    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def generate(self, prompt, max_new_tokens=256):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
