import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

class EmailGenerator:
    def __init__(self, base_model_name="EleutherAI/pythia-1.4b", lora_weights_path="./"):
        print("Initializing model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("Loading LoRA weights...")
        config = PeftConfig.from_pretrained(lora_weights_path)
        
        self.model = PeftModel.from_pretrained(
            base_model,
            lora_weights_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully!")

    def generate_response(self, email_text, max_length=512):
        prompt = f"""Task: Generate a professional email response.

Input email:
{email_text}

Requirements:
1. Professional and friendly tone
2. Clear and concise
3. Address all questions
4. Maintain business etiquette

Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Response:")[-1].strip()

if __name__ == "__main__":
    generator = EmailGenerator(lora_weights_path=".")
    
    test_email = """
    Dear Support Team,
    
    I recently purchased your software but I'm having trouble with the installation.
    Could you please help me with the setup process?
    
    Best regards,
    John
    """
    
    print("\nGenerating response...")
    response = generator.generate_response(test_email)
    print("\nGenerated response:")
    print("-" * 50)
    print(response)
    print("-" * 50)