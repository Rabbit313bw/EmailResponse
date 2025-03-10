from typing import Dict, Optional, List
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

@dataclass
class EmailStyle:
    formal: str = "Use formal business style with appropriate greetings and closings"
    friendly: str = "Use friendly and approachable tone while maintaining professionalism"
    professional: str = "Use clear, professional language focused on providing value"

class EmailPrompt(BaseModel):
    email_text: str
    style: str
    tone: Optional[str] = "neutral"
    context: Optional[List[Dict[str, str]]] = None

class EmailGenerator:
    def __init__(
        self,
        model: str = "distilgpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize email response generator.

        Args:
            model: Model name or path
            device: Device for inference ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.model.to(device)
        self.styles = EmailStyle()

    def _create_prompt(self, email_data: EmailPrompt) -> str:
        """
        Create a prompt for the model based on input data.

        Args:
            email_data: Data for response generation

        Returns:
            str: Prepared prompt for the model
        """
        style_instruction = getattr(self.styles, email_data.style, self.styles.formal)
        
        context_str = ""
        if email_data.context:
            context_str = "Previous correspondence:\n"
            for msg in email_data.context:
                context_str += f"From: {msg['from']}\n{msg['text']}\n\n"

        prompt = f"""Write a professional customer service email response.

{context_str}
Incoming email:
{email_data.email_text}

Guidelines:
1. {style_instruction}
2. Use {email_data.tone} tone
3. Address all questions directly
4. Provide specific information
5. Be concise and clear
6. Use standard business email format

Response:
Dear """
        return prompt

    def generate(
        self,
        email_text: str,
        style: str = "formal",
        tone: str = "neutral",
        context: Optional[List[Dict[str, str]]] = None,
        max_length: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate email response.

        Args:
            email_text: Input email text
            style: Response style ('formal', 'friendly', 'professional')
            tone: Response tone
            context: Previous correspondence context
            max_length: Maximum response length
            temperature: Generation temperature (0.0 - 1.0)

        Returns:
            str: Generated response
        """
        email_data = EmailPrompt(
            email_text=email_text,
            style=style,
            tone=tone,
            context=context
        )
        
        prompt = self._create_prompt(email_data)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated response after "Response:"
        response = response.split("Response:")[-1].strip()
        
        return response

    def __call__(self, *args, **kwargs) -> str:
        """Convenient way to call generation."""
        return self.generate(*args, **kwargs) 