import json
from typing import List, Dict
import torch
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import nltk
from tqdm import tqdm

class ModelTester:
    def __init__(self, model_path: str):
        """
        Инициализация тестера модели.
        
        Args:
            model_path: Путь к обученной модели
        """
        self.config = PeftConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(
            self.config.base_model_name_or_path,
            model_path,
            torch_dtype=torch.float32
        )
        self.model.eval()
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        nltk.download('punkt')

    def generate_response(self, email_text: str, style: str = "formal", tone: str = "neutral") -> str:
        """
        Генерация ответа на email.
        
        Args:
            email_text: Текст входящего письма
            style: Стиль ответа
            tone: Тон ответа
        
        Returns:
            str: Сгенерированный ответ
        """
        prompt = f"""Задача: Напишите ответ на email.

Текст письма:
{email_text}

Инструкции:
1. Используйте {style} стиль
2. Тон ответа должен быть {tone}
3. Ответ должен быть конкретным и по существу
4. Сохраняйте профессионализм

Ответ:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Ответ:")[-1].strip()

    def evaluate_response(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Оценка качества сгенерированного ответа.
        
        Args:
            generated: Сгенерированный ответ
            reference: Эталонный ответ
        
        Returns:
            Dict[str, float]: Метрики качества
        """
        rouge_scores = self.rouge_scorer.score(reference, generated)
        
        reference_tokens = nltk.word_tokenize(reference.lower())
        generated_tokens = nltk.word_tokenize(generated.lower())
        bleu_score = sentence_bleu([reference_tokens], generated_tokens)
        
        return {
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'bleu': bleu_score
        }

    def test_on_dataset(self, test_data_path: str, num_samples: int = None) -> Dict[str, float]:
        """
        Тестирование модели на наборе данных.
        
        Args:
            test_data_path: Путь к тестовому набору данных
            num_samples: Количество примеров для тестирования (None для всего датасета)
        
        Returns:
            Dict[str, float]: Усредненные метрики
        """
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        if num_samples:
            test_data = test_data[:num_samples]
        
        total_metrics = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'bleu': 0}
        
        for example in tqdm(test_data, desc="Тестирование"):
            generated = self.generate_response(
                example['input_email'],
                example['style'],
                example.get('tone', 'neutral')
            )
            metrics = self.evaluate_response(generated, example['response'])
            
            for key in total_metrics:
                total_metrics[key] += metrics[key]
        
        for key in total_metrics:
            total_metrics[key] /= len(test_data)
        
        return total_metrics

def main():
    model_path = "models/email_generator_v1/final_model" 
    test_data_path = "data/processed/test.json" 
    
    tester = ModelTester(model_path)
    
    email = "Добрый день! Хотел уточнить статус моего заказа #12345."
    response = tester.generate_response(email, style="formal", tone="friendly")
    print(f"Сгенерированный ответ:\n{response}\n")
    
    metrics = tester.test_on_dataset(test_data_path, num_samples=100)
    print("\nМетрики качества:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 