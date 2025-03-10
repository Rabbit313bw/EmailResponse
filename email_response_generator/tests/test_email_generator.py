import pytest
from src.email_generator import EmailGenerator, EmailPrompt, EmailStyle

def test_email_style_initialization():
    """Тест инициализации стилей."""
    style = EmailStyle()
    assert hasattr(style, "formal")
    assert hasattr(style, "friendly")
    assert hasattr(style, "professional")

def test_email_prompt_validation():
    """Тест валидации входных данных."""
    prompt = EmailPrompt(
        email_text="Test email",
        style="formal",
        tone="neutral"
    )
    assert prompt.email_text == "Test email"
    assert prompt.style == "formal"
    assert prompt.tone == "neutral"
    
    prompt = EmailPrompt(
        email_text="Test email",
        style="formal"
    )
    assert prompt.tone == "neutral"
    assert prompt.context is None

def test_email_generator_initialization():
    """Тест инициализации генератора."""
    generator = EmailGenerator()
    assert generator.model == "mistralai/Mistral-7B-Instruct-v0.2"
    assert hasattr(generator, "tokenizer")
    assert hasattr(generator, "pipeline")

def test_prompt_creation():
    """Тест создания промпта."""
    generator = EmailGenerator()
    
    prompt_data = EmailPrompt(
        email_text="Hello, how are you?",
        style="formal",
        tone="neutral"
    )
    
    prompt = generator._create_prompt(prompt_data)
    assert "Hello, how are you?" in prompt
    assert "формальный деловой стиль" in prompt.lower()
    
    prompt_data = EmailPrompt(
        email_text="Hello, how are you?",
        style="friendly",
        tone="positive",
        context=[
            {"from": "John", "text": "Hi there!"}
        ]
    )
    
    prompt = generator._create_prompt(prompt_data)
    assert "Предыдущая переписка:" in prompt
    assert "John" in prompt
    assert "Hi there!" in prompt

@pytest.mark.asyncio
async def test_generate_response():
    """Тест генерации ответа."""
    generator = EmailGenerator()
    
    response = generator.generate(
        email_text="Could you provide pricing information?",
        style="formal",
        tone="neutral"
    )
    
    assert isinstance(response, str)
    assert len(response) > 0

def test_invalid_style():
    """Тест обработки некорректного стиля."""
    generator = EmailGenerator()
    
    with pytest.raises(ValueError):
        generator.generate(
            email_text="Test email",
            style="invalid_style"
        )

def test_empty_email():
    """Тест обработки пустого письма."""
    generator = EmailGenerator()
    
    with pytest.raises(ValueError):
        generator.generate(
            email_text="",
            style="formal"
        )

def test_long_context():
    """Тест обработки длинного контекста."""
    generator = EmailGenerator()
    
    context = [
        {"from": f"User{i}", "text": f"Message {i}"} 
        for i in range(10)
    ]
    
    response = generator.generate(
        email_text="Test email",
        style="formal",
        context=context
    )
    
    assert isinstance(response, str)
    assert len(response) > 0

def test_temperature_bounds():
    """Тест ограничений параметра temperature."""
    generator = EmailGenerator()
    
    with pytest.raises(ValueError):
        generator.generate(
            email_text="Test email",
            style="formal",
            temperature=-0.1
        )
    
    with pytest.raises(ValueError):
        generator.generate(
            email_text="Test email",
            style="formal",
            temperature=1.5
        ) 