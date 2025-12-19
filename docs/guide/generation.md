# 💬 Text Generation

Generate text with various options and modes.

---

## Basic Generation

```python
from quantllm import turbo

model = turbo("meta-llama/Llama-3.2-3B")

response = model.generate("What is machine learning?")
print(response)
```

---

## Generation Parameters

### Temperature & Sampling

```python
response = model.generate(
    "Write a creative story about a robot.",
    max_new_tokens=200,        # Maximum tokens to generate
    temperature=0.7,           # Creativity (0.0 = deterministic, 1.0+ = creative)
    top_p=0.9,                 # Nucleus sampling (higher = more diverse)
    top_k=50,                  # Top-k sampling
    do_sample=True,            # Enable sampling (required for temperature > 0)
)
```

### Controlling Output

```python
response = model.generate(
    "List 5 programming languages:",
    max_new_tokens=100,
    repetition_penalty=1.1,    # Prevent repetition (1.0 = off, 1.2 = strong)
    no_repeat_ngram_size=3,    # Prevent repeating n-grams
)
```

### Parameter Guide

| Parameter | Range | Description |
|-----------|-------|-------------|
| `temperature` | 0.0-2.0 | 0.1-0.3 for factual, 0.7-0.9 for creative |
| `top_p` | 0.0-1.0 | 0.9 is a good default |
| `top_k` | 1-100 | 50 is a good default |
| `repetition_penalty` | 1.0-1.5 | 1.1-1.2 prevents repetition |
| `max_new_tokens` | 1-4096+ | Depends on model context length |

---

## Chat Mode

For conversational models with system prompts:

```python
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I read a file in Python?"},
]

response = model.chat(messages, max_new_tokens=200)
print(response)
```

### Multi-Turn Conversation

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

# First response
response = model.chat(messages)
print(f"Assistant: {response}")

# Continue conversation
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "What about JavaScript?"})

response = model.chat(messages)
print(f"Assistant: {response}")
```

---

## Streaming

Get tokens as they're generated for better UX:

```python
# Streaming generation
for token in model.generate("Write a poem about the ocean:", stream=True):
    print(token, end="", flush=True)
print()  # Newline at end
```

### Streaming with Chat

```python
messages = [{"role": "user", "content": "Tell me a story."}]

for token in model.chat(messages, stream=True):
    print(token, end="", flush=True)
```

---

## Stop Strings

Stop generation at specific patterns:

```python
response = model.generate(
    "Write a haiku:\n",
    max_new_tokens=100,
    stop_strings=["---", "\n\n\n"],  # Stop at these patterns
)
```

---

## Batch Generation

Generate multiple responses efficiently:

```python
prompts = [
    "What is Python?",
    "What is JavaScript?", 
    "What is Rust?",
]

for prompt in prompts:
    response = model.generate(prompt, max_new_tokens=100)
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

---

## Common Use Cases

### Factual Q&A

```python
response = model.generate(
    "What is the capital of France?",
    temperature=0.1,           # Low temperature for factual
    max_new_tokens=50,
)
```

### Creative Writing

```python
response = model.generate(
    "Write a short story about a dragon:",
    temperature=0.8,           # Higher temperature for creativity
    top_p=0.95,
    max_new_tokens=500,
)
```

### Code Generation

```python
response = model.generate(
    "Write a Python function to sort a list:",
    temperature=0.2,           # Low for accurate code
    max_new_tokens=200,
)
```

### Summarization

```python
text = "..." # Long text to summarize
response = model.generate(
    f"Summarize the following text:\n\n{text}\n\nSummary:",
    temperature=0.3,
    max_new_tokens=150,
)
```

---

## Best Practices

1. **Temperature**: Use 0.1-0.3 for factual, 0.7-0.9 for creative
2. **Max tokens**: Set reasonable limits to avoid runaway generation
3. **Repetition penalty**: Use 1.1-1.2 to reduce repetition
4. **Streaming**: Use for long responses to improve user experience
5. **Stop strings**: Define clear stopping points for structured output

---

## Next Steps

- [Fine-tuning →](finetuning.md) — Train the model on your data
- [GGUF Export →](gguf-export.md) — Export for deployment
- [API Reference →](../api/turbomodel.md) — Full API documentation
