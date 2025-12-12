# Text Generation

Generate text with various options and modes.

## Basic Generation

```python
from quantllm import turbo

model = turbo("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

response = model.generate("What is machine learning?")
print(response)
```

## Generation Parameters

```python
response = model.generate(
    prompt="Explain quantum computing.",
    max_new_tokens=200,        # Maximum tokens to generate
    temperature=0.7,           # Creativity (0.0 = deterministic, 1.0 = creative)
    top_p=0.9,                 # Nucleus sampling
    top_k=50,                  # Top-k sampling
    repetition_penalty=1.1,    # Prevent repetition
    do_sample=True,            # Enable sampling
)
```

## Chat Mode

For conversational models:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

response = model.chat(messages, max_new_tokens=100)
print(response)

# Continue conversation
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "What about JavaScript?"})

response = model.chat(messages)
```

## Streaming

Get tokens as they're generated:

```python
# Streaming generation
for token in model.generate("Count from 1 to 10:", stream=True):
    print(token, end="", flush=True)
print()  # Newline at end
```

## Stop Strings

Stop generation at specific patterns:

```python
response = model.generate(
    "Write a poem:\n",
    max_new_tokens=200,
    stop_strings=["The End", "\n\n\n"],
)
```

## Batch Generation

Generate multiple responses:

```python
prompts = [
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?",
]

for prompt in prompts:
    response = model.generate(prompt, max_new_tokens=50)
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

## Best Practices

1. **Temperature**: Use 0.1-0.3 for factual, 0.7-0.9 for creative
2. **Max tokens**: Set reasonable limits to avoid runaway generation
3. **Repetition penalty**: Use 1.1-1.2 to reduce repetition
4. **Streaming**: Use for long responses to improve UX
