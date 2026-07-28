"""Text generation and chat support for QuantLLM.

Provides the :class:`GenerationMixin` used by :class:`TurboModel` for
``generate``, ``chat``, and ``_enable_flash_attention``.
"""

import sys
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..utils import (
    logger,
    print_error,
    print_warning,
)


class GenerationMixin:
    """Generation methods mixed into :class:`TurboModel`."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        stream: bool = False,
        repetition_penalty: float = 1.1,
        stop_strings: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (higher = more creative).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            do_sample: Whether to sample (False = greedy).
            stream: Whether to stream output token by token.
            repetition_penalty: Penalty for repeating tokens.
            stop_strings: List of strings that stop generation.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length - max_new_tokens,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        stop_strings = stop_strings or []

        if stream:
            return self._generate_streaming(
                inputs,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                do_sample,
                repetition_penalty,
                stop_strings,
                **kwargs,
            )

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                **kwargs,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        for stop in stop_strings:
            if stop in response:
                response = response.split(stop)[0]
                break

        return response.strip()

    def _generate_streaming(
        self,
        inputs: Dict,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        repetition_penalty: float,
        stop_strings: List[str],
        **kwargs,
    ) -> str:
        """Generate with streaming output."""
        try:
            from transformers import TextIteratorStreamer
            from threading import Thread

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature if do_sample else 1.0,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": repetition_penalty,
                "streamer": streamer,
                **kwargs,
            }

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            generated_text = []
            for new_text in streamer:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                generated_text.append(new_text)

                full_text = "".join(generated_text)
                should_stop = any(stop in full_text for stop in stop_strings)
                if should_stop:
                    break

            thread.join()
            print()

            response = "".join(generated_text)
            for stop in stop_strings:
                if stop in response:
                    response = response.split(stop)[0]

            return response.strip()

        except ImportError:
            print("(Streaming not available, using batch generation)")
            return self.generate(
                self.tokenizer.decode(inputs["input_ids"][0]),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stream=False,
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """Generate response for chat-format messages.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            **kwargs: Additional generation parameters.
        """
        prompt = None

        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        if prompt is None:
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    parts.append(f"System: {content}\n")
                elif role == "user":
                    parts.append(f"User: {content}\n")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}\n")
            parts.append("Assistant:")
            prompt = "".join(parts)

        return self.generate(prompt, **kwargs)

    @staticmethod
    def _enable_flash_attention(model: PreTrainedModel, verbose: bool = True) -> None:
        """Enable Flash Attention if available."""
        try:
            if hasattr(model, "config"):
                if hasattr(model.config, "_attn_implementation"):
                    try:
                        import flash_attn

                        model.config._attn_implementation = "flash_attention_2"
                        if verbose:
                            logger.info("  Flash Attention 2 configured")
                    except ImportError:
                        model.config._attn_implementation = "sdpa"
                        if verbose:
                            logger.info("  SDPA (Scaled Dot-Product Attention) configured")
        except Exception:
            if verbose:
                logger.warning("  Flash Attention / SDPA not available")
