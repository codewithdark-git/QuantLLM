"""Text generation and chat support for QuantLLM.

Provides the :class:`GenerationMixin` used by :class:`TurboModel` for
``generate``, ``chat``, and ``_enable_flash_attention``.
"""

import sys
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel

from ..utils import (
    logger,
    print_error,
    print_warning,
)


class GenerationMixin:
    """Generation methods mixed into :class:`TurboModel`."""

    # -- declared by concrete class / other mixins (for mypy) --
    model: Any
    tokenizer: Any
    config: Any

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
        add_special_tokens: bool = True,
        **kwargs,
    ):
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (higher = more creative).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            do_sample: Whether to sample (False = greedy).
            stream: Whether to stream output token by token (returns generator).
            repetition_penalty: Penalty for repeating tokens.
            stop_strings: List of strings that stop generation.
            add_special_tokens: Whether the tokenizer adds special tokens (BOS).
                Set to False for prompts already rendered by a chat template.
        """
        self._prepare_model_for_generation()

        inputs = self._tokenize_prompt(prompt, max_new_tokens, add_special_tokens=add_special_tokens)

        stop_strings = [s for s in (stop_strings or []) if s]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 1.0,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
        }
        # Only fill in pad_token_id when the model's generation_config has none.
        # eos_token_id is deliberately NOT passed: generation_config may define
        # several EOS ids (e.g. llama-3 <|eot_id|>, gemma-2 <end_of_turn>).
        generation_config = getattr(self.model, "generation_config", None)
        if getattr(generation_config, "pad_token_id", None) is None:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if pad_token_id is not None:
                gen_kwargs["pad_token_id"] = pad_token_id
        if stop_strings:
            # Native transformers stop strings halt generation at the token level.
            gen_kwargs["stop_strings"] = stop_strings
            gen_kwargs["tokenizer"] = self.tokenizer
        gen_kwargs.update(kwargs)

        if stream:
            return self._generate_streaming(inputs, gen_kwargs, stop_strings)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        idx = _find_first_stop(response, stop_strings)
        if idx is not None:
            response = response[:idx]

        return response.strip()

    def _prepare_model_for_generation(self) -> None:
        """Put the model in inference mode (e.g. after ``finetune()``)."""
        model = self.model
        try:
            if getattr(model, "is_gradient_checkpointing", False) is True:
                model.gradient_checkpointing_disable()
            if callable(getattr(model, "eval", None)):
                model.eval()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Could not switch model to eval mode: {e}")

    def _input_device(self) -> Any:
        """Device of the input embeddings (robust with ``device_map="auto"``)."""
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return getattr(self.model, "device", torch.device("cpu"))

    def _tokenize_prompt(self, prompt: str, max_new_tokens: int, add_special_tokens: bool = True) -> Dict[str, Any]:
        """Tokenize a prompt, truncating from the left so the generation prompt survives."""
        tok_kwargs: Dict[str, Any] = {"return_tensors": "pt", "add_special_tokens": add_special_tokens}
        max_seq_length = getattr(self.config, "max_seq_length", None)
        if max_seq_length:
            tok_kwargs["truncation"] = True
            tok_kwargs["max_length"] = max(1, int(max_seq_length) - int(max_new_tokens))

        old_side = getattr(self.tokenizer, "truncation_side", None)
        try:
            if old_side is not None:
                self.tokenizer.truncation_side = "left"
            inputs = self.tokenizer(prompt, **tok_kwargs)
        finally:
            if old_side is not None:
                self.tokenizer.truncation_side = old_side

        inputs = dict(inputs)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        device = self._input_device()
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _generate_streaming(
        self,
        inputs: Dict,
        gen_kwargs: Dict[str, Any],
        stop_strings: List[str],
    ):
        """Generate with streaming output (yields text chunks).

        Stop strings are never yielded; generation is halted as soon as one is
        seen, and exceptions raised in the generation thread are re-raised here.
        """
        from threading import Event, Thread

        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        gen_kwargs = dict(gen_kwargs)
        timeout = gen_kwargs.pop("stream_timeout", 600.0)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            timeout=timeout,
            skip_special_tokens=True,
        )

        stop_event = Event()

        class _EventStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return torch.full((input_ids.shape[0],), stop_event.is_set(), dtype=torch.bool, device=input_ids.device)

        criteria = StoppingCriteriaList(list(gen_kwargs.pop("stopping_criteria", None) or []))
        criteria.append(_EventStoppingCriteria())

        generation_kwargs = {**inputs, **gen_kwargs, "streamer": streamer, "stopping_criteria": criteria}

        errors: List[BaseException] = []

        def _target():
            try:
                self.model.generate(**generation_kwargs)
            except BaseException as e:  # noqa: BLE001 - re-raised in consumer
                errors.append(e)
            finally:
                streamer.end()

        thread = Thread(target=_target, daemon=True)
        thread.start()

        # Hold back enough characters to never leak a partially-generated stop string.
        holdback = max((len(s) for s in stop_strings), default=1) - 1
        buffer = ""
        emitted = 0
        stopped = False
        try:
            for new_text in streamer:
                if not new_text:
                    continue
                buffer += new_text
                idx = _find_first_stop(buffer, stop_strings)
                if idx is not None:
                    if idx > emitted:
                        yield buffer[emitted:idx]
                    emitted = idx
                    stopped = True
                    break
                safe = len(buffer) - holdback
                if safe > emitted:
                    yield buffer[emitted:safe]
                    emitted = safe
            if not stopped and not errors and len(buffer) > emitted:
                yield buffer[emitted:]
        finally:
            stop_event.set()
            thread.join()

        if errors:
            raise errors[0]

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
        templated = False

        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                templated = isinstance(prompt, str)
                if not templated:
                    prompt = None
            except Exception:
                prompt = None

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

        if templated:
            # The rendered template already contains BOS; don't add it twice.
            kwargs.setdefault("add_special_tokens", False)
        return self.generate(prompt, **kwargs)

    @staticmethod
    def _enable_flash_attention(model: PreTrainedModel, verbose: bool = True) -> None:
        """Best-effort switch of the attention implementation on a loaded model.

        Mutating ``config._attn_implementation`` after load is a no-op (or
        harmful) on most transformers versions, so this only acts through
        ``model.set_attn_implementation`` when the model provides it. The
        preferred path is passing ``attn_implementation=`` to
        ``from_pretrained``.
        """
        setter = getattr(model, "set_attn_implementation", None)
        if not callable(setter):
            logger.debug("  set_attn_implementation not available; leaving attention implementation unchanged")
            return

        try:
            import flash_attn  # noqa: F401

            impl = "flash_attention_2"
        except ImportError:
            impl = "sdpa"

        try:
            setter(impl)
            if verbose:
                logger.info(f"  Attention implementation set to {impl}")
        except Exception as e:
            if verbose:
                logger.warning(f"  Could not set attention implementation to {impl}: {e}")


def _find_first_stop(text: str, stop_strings: List[str]) -> Optional[int]:
    """Return the index of the earliest stop string occurrence in ``text``."""
    positions = [i for i in (text.find(s) for s in stop_strings if s) if i != -1]
    return min(positions) if positions else None
