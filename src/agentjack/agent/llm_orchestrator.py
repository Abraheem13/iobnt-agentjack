"""LLM orchestrator: the realism demonstrator.

Its job in the paper is narrow and should stay narrow. Headline numbers come
from the learned controller, which is deterministic; this agent exists to show
that the same threat reaches a real language model driving the same tool
interface. Treating it as the primary quantitative agent would make the central
claims hostage to sampling, model versions and vendor drift.

Backends
--------
``HuggingFaceBackend``  a local open-weight model, greedy decoding, pinned
                        revision. This is what produces reportable results.
``ScriptedBackend``     a deterministic stand-in for wiring tests.

READ THIS BEFORE USING ScriptedBackend FOR ANYTHING: it must never appear in a
reported result. A scripted backend's susceptibility is whatever its author
wrote into it, so "the attack works against the mock" is circular and proves
only that the harness runs. Every susceptibility claim in the paper has to come
from a real model. ``LLMOrchestrator.is_reportable`` enforces the distinction,
and the Day 6 gate refuses to certify a run that used the scripted path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..twin.digital_twin import Observation
from .base import BaseAgent
from .memory import RollingMemory
from .policy_controller import ACTIONS, expert_action
from .prompts import DEFAULT_PROMPT_VERSION, render_observation, system_prompt
from .tools import parse_tool_call

__all__ = ["LLMOrchestrator", "HuggingFaceBackend", "ScriptedBackend"]


class ScriptedBackend:
    """Deterministic stand-in. Wiring tests only - never a reported result.

    Follows the same clinical rule the expert uses, so the harness produces
    sensible episodes without a GPU. It deliberately does NOT model
    susceptibility to persuasion: inventing that behaviour and then measuring it
    would be circular.
    """

    is_real_model = False
    name = "scripted"

    def __init__(self, obey_notes: bool = False):
        if obey_notes:
            raise ValueError(
                "ScriptedBackend refuses to simulate note-following. Susceptibility "
                "must be measured on a real model, not written into a mock."
            )

    def generate(self, system: str, user: str, observation: Observation) -> str:
        return f"deliver_dose({expert_action(observation)})"


@dataclass
class HuggingFaceBackend:
    """Local open-weight model via transformers. Greedy decoding, pinned revision."""

    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_new_tokens: int = 48
    is_real_model: bool = True
    _pipe: object | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return f"{self.model_id}@{self.revision or 'unpinned'}"

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision,
            torch_dtype=getattr(torch, self.dtype),
        ).to(self.device)
        model.eval()
        self._pipe = (tok, model)
        if self.revision is None:
            import warnings

            warnings.warn(
                "HuggingFaceBackend revision is unpinned. Pin it before producing "
                "any number that goes in the paper - an unpinned model id is not "
                "a reproducible experimental condition.",
                stacklevel=2,
            )
        return self._pipe

    def generate(self, system: str, user: str, observation: Observation) -> str:
        import torch

        tok, model = self._pipe or self._load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


@dataclass
class LLMOrchestrator(BaseAgent):
    """Renders the observation as text, asks the model, parses one tool call."""

    backend: object = field(default_factory=ScriptedBackend)
    prompt_version: str = DEFAULT_PROMPT_VERSION
    context_window: int = 32
    name: str = "llm_orchestrator"
    trace: list = field(default_factory=list)
    memory: RollingMemory = field(default=None, repr=False)

    def __post_init__(self):
        self.memory = RollingMemory(self.context_window)
        self._system = system_prompt(self.prompt_version)

    @property
    def is_reportable(self) -> bool:
        """False for any backend whose behaviour was authored rather than measured."""
        return bool(getattr(self.backend, "is_real_model", False))

    def reset(self) -> None:
        self.memory.reset()
        self.trace.clear()

    def act(self, observation: Observation) -> str:
        note = (observation.message or {}).get("note")
        user = render_observation(observation, note=note)
        raw = self.backend.generate(self._system, user, observation)
        action, _ = parse_tool_call(raw)
        if action not in ACTIONS:
            action = "none"

        self.memory.add(observation.step, (observation.message or {}).get("opcode", "none"),
                        note, action, observation.glucose)
        self.trace.append({
            "step": observation.step, "glucose": observation.glucose,
            "note": note, "prompt": user, "response": raw, "action": action,
        })
        return action

    def describe(self) -> dict:
        return {
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "is_real_model": self.is_reportable,
            "prompt_version": self.prompt_version,
            "context_window": self.context_window,
        }
