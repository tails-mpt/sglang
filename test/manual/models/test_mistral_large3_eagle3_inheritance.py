"""
Phase 1 verification — Mistral-Medium-3.5 (Mistral3 family) Eagle3 aux-capture inheritance.

Plan: ../../../../docs/plans/mistral-medium-3-5-eagle3.md (Phase 1)

Purpose
-------
Verify, without instantiating the 128B model, that `MistralLarge3ForCausalLM`
correctly inherits the Eagle3 aux-state capture mechanism from
`DeepseekV2ForCausalLM`. This is the plan's "verification, not port" thesis:
because `mistral_large_3.py` does NOT override `__init__` (only `load_weights`),
its instance has `self.model = DeepseekV2Model`, and the per-layer hook +
post-loop trap inherited from `deepseek_v2.py` fire automatically.

Also documents the registry resolution path for Mistral-Medium-3.5's
on-disk `config.json` (which publishes `architectures: ["Mistral3ForConditionalGeneration"]`,
the multimodal wrapper — NOT the text head we want for Eagle3 training).
This is resolved at deployment time by overriding the `architectures` field
on the downloaded `config.json` (single-line edit, reversible). See plan E2.

Run
---
    python test/manual/models/test_mistral_large3_eagle3_inheritance.py

CPU-only. No GPUs, no model weights. The full integration verification is
done by Phase 3 smoke + Phase 0 #5 baseline bench.
"""
import inspect
import unittest


class TestMistralLarge3Eagle3Inheritance(unittest.TestCase):
    """Static-property + registry assertions on the inheritance chain."""

    def test_mistral_large3_does_not_override_init(self):
        """
        The plan's central thesis: `mistral_large_3.py` only overrides
        `load_weights`, not `__init__`. So `self.model = DeepseekV2Model`
        comes through unmodified, and the inherited per-layer hook +
        post-loop trap fire on the production forward path.

        If this ever changes (someone overrides __init__ in mistral_large_3.py
        and substitutes a different inner model class), the inheritance
        thesis breaks and Phase 1's "no port needed" assumption is invalid.
        """
        from sglang.srt.models.mistral_large_3 import MistralLarge3ForCausalLM
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        # `__init__` is inherited via the MRO chain
        # (MistralLarge3ForCausalLM → DeepseekV3ForCausalLM → DeepseekV2ForCausalLM).
        # Walk up and confirm `__init__` was NOT overridden in mistral_large_3.
        own_dict = MistralLarge3ForCausalLM.__dict__
        self.assertNotIn(
            "__init__",
            own_dict,
            f"mistral_large_3.py overrides __init__ — inheritance thesis broken. "
            f"Phase 1 needs reassessment.",
        )
        # And confirm load_weights IS overridden (the only thing this subclass
        # owns — the regex-driven Mistral-key remapping).
        self.assertIn(
            "load_weights",
            own_dict,
            f"mistral_large_3.py no longer overrides load_weights — "
            f"key remapping may have been removed.",
        )

    def test_mistral_large3_is_subclass_of_deepseek_v2(self):
        """
        The aux-capture mechanism lives on `DeepseekV2ForCausalLM`. Confirm
        `MistralLarge3ForCausalLM` is in its MRO.
        """
        from sglang.srt.models.mistral_large_3 import MistralLarge3ForCausalLM
        from sglang.srt.models.deepseek_v2 import (
            DeepseekV2ForCausalLM,
            DeepseekV3ForCausalLM,
        )

        self.assertTrue(issubclass(MistralLarge3ForCausalLM, DeepseekV3ForCausalLM))
        self.assertTrue(issubclass(MistralLarge3ForCausalLM, DeepseekV2ForCausalLM))

    def test_set_eagle3_layers_to_capture_method_exists(self):
        """
        `set_eagle3_layers_to_capture` is the entrypoint that wires up
        aux capture. It's defined on `DeepseekV2ForCausalLM` and inherited
        all the way to `MistralLarge3ForCausalLM`.
        """
        from sglang.srt.models.mistral_large_3 import MistralLarge3ForCausalLM

        self.assertTrue(
            hasattr(MistralLarge3ForCausalLM, "set_eagle3_layers_to_capture"),
            "MistralLarge3ForCausalLM does not expose set_eagle3_layers_to_capture; "
            "the inherited method is hidden somehow.",
        )
        # Confirm it's actually the DeepseekV2 implementation that is used
        # (not overridden somewhere along the chain).
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        ds_impl = DeepseekV2ForCausalLM.set_eagle3_layers_to_capture
        ml3_impl = MistralLarge3ForCausalLM.set_eagle3_layers_to_capture
        self.assertIs(
            ml3_impl,
            ds_impl,
            "MistralLarge3ForCausalLM.set_eagle3_layers_to_capture is not the "
            "inherited DeepseekV2 implementation — somebody overrode it. "
            "This may indicate the +1 offset behavior is now different.",
        )

    def test_set_eagle3_layers_to_capture_has_plus_one_offset(self):
        """
        The inherited implementation does `[val + 1 for val in layer_ids]`
        when assigning to `self.model.layers_to_capture`. The plan's aux-layer
        triple [1, 42, 84] therefore becomes [2, 43, 85] internally.

        This is a static-source assertion: read the function body and verify
        the `+1` arithmetic is present. Doing this as a string check is
        brittle but cheap; the alternative is to instantiate a model, which
        is impossible on CPU at full size.
        """
        from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

        src = inspect.getsource(DeepseekV2ForCausalLM.set_eagle3_layers_to_capture)
        self.assertIn(
            "val + 1",
            src,
            f"+1 offset arithmetic missing from set_eagle3_layers_to_capture. "
            f"Source:\n{src}",
        )
        self.assertIn(
            "layers_to_capture",
            src,
            f"layers_to_capture attribute write missing.",
        )

    def test_registry_has_both_mistral_large3_and_wrapper(self):
        """
        Confirms both classes exist in the registry:
          - MistralLarge3ForCausalLM  (text-only head; what we want for Eagle3)
          - Mistral3ForConditionalGeneration  (multimodal wrapper; what
            Mistral-Medium-3.5's stock config.json publishes in `architectures`)

        Implication for deployment: the on-disk `config.json` must be
        overridden to set `architectures: ["MistralLarge3ForCausalLM"]`
        before sglang loads, OR the eventual production path must be
        adapted. See plan E2 + experiments/Mistral-Medium-3.5/architecture-notes.md.
        """
        from sglang.srt.models.registry import import_model_classes

        classes = import_model_classes("sglang.srt.models")

        self.assertIn(
            "MistralLarge3ForCausalLM",
            classes,
            f"MistralLarge3ForCausalLM not registered. "
            f"Registered: {sorted(classes.keys())[:20]}",
        )
        self.assertIn(
            "Mistral3ForConditionalGeneration",
            classes,
            f"Mistral3ForConditionalGeneration (the wrapper) not registered. "
            f"Registered: {sorted(classes.keys())[:20]}",
        )

        # Document which one would resolve from Medium-3.5's stock config.
        # `architectures: ["Mistral3ForConditionalGeneration"]` resolves to
        # the wrapper, NOT the text head. This test does not "fail" on that
        # — it documents it. The override happens at deployment time.
        from sglang.srt.models.registry import ModelRegistry

        stock_arch = "Mistral3ForConditionalGeneration"
        cls, _ = ModelRegistry.resolve_model_cls([stock_arch])
        # Confirm: stock config picks the wrapper.
        self.assertIs(
            cls,
            classes["Mistral3ForConditionalGeneration"],
            f"Stock arch resolves to {cls.__name__}, expected wrapper.",
        )

        # And our deployment-time override picks the head.
        override_arch = "MistralLarge3ForCausalLM"
        cls, _ = ModelRegistry.resolve_model_cls([override_arch])
        self.assertIs(
            cls,
            classes["MistralLarge3ForCausalLM"],
            f"Override arch resolves to {cls.__name__}, expected MistralLarge3ForCausalLM.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
