"""
Verify Ministral3ForCausalLM.load_weights strips model.language_model.* prefix.

Plan ref: docs/plans/mistral-medium-3-5-eagle3.md execution addendum + CLAUDE.md rule #26 step 7.

Mistral-Medium-3.5 ships its safetensors under the multimodal-wrapper layout:
keys are `model.language_model.layers.N.*`. When Ministral3ForCausalLM is
loaded directly via the architectures override (skipping the multimodal
wrapper), the inherited Llama loader can't find these keys without prefix
stripping. We added a `load_weights` override that strips the prefix and
skips vision_tower/multi_modal_projector tensors.

This is a static property test — no CUDA needed.
"""
import unittest


class TestMinistral3PrefixStrip(unittest.TestCase):
    def test_load_weights_overridden(self):
        """Ministral3ForCausalLM must override load_weights (not just inherit)."""
        from sglang.srt.models.ministral3 import Ministral3ForCausalLM

        # __dict__ contains class-defined methods; inherited ones are NOT in __dict__.
        self.assertIn(
            "load_weights",
            Ministral3ForCausalLM.__dict__,
            "load_weights override missing — multimodal-wrapper Mistral-Medium-3.5 will fail to load.",
        )

    def test_prefix_strip_via_generator(self):
        """
        Drive the override's generator with synthetic weights and verify
        the prefix is stripped + vision tensors are filtered.
        """
        from sglang.srt.models.ministral3 import Ministral3ForCausalLM

        # Capture what the override forwards to super().load_weights.
        captured = []

        class FakeSuper:
            @staticmethod
            def load_weights(generator):
                captured.extend(list(generator))

        # Monkey-patch super() — simpler: just exercise the generator logic
        # by extracting the inner closure. We can do this by calling the
        # method with a stub that records its argument.
        # Simplest: use the actual method but intercept super().load_weights.
        import unittest.mock as mock

        sample_weights = [
            ("model.language_model.embed_tokens.weight", "T_EMBED"),
            ("model.language_model.layers.0.input_layernorm.weight", "T_LN0"),
            ("model.language_model.layers.0.self_attn.q_proj.weight", "T_Q"),
            ("model.language_model.layers.85.self_attn.q_proj.weight", "T_Q85"),
            ("lm_head.weight", "T_LMHEAD"),
            # Should be skipped:
            ("model.vision_tower.transformer.layers.0.weight", "T_VISION"),
            ("model.multi_modal_projector.linear_1.weight", "T_PROJ"),
        ]

        # Patch LlamaForCausalLM.load_weights to capture the generator output.
        import sglang.srt.models.llama as llama_mod

        with mock.patch.object(
            llama_mod.LlamaForCausalLM,
            "load_weights",
            lambda self, gen: captured.extend(list(gen)),
        ):
            # We don't need a fully constructed instance — bound-method dispatch
            # works as long as `self` is an instance with `__class__` set right.
            instance = Ministral3ForCausalLM.__new__(Ministral3ForCausalLM)
            Ministral3ForCausalLM.load_weights(instance, sample_weights)

        captured_dict = dict(captured)

        # Prefix-stripped keys present:
        self.assertIn("model.embed_tokens.weight", captured_dict)
        self.assertIn("model.layers.0.input_layernorm.weight", captured_dict)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", captured_dict)
        self.assertIn("model.layers.85.self_attn.q_proj.weight", captured_dict)
        # lm_head stays at its original location (not under language_model):
        self.assertIn("lm_head.weight", captured_dict)
        # Vision tensors skipped:
        self.assertNotIn("model.vision_tower.transformer.layers.0.weight", captured_dict)
        self.assertNotIn("model.multi_modal_projector.linear_1.weight", captured_dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
