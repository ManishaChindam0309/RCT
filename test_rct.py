
import torch

from model.transformer import RCTModel


def main():

    torch.manual_seed(42)

    model = RCTModel(
        vocab_size=1000,
        hidden_size=64,
        layers=2,
        heads=4,
        max_seq_len=64
    )

    model.eval()

    # Create four identical prompts to test exact computation reuse.
    base_tokens = torch.randint(
        0,
        1000,
        (1, 8)
    )

    tokens = base_tokens.repeat(4, 1)

    print("\n===== RCT 0.2 DEDUPLICATION TEST =====")

    with torch.inference_mode():

        # 1. Standard execution.
        standard_output = model(
            tokens,
            ffn_mode="standard"
        )

        # 2. Reset statistics.
        model.reset_dedup_statistics()

        # 3. Deduplicated execution.
        deduplicated_output = model(
            tokens,
            ffn_mode="deduplicated"
        )

        print("\nOutput shape:", tuple(standard_output.shape))

        exact_equality = torch.equal(
            standard_output,
            deduplicated_output
        )

        numerical_equality = torch.allclose(
            standard_output,
            deduplicated_output,
            atol=1e-5,
            rtol=1e-5
        )

        print("Bitwise output equality:", exact_equality)
        print("Numerical output equality:", numerical_equality)

        max_difference = (
            standard_output - deduplicated_output
        ).abs().max().item()

        print("Maximum difference:", max_difference)

        # 4. Display FFN deduplication statistics.
        print("\nFFN deduplication statistics:")

        statistics = model.dedup_statistics()

        for layer, stats in statistics.items():
            print(layer, stats)

        total_inputs = sum(
            stats["total_inputs"]
            for stats in statistics.values()
        )

        unique_inputs = sum(
            stats["unique_inputs"]
            for stats in statistics.values()
        )

        overall_reuse = (
            1 - unique_inputs / total_inputs
            if total_inputs > 0
            else 0.0
        )

        print(
            "\nOverall FFN reuse ratio:",
            f"{overall_reuse:.4%}"
        )

        # 5. Test causal attention.
        single_prompt = base_tokens.clone()

        changed_tokens = single_prompt.clone()

        changed_tokens[:, -1] = (
            changed_tokens[:, -1] + 1
        ) % model.vocab_size

        original_logits = model(single_prompt)

        changed_logits = model(changed_tokens)

        causal_correct = torch.allclose(
            original_logits[:, :-1, :],
            changed_logits[:, :-1, :],
            atol=1e-6,
            rtol=1e-5
        )

        print(
            "\nCausal attention correct:",
            causal_correct
        )

        # 6. Test autoregressive generation.
        generated = model.generate(
            single_prompt[:, :3].clone(),
            max_new_tokens=5,
            temperature=1.0,
            top_k=20
        )

        print(
            "\nGenerated token shape:",
            tuple(generated.shape)
        )

        # 7. Assertions.
        assert numerical_equality, (
            "Standard and deduplicated outputs differ "
            "beyond numerical tolerance."
        )

        assert all(
            stats["calls"] > 0
            for stats in statistics.values()
        ), "Deduplication was not activated in every layer."

        assert overall_reuse > 0, (
            "No duplicate hidden-state vectors were reused."
        )

        assert causal_correct, (
            "Causal attention test failed."
        )

        assert tuple(generated.shape) == (1, 8), (
            "Unexpected generation shape."
        )

    print("\n===== ALL RCT TESTS PASSED =====")


if __name__ == "__main__":
    main()
