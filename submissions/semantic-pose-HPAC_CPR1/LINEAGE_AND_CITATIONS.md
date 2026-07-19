# semantic-pose-HPAC_CPR1 lineage and citations

- **Submission:** `semantic-pose-HPAC_CPR1`
- **Pull request:** [commaai/comma_video_compression_challenge#130][pr130]
- **As of:** 2026-07-19
- **Scope:** Technical provenance and attribution within the public comma video
  compression challenge lineage. This is not a global patent search or legal
  opinion.

## Executive conclusion

This submission does not claim to introduce the semantic-token plus HPAC codec
family. Its direct public predecessor is jas0xf's merged
[`jas0xf_adversarial_neural_representation` PR #86][pr86]. PR #86 already
combined:

- a semantic class-token stream rendered into RGB;
- patch-group hierarchical autoregressive coding;
- Type-A and Type-B masked convolutions;
- per-frame FiLM conditioning;
- previous-frame token context;
- arithmetic coding; and
- asymmetric master and slave frame renderers.

Those mechanisms are described in jas0xf's technical write-up, particularly
the renderer discussion on page 10 and HPAC discussion on pages 20-21 [3].
They are also present in the immutable merged source [4]. The HPAC architecture
itself derives from Li et al.'s *Rethinking Autoregressive Models for Lossless
Image Compression via Hierarchical Parallelism and Progressive Adaptation*
[1].

The broad ideas of mask/pose-conditioned generation and low-rank spatial pose
actuation also have earlier public challenge precedent in EthanYangTW's qpose
submissions, PRs [#67][pr67] and [#79][pr79]. The qpose decoder constructs a
DCT basis and applies per-frame coefficients with an
`einsum(coefficients, basis)` pattern [7].

Accordingly, this work is a **derived challenge-specific extension**. Its
strongest original contributions within the audited lineage are:

1. an exact integer-lattice implementation of the inherited HPAC backbone,
   with bounded integer intermediates, dyadic requantization, canonical
   entropy logits, and cross-device symbol/token verification [8, 9];
2. a standalone learned low-rank neutral-gray pose carrier replacing PR #86's
   NeRV slave renderer [8];
3. an exact deterministic carrier repack using canonical Huffman and Rice
   streams, with decoded-symbol and archive-hash gates [8, 9]; and
4. the validation and artifact-integrity system around those components,
   including a complete independent evaluator run, predecessor cross-GPU
   validations, and frozen equivalence/archive-provenance gates [9, 10].

“Original within this audited lineage” means the mechanism was not found in
the cited public challenge predecessors. It is deliberately narrower than a
claim of worldwide priority.

## Lineage map

```mermaid
flowchart LR
    HPAC["HPAC paper<br/>Li et al.<br/>2025-11-14"]
    QPOSE["qpose PRs #67 and #79<br/>mask and pose generation<br/>low-rank DCT actuator<br/>2026-05-03"]
    ANR["jas0xf PR #86<br/>semantic-token renderer<br/>patch-group HPAC<br/>master/slave split<br/>2026-05-04"]
    CPR1["PR #130<br/>semantic-pose-HPAC_CPR1<br/>integer HPAC<br/>learned gray carrier<br/>exact carrier repack"]

    HPAC -->|"HPAC architecture"| ANR
    ANR -->|"direct repository and architectural ancestor"| CPR1
    QPOSE -.->|"prior art limiting broad low-rank pose claims"| CPR1
```

Solid arrows denote verified architectural or repository ancestry. The dashed
arrow denotes novelty-limiting prior art; it does not assert that qpose code
was copied into this implementation.

## Chronology and immutable anchors

| Date | Artifact | Verified significance |
| --- | --- | --- |
| 2025-11-14 | HPAC paper [1] | Introduced the Hierarchical Parallel Autoregressive ConvNet and hierarchical factorization for practical lossless autoregressive coding. |
| 2026-05-03 | qpose PRs [#67][pr67] and [#79][pr79] | Public mask/pose-conditioned frame generation and low-rank DCT actuation [5-7]. |
| 2026-05-04 | jas0xf PR [#86][pr86], merge `14bcede815306415a0005c3cd98804151bce4049` | Public challenge adaptation combining a semantic-token renderer, NeRV slave, HPAC token model, and arithmetic decoding [2-4]. |
| 2026-07-19 | PR [#130][pr130] and release [`semantic-pose-HPAC_CPR1`][release] | Integer-lattice HPAC, learned gray pose carrier, exact carrier compression, and frozen provenance [8-10]. |

Repository ancestry is mechanically verifiable:

```text
git merge-base semantic-pose-HPAC_CPR1 \
  14bcede815306415a0005c3cd98804151bce4049

# 14bcede815306415a0005c3cd98804151bce4049
```

This establishes chronology and repository ancestry; it does not by itself
measure scientific novelty.

## Mechanism-by-mechanism assessment

| Mechanism | Verified predecessor | This implementation | Classification |
| --- | --- | --- | --- |
| Semantic class tokens rendered into RGB | PR #86 `TokenRendererV62`; write-up page 10 [3, 4] | Coordinate-aware token embeddings, four dilated residual blocks, frame embeddings, and RGB head [8] | **Inherited concept; new implementation** |
| Two frames with different evaluator roles | PR #86 assigns semantic work to the master and relative-pose work to the slave [3] | Semantic master plus independent gray pose carrier [8] | **Inherited factorization; materially changed slave realization** |
| Patch-group HPAC schedule | HPAC paper and PR #86 [1, 3, 4] | Same patch/group causal factorization [8] | **Inherited** |
| Type-A 7x7 followed by depthwise Type-B 5x5/dilation-2 and 3x3/dilation-4 layers | PR #86 `HPACMini` [4] | Same topology using integer convolution operators [8] | **Direct architectural continuation** |
| Frame conditioning | PR #86 FiLM/frame embedding [3, 4] | Integer frame shift and optional scale [8] | **Inherited mechanism; adapted arithmetic** |
| Previous-frame token context | PR #86 `conv_past` temporal branch [3, 4] | Integer `conv_past` [8] | **Inherited mechanism; adapted arithmetic** |
| Arithmetic/range coding of semantic tokens | PR #86 [2-4] | Canonical integer-logit entropy replay and exact token hashes [8, 9] | **Inherited coding principle; stronger portability contract** |
| Low-rank spatial pose actuation | qpose PRs #67/#79 and merged DCT actuator [5-7] | Learned quantized basis and per-pair coefficients [8] | **Prior art exists; basis/carrier design differs** |
| Standalone neutral-gray pose carrier | No equivalent found in the audited predecessors | `127.5 + amplitude * einsum(coeff, normalized_basis)`, independent of the semantic renderer [8] | **Original within this audited lineage** |
| Integer-lattice HPAC inference | PR #86 uses the same backbone, but not this bounded integer execution path [4] | Integer convolutions/linears, dyadic requantization, bounded activations, and a 1/8-logit lattice [8] | **Original extension within this audited lineage** |
| Canonical Huffman/Rice carrier repack | No equivalent found for this carrier in the audited predecessors | Exact basis/coeff repack, deterministic ZIP, decoded-state equality, and malformed-stream rejection [8, 9] | **Original artifact/codec engineering** |

## Direct architectural comparison

### Inherited HPAC skeleton

PR #86's `HPACMini` uses:

```text
frame embedding -> FiLM
Type-A masked 7x7 convolution
depthwise Type-B 5x5 convolution, dilation 2
depthwise Type-B 3x3 convolution, dilation 4
previous-token-frame 3x3 convolution
optional SPM
1x1 categorical head
```

`IntegerHPAC` retains the same ordered components and mask geometry [4, 8].
This correspondence is too specific to characterize as an independent
invention. The change is the numerical contract: layers become bounded integer
operators, intermediate values are explicitly requantized, and final logits
are canonicalized on a 1/8 lattice.

This exactness claim is limited to entropy inference and decoded semantic
symbols. The final RGB renderer still uses GPU floating-point kernels, so the
submission does **not** claim bit-identical raw RGB output across GPU
architectures [9].

### Renderer change

PR #86's master is a three-layer 3x3 semantic-token CNN with per-frame FiLM,
chosen to match SegNet's seven-pixel receptive field [3]. This renderer instead
embeds tokens, adds coordinate channels, and uses four residual blocks with
dilations `(1, 1, 2, 4)` before its RGB head [8].

That is a meaningful architecture change inside the inherited
semantic-token-to-RGB design, not the invention of semantic rendering.

### Pose-carrier change and qpose boundary

PR #86 uses a learned NeRV-style slave renderer driven by per-frame latents
[3]. This submission independently generates its slave frame as:

```text
carrier = einsum(per_pair_coefficients, learned_basis)
slave   = round(clamp(127.5 + amplitude * normalized(carrier)))
```

The implementation stores a quantized learned basis and 7,200 coefficients,
then expands the result from a neutral-gray center [8, 9]. This removes the
full NeRV slave network and makes the first frame a task-specific pose signal
rather than a conventional reconstruction.

qpose nevertheless predates this work's broad low-rank spatial-actuation idea
[5-7]. Therefore:

- **Defensible:** the standalone learned neutral-gray carrier, its separation
  from the semantic generator, and its exact packed representation.
- **Not defensible:** claiming to have invented low-rank pose actuation or
  evaluator-specific frame factorization in general.

## Achieved contribution

The canonical archive is 191,052 bytes with SHA-256
`0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`.
An independent RTX 2000 Ada run completed all 600 samples using the official
evaluator recipe with:

- PoseNet distortion: `0.00001967`;
- SegNet distortion: `0.00028609`;
- compression rate: `0.00508855`; and
- displayed score: `0.17`.

The final `linux-nvidia-t4` workflow remains the challenge's
hardware-specific gate [9, 10].

The CPR1 lossless repack reduced the frozen 194,380-byte predecessor by 3,328
bytes while requiring exact equality of the decoded semantic state, pose
basis, pose coefficients, HPAC model, and token stream [8, 9].

## Recommended public claim

> We extend jas0xf's public semantic-token/HPAC submission and the underlying
> HPAC architecture with an exact integer-lattice entropy path, a standalone
> quantized learned gray pose carrier, a compact dilated renderer,
> self-compressed model state, and cross-hardware symbol-level verification.
> We do not claim the semantic-token renderer, patch-group HPAC factorization,
> frame conditioning, previous-token context, masked-convolution backbone, or
> arithmetic coding as original. Low-rank pose actuation also has qpose prior
> art; our contribution is the carrier's standalone neutral-gray realization
> and exact packed deployment.

## Claim boundary

Supported claims:

- “We made the inherited HPAC entropy path deterministic at the integer
  symbol/logit level.”
- “We replaced the NeRV slave with a standalone learned gray pose carrier.”
- “We exactly repacked that carrier and proved decoded-state equality.”
- “We built a different semantic renderer within the inherited
  semantic-token rendering framework.”
- “We verified a complete evaluator run on independent hardware and preserved
  its hashes and provenance.”

Claims not made:

- “We invented semantic-token HPAC.”
- “We introduced patch-group causal entropy modeling.”
- “We introduced frame conditioning, previous-token context, masked HPAC
  convolutions, or arithmetic-coded HPAC tokens.”
- “We invented asymmetric master/slave evaluator factorization.”
- “We invented low-rank pose actuation.”
- “The whole decoder is bit-identical across GPUs.”

## References

1. Daxin Li, Yuanchao Bai, Kai Wang, Wenbo Zhao, Junjun Jiang, and Xianming
   Liu, “Rethinking Autoregressive Models for Lossless Image Compression via
   Hierarchical Parallelism and Progressive Adaptation,” arXiv:2511.10991,
   submitted 2025-11-14. <https://arxiv.org/abs/2511.10991>
2. jas0xf, “jas0xf_adversarial_neural_representation (0.27),” comma video
   compression challenge PR #86, merged 2026-05-04.
   <https://github.com/commaai/comma_video_compression_challenge/pull/86>
3. jas0xf, “Adversarial Neural Representation,” 24-page technical write-up,
   especially pages 10 and 20-21.
   <https://github.com/jas0xf/comma-anr-supplementary/blob/master/writeup.pdf>
4. Immutable PR #86 renderer, HPAC, and arithmetic-decoder source at merge
   commit `14bcede815306415a0005c3cd98804151bce4049`:
   - <https://github.com/commaai/comma_video_compression_challenge/blob/14bcede815306415a0005c3cd98804151bce4049/submissions/jas0xf_adversarial_neural_representation/inflate.py#L36-L121>
   - <https://github.com/commaai/comma_video_compression_challenge/blob/14bcede815306415a0005c3cd98804151bce4049/submissions/jas0xf_adversarial_neural_representation/inflate.py#L196-L268>
   - <https://github.com/commaai/comma_video_compression_challenge/blob/14bcede815306415a0005c3cd98804151bce4049/submissions/jas0xf_adversarial_neural_representation/inflate.py#L296-L354>
5. EthanYangTW, qpose challenge submission PR #67, merged 2026-05-03.
   <https://github.com/commaai/comma_video_compression_challenge/pull/67>
6. EthanYangTW, “qpose14_r55_segactions_minp v2 (0.31),” challenge PR #79,
   merged 2026-05-03.
   <https://github.com/commaai/comma_video_compression_challenge/pull/79>
7. Immutable qpose v2 source at merge commit
   `c74bd51046481997e4f123e3c24b14f906cac547`:
   - <https://github.com/commaai/comma_video_compression_challenge/blob/c74bd51046481997e4f123e3c24b14f906cac547/submissions/qpose14_r55_segactions_minp/inflate.py#L630-L672>
   - <https://github.com/commaai/comma_video_compression_challenge/blob/c74bd51046481997e4f123e3c24b14f906cac547/submissions/qpose14_r55_segactions_minp/inflate.py#L986-L992>
8. `semantic-pose-HPAC_CPR1` implementation at the release tag:
   - <https://github.com/fesalfayed/comma_video_compression_challenge/blob/semantic-pose-HPAC_CPR1/submissions/semantic-pose-HPAC_CPR1/hpac_integer.py#L165-L239>
   - <https://github.com/fesalfayed/comma_video_compression_challenge/blob/semantic-pose-HPAC_CPR1/submissions/semantic-pose-HPAC_CPR1/inflate.py#L119-L168>
   - <https://github.com/fesalfayed/comma_video_compression_challenge/blob/semantic-pose-HPAC_CPR1/submissions/semantic-pose-HPAC_CPR1/inflate.py#L611-L654>
   - <https://github.com/fesalfayed/comma_video_compression_challenge/blob/semantic-pose-HPAC_CPR1/submissions/semantic-pose-HPAC_CPR1/carrier_codec.py#L54-L140>
   - <https://github.com/fesalfayed/comma_video_compression_challenge/blob/semantic-pose-HPAC_CPR1/submissions/semantic-pose-HPAC_CPR1/repack_carrier.py#L212-L331>
9. Submission portability, equivalence, and artifact records:
   - [`README.md`](README.md)
   - [`verification.json`](verification.json)
   - [`MANIFEST.sha256`](MANIFEST.sha256)
10. PR #130 and canonical release:
    - <https://github.com/commaai/comma_video_compression_challenge/pull/130>
    - <https://github.com/fesalfayed/comma_video_compression_challenge/releases/tag/semantic-pose-HPAC_CPR1>

[pr67]: https://github.com/commaai/comma_video_compression_challenge/pull/67
[pr79]: https://github.com/commaai/comma_video_compression_challenge/pull/79
[pr86]: https://github.com/commaai/comma_video_compression_challenge/pull/86
[pr130]: https://github.com/commaai/comma_video_compression_challenge/pull/130
[release]: https://github.com/fesalfayed/comma_video_compression_challenge/releases/tag/semantic-pose-HPAC_CPR1
