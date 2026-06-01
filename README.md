# Temporal Factorization of Factual Recall in Language Models

## One-sentence summary

This project studies when relation information and entity information become *generation-controlling* during factual recall in decoder-only language models, and finds that relation information takes over at the final prediction position before entity information does.

## Main finding

Across four decoder-only models and eight prompt families, relation onset precedes entity onset by 10–16 tested layers (31–44% of network depth) at the final-token position. Entity information is *not* absent early rather it is already strongly active at the entity/subject-token position from early layers, but only becomes generation-controlling at the final token after being routed there. This pattern is referred to as **deferred entity commitment**.

## Models studied

* Llama-3.2-3B (28 layers)
* Llama-3-8B (32 layers)
* Qwen2.5-3B (36 layers)
* Phi-2 (32 layers)

## Prompt families

Eight controlled fill-in-the-blank families spanning factual, morphological, lexical, and symbolic transformations: *capital, language, past tense, present participle, plural, opposite, comparative, chemical symbol*.

## Experiments included

### Experiment 1: Transfer curves and onset
Final-token activation patching with two conditions: *relation transfer* (same entity, different relation) and *entity transfer* (same relation, different entity). Onset = first tested layer crossing 0.4 transfer for two consecutive tested layers. A *wrong-entity* control separates true relation-only transfer from donor-answer copying.

### Experiment 2: Both-change competition
Donor differs in *both* relation and entity, forcing direct competition. Outputs are classified as relation-wins, entity-wins, original-retained, or mixed. Includes noise, self-patch, alternate-donor, and unrelated-donor controls.

### Experiment 3: Entity-token vs final-token patching
The key control: patching the donor's entity-token hidden state into the recipient's entity-token position vs. patching at the final-token position. Tests whether entity information is *absent* early or just *not yet routed* to the final token.

### Experiment 4: Layer-zone steering
Mean-difference relation and entity steering vectors applied at mid- and late-layer zones, compared against matched-norm random-direction baselines.

## Key results

Pair-balanced onset at threshold 0.4 (relation onset always precedes entity onset):

| Model        | Rel. onset | Ent. onset | Gap | Depth |
| ------------ | ---------: | ---------: | --: | ----: |
| Llama-3.2-3B |         L6 |        L18 |  12 |   43% |
| Llama-3-8B   |        L10 |        L20 |  10 |   31% |
| Qwen2.5-3B   |        L16 |        L32 |  16 |   44% |
| Phi-2        |        L14 |        L24 |  10 |   31% |

The ordering holds in **16/16 model-threshold combinations** for thresholds 0.2–0.5.

In Experiment 3, entity-token patching transfers entity identity at **90–100%** in early layers while final-token patching is at **≈2.4%**, with the pattern cleanly reversing in late layers.

## Repository contents

```text
.
├── README.md
├── PROJECT_SUMMARY.pdf             2-page research summary
├── figures/
│   ├── fig2_transfer_curves.pdf
│   ├── fig3_both_change_competition.pdf
│   ├── fig4_subject_vs_last_patching.pdf
│   └── fig5_steering_asymmetry.pdf
├── results/                       Processed CSV summaries (used for figures)
│   ├── exp1_relation_entity_transfer/
│   ├── exp2_both_change/
│   ├── exp3_subject_token_patching/
│   └── exp4_steering/
└── scripts/
    └── make_figures.py      Regenerates all figures from CSVs (no GPU)
```

## Reproducing figures

The main figures can be regenerated from the included CSV summaries without loading any language models:

```bash
pip install pandas numpy matplotlib
python scripts/make_figures.py
```

Output is written to `figures/`.

## Artifact note

This is a compact research artifact prepared for application review. It contains the main figures, processed result summaries, and lightweight reproduction code. The full GPU experiment pipeline is not included here; the included CSVs are sufficient to verify every figure and table in the summary.
