# Official FlatLands implementation/weights check

Audit date: **2026-09-02 (America/New_York)**  
Decision: **reference-only for the current ConPath validation matrix**

## Sources checked

- [Official FlatLands project page](https://1ssb.github.io/Flat_Lands/)
- [Official FlatLands GitHub repository](https://github.com/1ssb/Flat_Lands/)
- [Official Hugging Face dataset card](https://huggingface.co/datasets/Rudra1ssb/FlatLands)
- [FlatLands paper](https://arxiv.org/abs/2603.16016)

## Findings

The official release provides the validated `FlatLands_final_dataset.zip`, dataset documentation,
provenance, licensing, and benchmark description. The official repository's release-status section
states that model weights, construction code, and additional benchmark tooling are planned for
release; its visible repository contents currently contain documentation and release metadata rather
than an importable model checkpoint or an executable official evaluator.

The project page describes eleven benchmark families, including diffusion and flow-matching methods,
but those published cross-task/model-family summaries do not provide a checkpoint that can be loaded
under our exact three-channel, 256×256, provenance-original scene-disjoint validation contract.

## Compatibility decision

No official FlatLands model score is copied into the ConPath same-dataset table. Importing a future
checkpoint will require a new audit of input preprocessing, scene split, hidden-cell mask, raster
resolution, query/event protocol, and license terms. Until those artifacts are public and pass those
checks, the official FlatLands implementation is tracked as **not yet importable / reference-only**.
Our existing PaSCo-inspired and S4C-inspired controls remain explicitly labelled adapters rather than
reproductions of their original systems.

This decision does not prevent a separately implemented 2-D diffusion-style control, but that control
must be reported as our own same-contract adapter and cannot inherit FlatLands' published leaderboard
numbers.
