# SPACE

Training code for SPACE. Each video is represented by sparse-autoencoder (SAE) features of eight frames taken from LLaVA-NeXT-LLaMA3-8B layer 24. SPACE learns a global feature gate, a linear evidence head and a policy projector on these features, then selects the deployed feature set on a Pareto front.

| Path | Contents |
|---|---|
| `config.sh` | Paths and interpreter (section 1) |
| `train/train_space.py` | Training, Pareto feature selection and test evaluation |
| `sbatch/train_space.sbatch` | Slurm job that runs `train_space.py` |
| `aggregate/agg_seeds.py` | For each seed, takes the fold with the best validation score and reports its test result |

---

## 1. `config.sh`

Scripts read their paths from environment variables. Set them, then source `config.sh`:

```bash
export SPACE_DATA=/path/to/data
export SPACE_PYTHON=/path/to/env/bin/python
source /path/to/package/config.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `SPACE_ROOT` | directory of `config.sh` | Package root |
| `SPACE_DATA` | `$SPACE_ROOT/data` | Inputs and all outputs (section 2) |
| `SPACE_PYTHON` | `python` | Interpreter (section 3) |
| `SEEDS` | `0,…,9` | Seed list for `train_space.sbatch`, comma-separated |

---

## 2. Inputs under `$SPACE_DATA`

| Path | Format |
|---|---|
| `merged_output/cell3_cache_v5.npz` | `Z_frames`: float32, N × 8 × 131,072, SAE features of the eight sampled frames of each video in temporal order (SAE `lmms-lab/llama3-llava-next-8b-hf-sae-131k` on layer 24 of `llava-hf/llama3-llava-next-8b-hf`). `y_class`: int, N, video label (0 Non-Toxic, 1 Eating Disorders, 2 Body Anxiety Comparison; videos labelled 3 or 4 are removed before training). `masks_mc`: int, N × 8, the video's label on frames annotated as harmful and 0 elsewhere. |
| `merged_output/cell3_meta_v5.json` | `{"filenames": [...]}`, one filename per row of the cache, in the same order. Filenames have the form `@<account>_video_<video_id>.mp4`; the account groups videos in cross-validation. |
| `merged_output/dataset_records_final.json` | List with one record per video, with fields `filename` and `video_id`. Maps cache rows to video ids. |
| `merged_output/policy_kb/policy_anchors_cat.pt` | float32 tensor, 5 × 768: L2-normalised CLIP text embeddings of the policy text of classes 0–4 (OpenCLIP `ViT-L-14`, pretrained weights `laion2b_s32b_b82k`). |
| `baseline_outputs/train_test_split.json` | `{"train_video_ids": [...], "test_video_ids": [...]}`. Videos in `test_video_ids` form the held-out test set; all other videos are used for 5-fold cross-validation. |

---

## 3. Environment

Python 3.10 with

```
torch==2.5.1 numpy==2.2.6 scikit-learn==1.7.2
```

Training needs one GPU. It loads the feature cache into host memory (about 16 GB as float32) and, by default, copies the features of the 3,711 retained videos to GPU memory in bf16 (about 7.8 GB). `--no-preload` keeps them in host memory and sends each batch in float32; `--no-half` stores them on the GPU in float32.

---

## 4. Running

From the package root, after `source config.sh`:

```bash
# 1. Training -> $SPACE_DATA/merged_output/main_softmax/
sbatch sbatch/train_space.sbatch            # or: $SPACE_PYTHON train/train_space.py

# 2. Result per seed (validation-best fold) and mean ± SD over seeds
$SPACE_PYTHON aggregate/agg_seeds.py --dirs $SPACE_DATA/merged_output/main_softmax --out agg_seeds.json
```

The `#SBATCH` header requests only a GPU, CPUs, memory and time; add your cluster's partition or account on the command line (`sbatch -p <partition> -A <account> ...`). The job stops immediately if CUDA is not available.

### Outputs of training

`$SPACE_DATA/merged_output/main_softmax<--out-suffix>/`:

- `retrain_results.json`: key `full`, one record per (fold, seed). Fields include `toxic_only` (test Macro F1 over classes 1 and 2 on harmful test videos), `false_alarm`, `n_benign`, `val_best` (validation score of the selected model), `pareto_C` (number of retained features), `pareto_front`, `top1_hit_toxic` (share of harmful test videos whose highest-evidence frame is annotated harmful) and `policy_anchor_doc_harmful` (share of harmful test videos whose projected representation is closest to the anchor of their own category).
- `proba_full_f<fold>_s<seed>.npz`: test-set class probabilities, `video_ids`, `y_true`.
- `params_full_f<fold>_s<seed>.npz`: gate values, selection mask, evidence-head weights and bias.

Running the same command again skips the (fold, seed) pairs already in `retrain_results.json`.

---

## 5. Training configuration

The defaults of `train_space.py` are the configuration used in the paper.

| Setting | Value |
|---|---|
| Data split | Videos with label 3 or 4 removed. 5-fold `StratifiedGroupKFold` over the training videos, grouped by account (`random_state=42`). |
| Classes | Harmful classes 1 and 2. The head has five outputs; outputs 3 and 4 are masked at prediction. |
| Input scaling | Features multiplied by 1 / mean(\|z\|) over the training videos. |
| Model | Gate logits initialised at 2.2; linear evidence head on gated frame features; video logits = maximum over the 8 frames; policy projector 131,072 → 1,024 → 768 with ReLU and L2 normalisation. |
| Class weights | Inverse class frequency on the training portion of the fold, normalised to mean 1. |
| Phase 1 | 10 epochs, gate frozen: video cross-entropy + 0.5 × frame cross-entropy. |
| Phase 2 | 12 epochs, gate frozen: adds within-video ranking (0.2) and cross-video contrast (0.05, temperature 0.2), both ramped linearly over the first 30% of the phase. |
| Phase 3 | Up to 150 epochs, gate trained: adds policy alignment (0.1, temperature 0.1) and the sparsity penalty; ranking, contrast and policy alignment are ramped again over the first 30% of the phase. The sparsity coefficient starts at 5 × 10⁻³ and is multiplied each epoch by (active gates / 512)^0.25. Validation every 5 epochs; training stops after 40 epochs without improvement once the number of active gates is within 256–1,024, and the best checkpoint is restored. |
| Optimiser | AdamW, batch 32, gradient clipping 1.0. Evidence head: learning rate 5 × 10⁻⁴, weight decay 10⁻². Gate: 0.1, no weight decay. Projector: 10⁻³, weight decay 10⁻². |
| Validation score | Macro F1 over classes 1 and 2 on the harmful validation videos, multiplied by 0.107 / FA when the validation false-alarm rate FA exceeds 0.107 (`--fa-cap`). Used for checkpoint selection, Pareto selection and `val_best`. |
| Pareto selection | Candidate feature sets S = {k : sigmoid(g_k) > δ} for δ ∈ {0.01, 0.05, 0.1, 0.2, …, 0.9, 0.95, 0.99}; Pareto front over (validation score, \|S\|); among front points within one standard error of the best (account-level bootstrap, 500 resamples), the one with the smallest \|S\|. |
| Seeds | The model of fold f under seed s is trained with random seed 100·s + f. |

---

## License

Released under the MIT License (see `LICENSE`).
