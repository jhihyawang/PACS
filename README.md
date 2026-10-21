# Four-View Mammography BI-RADS Triage

This repository classifies each four-view mammography examination (L-CC,
L-MLO, R-CC, and R-MLO) into BI-RADS 0, BI-RADS 1-3, or BI-RADS 4-5. Only an
examination-level label is available, so the model must combine four images
without image-level or breast-level supervision.

## Dataset

| Split | BI-RADS 0 | BI-RADS 1-3 | BI-RADS 4-5 | Total |
|---|---:|---:|---:|---:|
| Train | 833 | 1,045 | 210 | 2,088 |
| Validation | 119 | 149 | 30 | 298 |
| Test | 238 | 299 | 60 | 597 |

The 2,983 examinations are split by patient and stratified by target group.
Split CSV files are generated locally and are not distributed with this repository.

Expected dataset layout:

```text
data/raw/datasets/
├── Category 0/<patient_id>/{L-CC,R-CC,L-MLO,R-MLO}.jpg
├── Category 1/<patient_id>/{L-CC,R-CC,L-MLO,R-MLO}.jpg
├── ...
└── Category 5/<patient_id>/{L-CC,R-CC,L-MLO,R-MLO}.jpg
```

Generate the stratified 70/10/20 split and five development folds before
training:

```bash
python prepare_data.py \
  --data-root data/raw/datasets \
  --output-dir csv \
  --seed 42
```

This creates:

```text
csv/three_classes/{train_labels,val_labels,test_labels}.csv
csv/e17_folds/fold_{0..4}/{train_labels,val_labels}.csv
csv/split_manifest.json
```

The original project did not retain the code that produced its historical
patient assignments. This script reproduces the same split sizes and class
counts deterministically, but the generated patient membership is a new split;
historical test metrics should therefore be treated as reference results rather
than exact reproduction targets.

## Common Setup

- ImageNet-pretrained ConvNeXt-Tiny backbone
- Grayscale images replicated into three channels
- Right-breast horizontal flip for orientation normalization
- Foreground crop: intensity > 5, with a 5% width/height margin
- Bicubic resize to 1024 x 512
- Mild affine and brightness/contrast training augmentation
- Weighted cross-entropy and validation Macro-F1 early stopping
- Fixed folds with seeds 42-46

## Experiment Configurations

| Configuration | Architecture | Auxiliary asymmetry BCE |
|---|---|---:|
| `baseline` | CNN baseline | No |
| `cross_view` | CNN baseline + cross-view attention | No |
| `baseline_aux` | CNN baseline | Yes |
| `cross_view_aux` | CNN baseline + cross-view attention | Yes |
| `conditional` | Conditional classification with view/side attention | No |
| `conditional_aux` | Conditional classification with view/side attention | Yes |

The auxiliary branch predicts `label == BI-RADS 0` from the absolute left/right
feature difference. It adds a BCE term during training but does not adjust any
class score during inference.

## Architectures

**CNN baseline:** The shared backbone extracts one feature vector from each view.
CC and MLO features from the same breast are concatenated and passed through a
shared 2FC classifier. The left and right probabilities are averaged.

**Cross-view attention:** Bidirectional attention allows each view feature map to
retrieve complementary information from the other views. Residual fusion is
followed by the same pooling and classification procedure as the baseline.

**Conditional classification:** Each view predicts an Action logit (BI-RADS 1-3
vs. BI-RADS 0/4-5) and a Type logit (BI-RADS 0 vs. BI-RADS 4-5). View attention
combines CC and MLO, and side attention combines the left and right breasts:

```text
P(0)   = p_action * (1 - p_type)
P(1-3) = 1 - p_action
P(4-5) = p_action * p_type
```

For configurations with the `_aux` suffix, the training objective is:

```text
L_total = L_WCE + 0.3 * L_asymmetry_BCE
```

## Results

| Architecture | Aux. asymmetry | Accuracy | Macro-F1 | Recall 0 | Recall 1-3 | Recall 4-5 |
|---|---:|---:|---:|---:|---:|---:|
| CNN baseline | No | 0.5511 | 0.5132 | 0.3571 | 0.7258 | 0.4500 |
| CNN baseline | Yes | 0.5662 | 0.5435 | 0.3866 | 0.7224 | 0.5000 |
| Cross-view attention | No | 0.5327 | 0.5084 | 0.4412 | 0.6254 | 0.4333 |
| Cross-view attention | Yes | 0.5544 | 0.4978 | 0.2479 | 0.8261 | 0.4167 |
| Conditional classification | No | 0.5343 | 0.5113 | 0.4370 | 0.6321 | 0.4333 |
| Conditional classification | Yes | **0.5712** | **0.5664** | **0.4832** | 0.6421 | **0.5667** |

## Run

```bash
conda env create -f environment.yml
conda activate birads

python prepare_data.py --data-root data/raw/datasets --output-dir csv --seed 42

./train_baseline_cross_view.sh
./train_baseline_cross_view_aux.sh
./train_conditional.sh
./train_conditional_aux.sh
```

`RUN_ID`, `FOLDS`, and `PYTHON_BIN` can be overridden.

Checkpoints are not distributed with this repository. Run the corresponding
training script first, then use its printed `RUN_ID` with the unified evaluator:

```bash
python evaluate.py oof --experiment conditional_aux --run-id <RUN_ID> \
  --output-dir experiments/conditional_aux_oof

python evaluate.py test --experiment conditional_aux --run-id <RUN_ID> \
  --output-dir experiments/conditional_aux_test
```

Available configurations are `baseline`, `cross_view`, `baseline_aux`,
`cross_view_aux`, `conditional`, and `conditional_aux`.

The evaluator writes probabilities, metrics, bootstrap confidence intervals,
classification reports, and confusion matrices to the selected output directory.
