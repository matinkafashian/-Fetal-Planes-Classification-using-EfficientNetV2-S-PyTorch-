# Fetal ultrasound image classification

Notebook project using an ImageNet-pretrained EfficientNetV2-S model and PyTorch to classify six categories in the supplied Fetal Planes DB dataset. This is a dataset experiment, not a clinically validated diagnostic system.

## Repository contents

- `Fetal_Planes_DB.ipynb`: data inspection, training, validation and test evaluation.
- `LICENSE`: repository license.

There are no standalone `train.py` or `test.py` scripts. Run the notebook itself.

## Run the notebook

1. Import `Fetal_Planes_DB.ipynb` into a Kaggle notebook. A GPU is recommended; the code falls back to CPU.
2. Attach the Kaggle dataset `minhnhtl05/fetal-planes-db-dataset` and enable internet access for pretrained weights if they are not cached.
3. Verify the dataset is available at `/kaggle/input/fetal-planes-db-dataset/Fetal_Planes_DB`, with `train/`, `val/` and `test/` class folders. The notebook also contains a `kagglehub.dataset_download` cell, but subsequent cells use the explicit Kaggle path rather than the returned download path.
4. Outside Kaggle, update `dataset_path`, `base_path` and `base_dir` to your actual dataset location.
5. Run the cells in order. Dependencies used are Python, torch, torchvision, pandas, scikit-learn, matplotlib, Pillow and kagglehub. Exact versions from the original run were not recorded; matching them reproducibly remains future work.
6. The training cell runs 15 epochs with batch size 32, AdamW (learning rate 1e-4, weight decay 1e-2), and CrossEntropyLoss. It saves the best validation checkpoint as `best_efficientnet_v2_s_fetal_planes.pth`, reloads it and evaluates the supplied test split.

## Interpreting results

Results in the notebook are saved historical outputs, not a new reproduction of training. The resume reports 94.68% test accuracy on the supplied 3,725-image test split, with 7,437 training images. Inspect the notebook outputs alongside that claim.

The code uses grayscale-to-RGB conversion, image augmentation for training, and deterministic resize/center-crop transforms for validation and test. It does not establish patient independence across the supplied splits or clinical generalization. A fixed training seed, dependency lockfile, per-class metrics and patient-level split audit would improve reproducibility. No performance guarantee is made for new datasets.
