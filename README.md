Fetal Planes Classification (EfficientNetV2-S | PyTorch)

This project classifies fetal ultrasound images into 6 anatomical planes using EfficientNetV2-S with transfer learning in PyTorch.
The model is trained on the Fetal Planes DB dataset and achieves 90%+ accuracy on validation and test sets.

🚀 Features

EfficientNetV2-S (pretrained on ImageNet)

Custom classifier for 6 classes

Medical-specific data augmentations

Automatic best-model saving

Clean training & evaluation pipeline

📂 Dataset

Kaggle: Fetal Planes DB
Structure:

train/
val/
test/

🧠 Training Pipeline

ImageFolder + transforms

EfficientNetV2-S with replaced final layer

Loss: CrossEntropy

Optimizer: AdamW

Saves the best model as: best_efficientnet_v2_s_fetal_planes.pth

📊 Results

Validation Accuracy: 90%+

Test Accuracy: 90%+

▶️ Run
python train.py
python test.py
