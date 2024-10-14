import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from inaturalist2018 import LT_Dataset_Test, iNaturalist2018  # Assuming harbarium2.py contains your dataset classes and configurations

# Define paths and configurations
checkpoint_path = './data/iNaturalist2018/resnet50_reslt_bt256/checkpoint.pth.tar'
root = "/mnt/3ff84d1d-7336-4128-810e-2e29f65bcbc5/visual_categorization/herbarium-2022-fgvc9/"
val_txt = "/mnt/3ff84d1d-7336-4128-810e-2e29f65bcbc5/visual_categorization/herbarium-2022-fgvc9/val_hbm.txt"

batch_size = 32
num_works = 12

# Define transforms for test data
normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Load model from checkpoint
checkpoint = torch.load(checkpoint_path)
model = checkpoint['model']
model.eval()

# Load dataset for testing
testset = LT_Dataset_Test(root, val_txt, transform=transform_test, class_map=checkpoint['class_map'])
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_works, pin_memory=True)

# Initialize lists to store predictions and true labels
predicted_labels = []
true_labels = []

# Run inference on test data
with torch.no_grad():
    for images, targets in test_loader:
        images = images.cuda()  # Assuming GPU availability
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predicted_labels.extend(preds.cpu().numpy())
        true_labels.extend(targets.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
confusion_matrix.py