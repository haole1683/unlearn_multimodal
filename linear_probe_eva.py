import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

# check_point_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/cifar10-Pretrain/checkpoint_epoch_64.pth"

# model, preprocess = clip.load(check_point_path, device)
model, preprocess = clip.load('RN50', device)
# make sure to convert the model parameters to fp32
model = model.float()
model = model.to(device) 
model = model.eval()

# checkpoint = torch.load(check_point_path, map_location='cpu') 
# model.load_state_dict(checkpoint['model'])



# Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR10(root, download=True, train=True, transform=preprocess)
test = CIFAR10(root, download=True, train=False, transform=preprocess)



def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=128)):
            features = model.encode_image(images.to(device))

            # judge the feature whether has nan
            if torch.isnan(features).any():
                for index in range(len(features)):
                    if torch.isnan(features[index]).any():
                        features = model.encode_image(images.to(device))
                        print("has nan")
                        img_tensor = images[index]
                        img_tensor = img_tensor.cpu().numpy()
                        img_tensor = np.transpose(img_tensor, (1, 2, 0))
                        img_tensor = img_tensor * 255
                        img_tensor = img_tensor.astype(np.uint8)
                        img = Image.fromarray(img_tensor)
                        img.save("has_nan.jpg")
                continue
            
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")