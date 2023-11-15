import os
import clip
import torch
from torchvision.datasets import CIFAR10

# Load the model

device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load('RN50', device)

# check_point_path = "/remote-home/songtianwei/research/unlearn_multimodal/output/cifar10-Pretrain/checkpoint_epoch_64.pth"
# checkpoint = torch.load(check_point_path, map_location='cpu') 
# model.load_state_dict(checkpoint['model'])

# Download the dataset
cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar10[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar10.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(10)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar10.classes[index]:>16s}: {100 * value.item():.2f}%")