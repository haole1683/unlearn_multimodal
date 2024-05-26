import torch

from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

import clip
from tqdm import tqdm

# normalize from clip.py
def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


# [224,224]
clip_transform_224 = transforms.Compose([
    Resize((224,224),interpolation=BICUBIC),
    CenterCrop(224),
    _convert_image_to_rgb,
    ToTensor(),
])

clip_transform_256 = transforms.Compose([
    Resize((256,256),interpolation=BICUBIC),
    CenterCrop(256),
    _convert_image_to_rgb,
    ToTensor(),
])

clip_transform_288 = transforms.Compose([
    Resize((288,288),interpolation=BICUBIC),
    CenterCrop(288),
    _convert_image_to_rgb,
    ToTensor(),
])

clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

prompt_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

def zeroshot_classifier(clip_model, device, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = clip_model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def get_embedding(clip_model, device, text):
    with torch.no_grad():
        text = clip.tokenize(text).to(device) #tokenize
        text_embedding = clip_model.encode_text(text) #embed with text encoder
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.mean(dim=0)
        text_embedding /= text_embedding.norm()
    return text_embedding


def get_clip_model(clip_version, device):
    model, _ = clip.load(clip_version, device, jit=False)
    model = model.float()
    model = model.to(device)
    return model

import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIP(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.transformer
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(1024, 4).to(clip_model.dtype).to(clip_model.text_projection.device)
            
    def forward(self, image, text):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        ratio = 0.2
        image_features = ratio * x + (1 - ratio) * image_features

        # text_features = self.text_encoder(text)
        text_features = self.clip_model.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        logits_per_image = logits
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
