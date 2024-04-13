# from downstream_train_solo import classify
# from utils.clip_util import get_clip_model
# import time 

# t = time.time()

# start_time = time.time()

# clip = get_clip_model('ViT-B/16','cuda:0')

# import torch

# from utils.data_utils import load_poison_dataset, load_class_dataset

# noise = torch.load("/remote-home/songtianwei/research/unlearn_multimodal/output/train_g_unlearn/cat_noise.pt")

# from torchvision import transforms

# myTrans = transforms.Compose([
#     transforms.ToTensor()
#     ])

# ue_train_cifar10, cifar_test = load_poison_dataset("cifar10", noise, myTrans)

# clean_train_cifar10, cifar_test = load_class_dataset("CIFAR10", myTrans)

# classify(clip, ue_train_cifar10, cifar_test)

# end_time = time.time()

# print(f'coast:{end_time - start_time:.4f}s')



from utils.data_utils import ImageTextDatasetFromSupervisedDataset

train_ds = ImageTextDatasetFromSupervisedDataset("CIFAR10", 'train')
test_ds = ImageTextDatasetFromSupervisedDataset("CIFAR10", 'test')

print(train_ds[0])
print(test_ds[0])