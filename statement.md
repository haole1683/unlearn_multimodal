
# error result
32 8
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:16<00:00,  2.11s/it]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:02<00:00,  3.56it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:05<00:00,  1.43it/s]
From advCLIP
Top1 test acc: 74.6845, Top1 solo_adv_test acc: 3.5757, Fooling rate: 96.1952
Clean downstream accuracy: 74.6845%
Adv downstream accuracy: 3.5757%
Decline accuracy rate: 95.2122%
Downstream fooling rate: 96.1952%

From my
Top1 test acc: 74.6845, Top1 solo_adv_test acc: 25.3230, Fooling rate: 73.0093
Clean downstream accuracy: 74.6845%
Adv downstream accuracy: 25.3230%
Decline accuracy rate: 66.0933%
Downstream fooling rate: 73.0093%

my linear probe:
10.63

advCLIP:
82.64

# result in CLIP ViT/16 in cifar-10

## clean
### zero-shot
Top-1 accuracy: 69.24
Top-5 accuracy: 92.46
### linear-probe
87.42
### linear-probe-unlearn
87.39

## advCLIP
### zero-shot
Top-1 accuracy: 0.9900000000000001
Top-5 accuracy: 17.98
### linear-probe
Accuary 0.280
### linear-probe-unlearn
Accuary 8.65

## my
### zero-shot
Top-1 accuracy: 14.52
Top-5 accuracy: 54.39000000000001
### linear-probe
Accuary 11.04
### linear-probe-unlearn
Accuracy = 39.39

## my 200 epoch
这里是max-loss
### zero-shot
Top-1 accuary: 9.21
Top-5 accuary: 51.29
### linear-probe
Accuary 10.58
### linear-probe-unlearn
Accuary 27.80

## my 200 epoch min loss
### zero-shot
Top-1 accuracy: 86.75
Top-5 accuracy: 99.16
### linear-probe
Accuary 82.48
### linear-probe-unlearn
Accuary 64.95


# result in CLIP ViT/16 in cifar-100
## advCLIP
### zero-shot
Top-1 accuracy: 0.22999999999999998
Top-5 accuracy: 1.7500000000000002
### linear-probe
Accuary 0.14
### linear-probe-unlearn
Accuary 0.64

# result in CLIP ViT/32 in cifar-100 
Note: 这里攻击的补丁以及noise都是在ViT/16上面训练得到的。
现象：当在不同模型之间迁移的时候，攻击效果失效了

## advCLIP
Top-1 accuracy: 64.74
Top-5 accuracy: 88.37
### linear-probe
63.30
### linear-probe-unlearn
35.920

## my
Top-1 accuracy: 61.12
Top-5 accuracy: 86.26
### linear-probe
54.640
### linear-probe-unlearn
30.06

# result in CLIP ViT/16 in cifar-10 My Targeted Attack
## My Text-guide targeted attack
Attackers' abilities : Add noise to the downstream test image, knowing the downstream classes 
(eg: cifar10-10 classes names:['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

Step 1:
Train a generator to generate noise
Step 2:
Construct the attack prompt , "A image of {}".format("dog"), Attacker would like to mislead model to classify all image to "dog"
Step 3:
Test Attack performance.

Here is the accuary rate after attack:
### zero-shot
Top-1 accuracy: 12.770000000000001
Top-5 accuracy: 86.72999999999999
### linear-probe
Linear probe result: 16.97

# idea
- 1.将advCLIP的patch或者my noise attach到下游任务训练集上，在下游任务训练集上训练一下，看下效果。 fail
- 2.advCLIP迁移性不行，必须要指定CLIP对应的模型类型，要不然白搭。
- 3.做防御？
- 4.单模态文本对抗攻击。
- 5.有目标攻击(文本指导的有目标攻击) Text-guide targeted attack，根据下游任务，可以指定文本输入内容，然后生成对应的patch或者noise，然后noise覆盖到所有图像。
    - 有目标攻击，已经知道下游数据集的一个类别，以及使用的白盒的模型，然后生成对应的patch或者noise，然后noise覆盖到所有图像上，完成有目标攻击。
- 6.优化损失函数，取代cos loss余弦相似度损失。
- 7.CLIP的对比损失只将自己作为正样本，其余全部作为负样本，但是实际上同一只猫的不同图片应该是正样本，不同猫的图片应该是负样本，这样的话，可以优化对比损失函数。或者是同一类别下的图片作为正样本，不同类别的图片作为负样本。也就是在原本的对比学习中能否额外加入有监督学习的监督信号？
- 8.更多模态的攻击后门（比如声音，视频，文本，图像，等等）

# Phenomenon
目前的方法都无法做到迁移性，CLIP的encoder有多个版本，但是针对一个模型的攻击（ViT-B16）无法迁移到
其他的模型（ViT-B32），一迁移攻击效果为0。