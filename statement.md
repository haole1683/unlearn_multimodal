
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

# correct result in CLIP ViT/16 in cifar-10

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


# correct result in CLIP ViT/16 in cifar-100
## advCLIP
### zero-shot
Top-1 accuracy: 0.22999999999999998
Top-5 accuracy: 1.7500000000000002
### linear-probe
Accuary 0.14
### linear-probe-unlearn
Accuary 0.64
# idea
- 1.将advCLIP的patch或者my noise attach到下游任务训练集上，在下游任务训练集上训练一下，看下效果。