
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

# correct result
## clean
### zero-shot
Top-1 accuracy: tensor([108.5625], device='cuda:0')                                                                                     
Top-5 accuracy: tensor([145.1719], device='cuda:0')
Zero shot result: top1: tensor([108.5625], device='cuda:0'), top5: tensor([145.1719], device='cuda:0')
### linear-probe
87.42
### linear-probe-unlearn
87.39
## advCLIP
### zero-shot
Top-1 accuracy: tensor([1.5469], device='cuda:0')                                                                                       
Top-5 accuracy: tensor([28.1875], device='cuda:0')                                                                                      
Zero shot result: top1: tensor([1.5469], device='cuda:0'), top5: tensor([28.1875], device='cuda:0')  
### linear-probe
Accuary 0.31
### linear-probe-unlearn
Accuary 8.63
## my
### zero-shot
Top-1 accuracy: tensor([22.7188], device='cuda:0')                                                                                      
Top-5 accuracy: tensor([85.2656], device='cuda:0')                                                                                      
Zero shot result: top1: tensor([22.7188], device='cuda:0'), top5: tensor([85.2656], device='cuda:0')  
### linear-probe
Accuary 11.48
### linear-probe-unlearn
Accuracy = 39.920

# idea
- 1.将advCLIP的patch或者my noise attach到下游任务训练集上，在下游任务训练集上训练一下，看下效果。