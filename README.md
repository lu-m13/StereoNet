My implementation of StereoNet (https://arxiv.org/abs/1807.08865).

### EPE (ALL) Results on SceneFlow (8x Multi)
| Paper      | My implementation |
| ---------- | ----------------- | 
| 1.101      | 1.323             |  


Different from the paper, I simply use Smooth L1 Loss and all the stages are equally weighted. 

Better results might be achieved by some parameter finetuning.

## Acknowledgments

https://github.com/meteorshowers/StereoNet-ActiveStereoNet
