# Distributed-Deep-Learning-for-Kidney-Tumor-Segmentation
Implementing and experimenting with the effects of parallel processing over a single node, multi-GPUs, and distributed
learning across multiple nodes on a cluster.

Achieving a 3.15 times speedup through distributed learning over 4 nodes, with a single GPU on each, where the batch
size is large enough

![Impact of GPU Parallelization Across Different Batch Sizes](https://github.com/Hosein-Beheshti/Distributed-Deep-Learning-for-Kidney-Tumor-Segmentation/blob/main/Impact%20of%20GPU%20Parallelization%20Across%20Different%20Batch%20Sizes.png)
Figure 4: Time comparison for running the training process in parallel across different batch sizes on a small proportion of the dataset

![Time Comparison Table](https://github.com/Hosein-Beheshti/Distributed-Deep-Learning-for-Kidney-Tumor-Segmentation/blob/main/Time%20Comparison.png)
Table 1: Time comparison of training model for 50 epochs using three approach of single GPU, data parallel, and distributed learning
