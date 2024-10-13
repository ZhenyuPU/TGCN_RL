# TGCN_RL

This is a project of the RL application in hydrogen-based integrated energy system. A novel method to combine TGCN and RL is proposed to enhance the performance of RL. TGCN works as feature representations. It can effectively handle spatial and temporal information so as to leverage input features of RL. But it also needs to be improved. T6CN should be trained at first and then fine tuned in the training process of RL, which can avoid performance degradation. The training target of TGCN is in fact not the same as that of RL.


## Principle

The following picture shows the main principle of TGCN-RL.

![T-GCN_RL](https://cdn.jsdelivr.net/gh/ZhenyuPU/picx-images-hosting@master/20241013/T-GCN_RL.4ain4ncxl9e0.jpg)
