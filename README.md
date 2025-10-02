# Label-guided-teacher-for-surgical-phase-recognition

Hi, this is the code repository for "Label-guided Teacher for Surgical Phase
Recognition via Knowledge Distillation" (Accepted by MICCAI2024) in PyTorch.

![avatar](https://github.com/guascy666/Label-guided-teacher-for-surgical-phase-recognition/blob/main/model.png)

In this paper, we propose a novel label-guided teacher network for knowledge
distillation. Specifically, our teacher network takes both video frames and
ground-truth labels as input. Instead of only using labels to supervise the
final predictions, we additionally introduce two types of label guidance
to learn a better teacher: 1) we propose label embedding-frame feature
cross-attention transformer blocks for feature enhancement; and 2) we
propose to use label information to sample positive (from same phase)
and negative features (from different phases) in a supervised contrastive
learning framework to learn better feature embeddings. Then, by minimizing feature similarity, the knowledge learnt by our teacher network
is effectively distilled into a student network. At inference stage, the dis
tilled student network can perform accurate surgical phase recognition
taking only video frames as input.

