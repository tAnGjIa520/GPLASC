import os
import copy
import sys
import argparse
import shutil
import time
import math
import random
import numpy as np
import torch.nn.functional as F
import torch


import matplotlib.pyplot as plt

from tangjia.tsne import draw_simple_tsne_


features=torch.load("etf_feature.pth")
plt.figure()
import  numpy as np
# plt.xlim((-1.2, 1.2))
# plt.ylim((-1.2, 1.2))
label_list=[]
features_list=[]
for feature_,class_ids,taslID in features:
    # normal_feature=feature_ #F.normalize(feature_, dim=1)

    normal_feature=feature_
    label_list.append(taslID*torch.ones_like(class_ids))
    features_list.append(normal_feature)


a=1
# 画奇异值
# for index,feature in enumerate(features_list):
#
#     U, S, V=torch.linalg.svd(feature)
#
#     plt.plot(np.arange(1, S.size()[0]+1, 1),S.detach().numpy(), 's-', color='r', label="ATT-RLSTM")
#     plt.show()
#
#     a=1
#     break

# figure 1
draw_simple_tsne_(torch.concat(features_list,dim=0).detach().numpy(),torch.concat(label_list).detach().numpy())




# plt.scatter(normal_feature.detach().numpy()[:,0], normal_feature.detach().numpy()[:,1],
#             marker='o',
#             c=class_ids,
#             cmap='viridis'
#             )
#
# plt.show()
