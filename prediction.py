# Pre-import and set for a clear ploting
import os
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from utils.datasets import mani_dataset # special Dataset class
import torch
import cv2
import utils.datasets
import utils.iml_transforms
import utils.misc as misc
import numpy as np
import iml_vit_model
import utils.evaluation as evaluation
from main_train import get_args_parser

def cal_precise_AUC_with_shape(predict, target, shape):
    predict2 = predict[0][0][:shape[0], :shape[1]]
    target2 = target[0][0][:shape[0], :shape[1]]
    # flat to single dimension fit the requirements of the sklearn
    predict3 = predict2.reshape(-1).cpu()
    target3 = target2.reshape(-1).cpu()
    # -----visualize roc curve-----
    # fpr, tpr, thresholds = roc_curve(target3, predict3, pos_label=1)
    # plt.plot(fpr, tpr)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.savefig("./appro2.png")
    # ------------------------------
    try:
        AUC = roc_auc_score(target3, predict3)
    except:
        AUC = 0
    return AUC


plt.rcParams['figure.dpi'] = 60
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (24, 6)

# here we use mani dataset as example
input_dir = "./dataset/casia/"

test_transform = utils.iml_transforms.get_albu_transforms('test')
dataset = mani_dataset(
    transform=test_transform,
    path = input_dir,
    edge_width=7, # specify the edge mask, other wise only return 2 objects
    if_return_shape=True
)
print(dataset)
print(f":ength of this dataset: {len(dataset)}")

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = get_args_parser()
args = args.parse_args()
ckpt_path = "./output_dir/checkpoint.pth"  # checkpoints/iml-vit_checkpoint
args = get_args_parser()
args = args.parse_args()
model = iml_vit_model.iml_vit_model(
    vit_pretrain_path=args.vit_pretrain_path,
    predict_head_norm=args.predict_head_norm,
    edge_lambda=args.edge_lambda
)
checkpoint = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model = model.to(device)
### Inference
# This process require your GPU has at least 6GB of memory.
results = []
local_f1s = 0
AUCs = 0
i = 0
model.eval()
with torch.no_grad():
    for img, gt, edge_mask, shape in dataset:  # Inference don't need edge mask.
        img, gt, edge_mask = img.to(device), gt.to(device), edge_mask.to(device)
        name = dataset.gt_path[i].split(input_dir+'Gt/')[-1]
        print(i)
        i += 1
        # Since no Dataloader, manually create a Batch with size==1
        img = img.unsqueeze(0)  # CHW -> 1CHW
        gt = gt.unsqueeze(0)
        edge_mask = edge_mask.unsqueeze(0)

        # inference
        predict_loss, mask_pred, edge_loss = model(img, gt, edge_mask)
        predict = mask_pred.detach()
        # ---- Training evaluation ----
        # region_mask is for cutting of the zero-padding area.
        region_mask = torch.zeros_like(gt)
        region_mask[:, :, :shape[0], :shape[1]] = 1
        TP, TN, FP, FN = evaluation.cal_confusion_matrix(predict, gt, region_mask)
        AUC = cal_precise_AUC_with_shape(predict, gt, shape.numpy())
        local_f1_single = evaluation.cal_F1(TP, TN, FP, FN)
        local_f1s += local_f1_single
        AUCs += AUC
        output = mask_pred
        # Cut the origin area from padded image
        output = output[0, :, 0:shape[0], 0:shape[1]].permute(1, 2, 0).cpu().numpy()
        gt = gt[0, :, 0:shape[0], 0:shape[1]].permute(1, 2, 0).cpu().numpy()
        # results.append(output)
        res = np.where(output > 0.5, 1, 0)
        plt.imshow(res, cmap="gray")
        plt.show()

print("Done!")
local_f1s /= len(dataset)
AUCs /= len(dataset)
print(f"F1:{local_f1s}, AUC: {AUCs}")
