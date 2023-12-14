import os
import torch
import argparse
import numpy as np
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat

from utils.data_loader import chooose_train_and_test_point, mirror_hsi, train_and_test_data, train_and_test_label
from utils.train_utils import test_epoch, output_metric
from vit_models import FactoFormer

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'WHU-Hi-HC', 'WHU-Hi-HH', 'WHU-Hi-LK'], default='Indian', help='dataset to use')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--input_size', type=int, default=7, help='Input sample size')
parser.add_argument('--model_path', type=str, default='./pretrained_models/WHU-Hi-LongKou/finetuned/WHU-Hi-LK_ft.pt')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

# prepare data
if args.dataset == 'Indian':
    print("Loading Indian Pines Dataset")
    data = loadmat('./data/IndianPine.mat')
elif args.dataset == 'Pavia':
    print("Loading University of Pavia Dataset")
    data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Houston':
    print("Loading Houston Dataset")
    data = loadmat('./data/Houston.mat')
elif args.dataset == 'WHU-Hi-HC':
    data = loadmat('./data/WHU-Hi-HC/WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
    TR = loadmat('./data/WHU-Hi-HC/Train100.mat')['Train100']
    TE = loadmat('./data/WHU-Hi-HC/Test100.mat')['Test100']
elif args.dataset == 'WHU-Hi-HH':
    data = loadmat('./data/WHU-Hi-HH/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
    TR = loadmat('./data/WHU-Hi-HH/Train100.mat')['HHCYtrain100']
    TE = loadmat('./data/WHU-Hi-HH/Test100.mat')['HHCYtest100']
elif args.dataset == 'WHU-Hi-LK':
    data = loadmat('./data/WHU-Hi-LK/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
    TR = loadmat('./data/WHU-Hi-LK/Train100.mat')['LKtrain100']
    TE = loadmat('./data/WHU-Hi-LK/Test100.mat')['LKtest100']
else:
    raise ValueError("Unkknow dataset")


if args.dataset == 'WHU-Hi-LK' or args.dataset == 'WHU-Hi-HC' or args.dataset == 'WHU-Hi-HH':
    label = TR + TE
    input = data
else:
    TR = data['TR']
    TE = data['TE']
    input = data['input']
    label = TR + TE


num_classes = np.max(TR)

# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
# data size

height, width, bands = input.shape
#-------------------------------------------------------------------------------
# obtain train and test data
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, bands, input_normalize, patch=args.input_size)
x_train, x_test, x_true = train_and_test_data(mirror_image, bands, total_pos_train, total_pos_test, total_pos_true, patch=args.input_size)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)

#-------------------------------------------------------------------------------
# load data
x_test = torch.from_numpy(x_test.transpose(0,3,1,2)).type(torch.FloatTensor) # [9671, 200, 7, 7]
y_test = torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
Label_test = Data.TensorDataset(x_test,y_test)

label_test_loader = Data.DataLoader(Label_test, batch_size=200, shuffle=False)

#-------------------------------------------------------------------------------
# create model
model = FactoFormer(
    img_size=[bands, args.input_size, args.input_size],
    spatial_patch=[bands, 1, 1],
    spectral_patch=[1, args.input_size, args.input_size],
    spatial_embed_dim=64,
    spectral_embed_dim=32,
    bands=bands,
    n_classes=num_classes,
    depth=5,
    n_heads=4,
    qkv_bias=True,
    attn_p=0.1,
    proj_p=0.1
)


model = model.cuda()

# print(model)

checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
tar_v, pre_v = test_epoch(model, label_test_loader)
OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

print("**************************************************")
print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print(AA2)
print("**************************************************")

