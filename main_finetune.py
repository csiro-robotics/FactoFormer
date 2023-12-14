import os
import torch
import argparse
import time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat

from utils.train_utils import train_epoch, valid_epoch, output_metric
from utils.data_loader import chooose_train_and_test_point, mirror_hsi, train_and_test_data, train_and_test_label
from vit_models import FactoFormer
from utils.misc import load_state_dict, load_pretrained


parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'WHU-Hi-HC', 'WHU-Hi-HH', 'WHU-Hi-LK'], default='Indian', help='dataset to use')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')
parser.add_argument('--input_size', type=int, default=7, help='input sample size')
parser.add_argument('--epochs', type=int, default=80, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
parser.add_argument('--pretrained_spectral', type=str, default='./output/final/Houston/pretrained/pretrained_spectral.pth')
parser.add_argument('--pretrained_spatial', type=str, default='./output/final/Houston/pretrained/pretrained_spatial.pth')
parser.add_argument('--output_dir', type=str, default='./output/test/')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))

print_args(vars(args))

# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data
if args.dataset == 'Indian':
    data = loadmat('./data/IndianPine.mat')
elif args.dataset == 'Pavia':
    data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Houston':
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

height, width, bands = input.shape
print("height={0},width={1},bands={2}".format(height, width, bands))

total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, bands, input_normalize, patch=args.input_size)
x_train, x_test, x_true = train_and_test_data(mirror_image, bands, total_pos_train, total_pos_test, total_pos_true, patch=args.input_size)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)

x_train = torch.from_numpy(x_train.transpose(0,3,1,2)).type(torch.FloatTensor) #[695, 200, 7, 7]
y_train = torch.from_numpy(y_train).type(torch.LongTensor) #[695]
Label_train = Data.TensorDataset(x_train,y_train)

x_test = torch.from_numpy(x_test.transpose(0,3,1,2)).type(torch.FloatTensor) # [9671, 200, 7, 7]
y_test = torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
Label_test = Data.TensorDataset(x_test,y_test)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)

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

checkpoint_spectral = load_pretrained(args.pretrained_spectral, model, 'spectral')
checkpoint_spatial = load_pretrained(args.pretrained_spatial, model, 'spatial')

load_state_dict(model, checkpoint_spectral, prefix='')
load_state_dict(model, checkpoint_spatial, prefix='')

print(model)

# criterion
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//10, gamma=args.gamma)

print("start fine-tuning")
tic = time.time()
for epoch in range(args.epochs):
    tic = time.time()
    # train model
    model.train()
    train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
    OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
    print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}".format(epoch+1, train_obj, train_acc))
    scheduler.step()

    if (epoch % args.test_freq == 0) | (epoch == args.epochs - 1):
        model.eval()
        tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)


torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, args.output_dir + '_model_test.pt')

toc = time.time()
print("Running Time: {:.2f}".format(toc-tic))
print("**************************************************")

print("Final result:")
print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
print(AA2)
print("**************************************************")
print("Parameter:")
print("**************************************************")

print_args(vars(args))
