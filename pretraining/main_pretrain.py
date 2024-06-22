import torch
import argparse
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os
from utils import *
from dataloader_utils import mirror_hsi, train_and_test_data, choose_train_and_test_point
from train_utils import train_epoch
from scipy.io import loadmat

from spectral import *
from mim_hsi import VisionTransformerForMIM, HyperMIM

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'WHU-Hi-HC', 'WHU-Hi-HH', 'WHU-Hi-LK'], default='Indian', help='dataset to use')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--input_size', type=int, default=7, help='input sample size')
parser.add_argument('--epochs', type=int, default=200, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay')
parser.add_argument('--output_dir', default='./output/test/', help='path where to save checkpoints')
parser.add_argument('--save_ckpt_freq', default=50, type=int, help='Frequency to save a checkpoint of the model')
parser.add_argument('--mask_ratio', default=0.7, type=float, help='ratio of the visual tokens/patches need be masked')
parser.add_argument('--device', default="1", type=str)
parser.add_argument('--pretrain_mode', default="spectral")


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
print("Cuda availability : ", torch.cuda.current_device())
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
cudnn.benchmark = True
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))


print_args(vars(args))

if args.dataset == 'Indian':
    data = loadmat('../data/IndianPine.mat')
elif args.dataset == 'Pavia':
    data = loadmat('../data/Pavia.mat')
elif args.dataset == 'Houston':
    data = loadmat('../data/Houston.mat')
elif args.dataset == 'WHU-Hi-HC':
    data = loadmat('../data/WHU-Hi-HC/WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
    TR = loadmat('../data/WHU-Hi-HC/Train100.mat')['Train100']
    TE = loadmat('../data/WHU-Hi-HC/Test100.mat')['Test100']
elif args.dataset == 'WHU-Hi-HH':
    data = loadmat('../data/WHU-Hi-HH/WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
    TR = loadmat('../data/WHU-Hi-HH/Train100.mat')['HHCYtrain100']
    TE = loadmat('../data/WHU-Hi-HH/Test100.mat')['HHCYtest100']
elif args.dataset == 'WHU-Hi-LK':
    data = loadmat('../data/WHU-Hi-LK/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
    TR = loadmat('../data/WHU-Hi-LK/Train100.mat')['LKtrain100']
    TE = loadmat('../data/WHU-Hi-LK/Test100.mat')['LKtest100']
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

input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:, :, i])
    input_min = np.min(input[:, :, i])
    input_normalize[:, :, i] = (input[:, :, i] - input_min) / (input_max - input_min)

height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))

if args.pretrain_mode == 'spatial':
    args.number_patches = args.input_size * args.input_size
elif args.pretrain_mode == 'spectral':
    args.number_patches = band

total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = choose_train_and_test_point(
    TR, TE, label, num_classes)
print("Number of unlabelled samples : ", total_pos_true.shape[0])

mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.input_size)
x_train_band, x_test_band, x_unlabelled = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test,
                                                              total_pos_true, patch=args.input_size)
x_train = torch.from_numpy(x_unlabelled.transpose(0, 3, 1, 2)).type(torch.FloatTensor)

masked_positional_generator = RandomMaskingGenerator(args.number_patches, args.mask_ratio)
bool_masked_pos_t = torch.zeros(x_unlabelled.shape[0], args.number_patches)

for b in range(x_train.shape[0]):
    bool_masked_pos_t[b, :] = torch.from_numpy(masked_positional_generator())
bool_masked_pos_t = bool_masked_pos_t > 0

Label_train = Data.TensorDataset(x_train, bool_masked_pos_t)
label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)

if args.pretrain_mode == 'spatial':
    encoder = VisionTransformerForMIM(image_size=args.input_size, bands=band, num_classes=0, dim=64, depth=5, heads=4, mlp_dim=8, pretran_mode='spatial')
elif args.pretrain_mode == 'spectral':
    encoder = VisionTransformerForMIM(image_size=args.input_size, bands=band, num_classes=0, dim=32, depth=5, heads=4, mlp_dim=4, pretran_mode='spectral')
model = HyperMIM(encoder, band, args.input_size, args.pretrain_mode)

model = model.cuda()
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model = %s" % str(model))
print('number of params: {}'.format(n_parameters))

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 10, gamma=args.gamma, verbose=True)

start_epoch = 0

print("start training")
tic = time.time()
for epoch in range(start_epoch, args.epochs):

    model.train()
    train_obj = train_epoch(model, label_train_loader, optimizer, args.pretrain_mode)
    scheduler.step()
    print("Epoch: {:03d} train_loss: {:.8f}".format(epoch + 1, train_obj))
    if args.output_dir:
        if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            save_model(
                args=args, model=model, optimizer=optimizer,
                epoch=epoch)


toc = time.time()
print("Total Run Time: {:.2f}".format(toc - tic))
print("**************************************************")
