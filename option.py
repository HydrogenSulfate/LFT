import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--angRes_in", type=int, default=3, help="angular resolution")
parser.add_argument("--angRes_out", type=int, default=5, help="angular resolution")
parser.add_argument('--model_name', type=str, default='LFATSRNet', help="model name")
parser.add_argument("--channels", type=int, default=64, help="channels")

parser.add_argument("--use_pre_pth", type=bool, default=False, help="use pre model ckpt")
parser.add_argument("--path_pre_pth", type=str, default='./pth/LFT_5x5_4x_epoch_50_model.pth',help="path for pre model ckpt")
parser.add_argument('--data_name', type=str, default='ALL',help='Boxer, Chess, ALL')
parser.add_argument('--test_data_name', type=str, default='ALL',help='Boxer, Chess, ALL')
parser.add_argument('--path_for_train', type=str, default='./data/LFSR_processed_rgb/data_for_train/')
parser.add_argument('--path_for_test', type=str, default='./data/LFSR_processed_rgb/data_for_test/')
parser.add_argument('--path_log', type=str, default='./log/')

parser.add_argument('--patch_size_for_test', default=256, type=int, help='patch size')
parser.add_argument('--stride_for_test', default=128, type=int, help='stride')

parser.add_argument('--global_batch_size', default=8, type=int, help='stride')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
parser.add_argument('--n_steps', type=int, default=50, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
parser.add_argument('--epi_loss', type=float, default=None, help='EPI loss weight factor')
parser.add_argument('--epoch', default=200, type=int, help='Epoch to run [default: 50]')
parser.add_argument('--save_epoch', default=10, type=int, help='Epoch interval when save [default: 10]')
parser.add_argument('--num_workers', type=int, default=4, help='num workers of the Data Loader')
parser.add_argument('--amp', action='store_true', help='Whether use fp16 mode')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0)

args = parser.parse_args()
