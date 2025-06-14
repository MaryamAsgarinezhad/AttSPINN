from dataloader.dataloaderForSoc import NASAdata
from Model.AttSPINN import count_parameters, AttSPINN as SPINN
import argparse, os
import numpy as np
from sklearn.metrics import mean_squared_error
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calc_rmse(path):
    pred_label = np.load(os.path.join(path, "pred_label.npy"))
    true_label = np.load(os.path.join(path, "true_label.npy"))
    return np.sqrt(mean_squared_error(true_label, pred_label))

def load_data(args):
    # ---- change: load *all* batteries & modes
    data = NASAdata(root=args.data_root, args=args)
    loader_dict = data.read_all(mode=None)  # None ⇒ both 'charge' & 'discharge'
    return {
        'train': loader_dict['train_2'],
        'valid': loader_dict['valid_2'],
        'test':  loader_dict['test_3'],
    }

    # return {
    #     'train': loader_dict['train_2'],
    #     'test' : loader_dict['valid_2'],
    # }

def main():
    args = get_args()

    for exp in range(10):
        # each experiment gets its own folder
        save_folder = os.path.join(args.save_root, f"Experiment{exp+1}")
        os.makedirs(save_folder, exist_ok=True)
        args.save_folder = save_folder
        args.log_dir     = 'logging.txt'

        print(f"\n--- Experiment {exp+1} ---")
        print("Loading all NASA data...")
        dataloader = load_data(args)

        # infer feature-dim from the first batch
        x1_sample, _, _, _ = next(iter(dataloader['train']))
        x_dim = x1_sample.shape[1]

        architecture_args = {
            "solution_u_subnet_args": {
                "output_dim": 16,
                "layers_num": 5,
                "hidden_dim": 15,
                "dropout": 0,
                "activation": "leaky-relu"
            },
            "dynamical_F_subnet_args": {
                "output_dim": 16,
                "layers_num": 5,
                "hidden_dim": 15,
                "dropout": 0,
                "activation": "leaky-relu"
            },
            "attn_embed_dim_u": 16,
            "attn_heads_u": 2,
            "attn_embed_dim_F": 16,
            "attn_heads_F": 2
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = SPINN(args, x_dim=x_dim, architecture_args=architecture_args).to(device)
        # model = SPINN(args, x_dim=x_dim, architecture_args=architecture_args).cuda()
        count_parameters(model)

        print("Training on *all* NASA batteries...")
        model.Train(
            trainloader=dataloader['train'],
            validloader=dataloader['valid'],
            testloader=dataloader['test']
        )

        rmse = calc_rmse(save_folder)
        print(f"Experiment {exp+1} finished; RMSE = {rmse:.4f}")

def get_args():
    parser = argparse.ArgumentParser('Hyperparameters for NASA data')
    parser.add_argument('--data_root', type=str,
                        default='data/NASA data',
                        help='root folder containing charge/ and discharge/ subfolders')
    parser.add_argument('--save_root', type=str,
                        default='results/NASA',
                        help='base directory to save experiment outputs')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--normalization_method', type=str,
                        default='min-max', choices=['min-max','z-score'])
    # training schedule
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=30)
    parser.add_argument('--warmup_lr', type=float, default=0.002)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--final_lr', type=float, default=0.0002)
    parser.add_argument('--lr_F', type=float, default=0.001)
    # loss weights
    parser.add_argument('--alpha', type=float,
                        default=0.08549401305482651)
    parser.add_argument('--beta', type=float,
                        default=6.040426381151848)
    # logging
    parser.add_argument('--log_dir', type=str, default='logging.txt')
    return parser.parse_args()

if __name__ == '__main__':
    main()
