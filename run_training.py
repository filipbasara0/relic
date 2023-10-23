import argparse

from train import train_relic

parser = argparse.ArgumentParser(description='RELIC')
parser.add_argument('--dataset_path',
                    default='./data',
                    help='Path where datasets will be saved')
parser.add_argument('--dataset_name',
                    default='stl10',
                    help='Dataset name',
                    choices=['stl10', 'cifar10', 'mnist'])
parser.add_argument(
    '-m',
    '--encoder_model_name',
    default='resnet18',
    choices=['resnet18', 'convnext'],
    help='model architecture: resnet18 or convnext (default: resnet18)')
parser.add_argument('-save_model_dir',
                    default='./models',
                    help='Path where models')
parser.add_argument('--num_epochs',
                    default=200,
                    type=int,
                    help='Number of epochs for training')
parser.add_argument('--warmup_epochs',
                    default=10,
                    type=int,
                    help='Number of warmup epochs')
parser.add_argument('-b',
                    '--batch_size',
                    default=256,
                    type=int,
                    help='Batch size')
parser.add_argument('-lr', '--learning_rate', default=3e-4, type=float)
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float)
parser.add_argument(
    '--fp16_precision',
    action='store_true',
    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--proj_out_dim',
                    default=128,
                    type=int,
                    help='Projector MLP out dimension')
parser.add_argument('--proj_hidden_dim',
                    default=4096,
                    type=int,
                    help='Projector MLP hidden dimension')
parser.add_argument('--log_every_n_steps',
                    default=400,
                    type=int,
                    help='Log every n steps')
parser.add_argument('--gamma',
                    default=0.996,
                    type=float,
                    help='Initial EMA coefficient')
parser.add_argument('--tau',
                    default=1.0,
                    type=float,
                    help='Softmax temperature')
parser.add_argument('--alpha',
                    default=1.0,
                    type=float,
                    help='KL divergence loss parameter')
parser.add_argument('--update_gamma_after_step',
                    default=100,
                    type=int,
                    help='Update EMA gamma after this step')
parser.add_argument('--update_gamma_every_n_steps',
                    default=1,
                    type=int,
                    help='Update EMA gamma after this many steps')


def main():
    args = parser.parse_args()
    train_relic(args)


if __name__ == "__main__":
    main()