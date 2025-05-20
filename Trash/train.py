import argparse
import os
import training
import model
import dataset
import wandb
import torch
from typing import *
import training

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_name(params: Dict[str, Any]) -> str:
    """
    Generate a name for the experiment based on the given parameters.

    Args:
        params (Dict[str, Any]): Dictionary containing experiment parameters.

    Returns:
        str: Name for the experiment.
    """
    keys = [key for key in params.keys()]
    values = [params[key] for key in keys]

    name = ''
    for key, val in zip(keys, values):
        name += ''.join([i[0] for i in key.split('_')])+':'+str(val)+'_'
    return name


def train_model(args):
    pass


def train_sweep(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Trains the model using the wandb sweep function.

    Args:
        config (Optional[Dict[str, Any]], optional): Configuration for the experiment. Defaults to None.

    Returns:
        None
    """
    with wandb.init(config=config) as run:
        config = wandb.config  # type: Dict[str, Any]
        run.name = get_name(config)  # type: str

        input_lang, output_lang, train_loader, valid_loader = dataset.get_dataloader(
            config["batch_size"])  # type: dataset.Languages, dataset.Languages, torch.utils.data.DataLoader, torch.utils.data.DataLoader
        encoder = model.Encoder(
            type_=config["cell_type"],
            num_layers_=config["encoder_num_layers"],
            hidden_dim_=config["hidden_units"],
            embed_dim_=config["embed_dim"],
            dropout_rate=config["dropout"],
            bidirectional_=config["bidirectional"],
            batch_first_=True).to(device)
        decoder = model.Decoder(
            type=config["cell_type"],
            num_layers=config["decoder_num_layers"],
            hidden_dim=config["hidden_units"],
            output_dim=config["output_dim"],
            embed_dim=config["embed_dim"],
            dropout_rate=config["dropout"],
            bidirectional_=config["bidirectional"],
            batch_first=True).to(device)
        training.train(train_loader, valid_loader, input_lang, output_lang,
                       encoder, decoder, config["epochs"], wandb_log=True, name=run.name, learning_rate=config["learning_rate"])


def main() -> None:
    """
    Parses command line arguments and trains the model.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--epochs', type=int,
                        default=100, help='number of epochs to train for')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=64, help='input batch size for training')
    parser.add_argument('-pe', '--print_every', type=int,
                        default=10, help='number of batches to print the loss')
    parser.add_argument('-hu', '--hidden_units', type=int,
                        default=512, help='number of hidden units')
    parser.add_argument('-ed', '--embed_dim', type=int,
                        default=256, help='number of embedding dimensions')
    parser.add_argument('-nel', '--encoder_num_layers', type=int,
                        default=2, help='number of encoder layers')
    parser.add_argument('-ndl', '--decoder_num_layers', type=int,
                        default=2, help='number of decoder layers')
    parser.add_argument('-dp', '--dropout', type=float,
                        default=0.2, help='dropout probability')
    parser.add_argument('-cell', '--cell_type', type=str,
                        default='LSTM', help='RNN cell type')
    parser.add_argument('-bid', '--bidirectional', type=bool,
                        default=True, help='whether to use bidirectional RNN')
    parser.add_argument('-gpu', '--gpu_number', type=int,
                        default=0, help='gpu number')
    parser.add_argument('-wp', '--wandb_project',
                        default='CS23E001_DL_3', help='wandb project name')
    parser.add_argument('-we', '--wandb_entity',
                        default='Shuvrajeet', help='wandb entity name')
    parser.add_argument('-s', '--sweep', action='store_true',
                        help='Perform hyper parameter tuning')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    if args.sweep:
        sweep_config = {
            'method': 'bayes',
        }
        metric = {
            'name': 'valid_acc',
            'goal': 'maximize'
        }
        sweep_config['metric'] = metric

        parameters_dict = {
            'epochs': {
                'values': [20, 40, 60]
            },
            'batch_size': {
                'values': [64, 128, 256]
            },
            'hidden_units': {
                'values': [128, 256, 512]
            },
            'embed_dim': {
                'values': [64, 128, 256]
            },
            'encoder_num_layers': {
                'values': [1, 2, 3]
            },
            'decoder_num_layers': {
                'values': [1, 2, 3]
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]
            },
            'cell_type': {
                'values': ['LSTM', 'GRU', 'RNN']
            },
            'bidirectional': {
                'values': [True, False]
            },
            'learning_rate': {
                'values': [0.0005, 0.001, 0.003, 0.01]
            }
        }
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, train_sweep)


if __name__ == '__main__':
    main()
