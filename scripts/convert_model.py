import argparse
import os
import sys
from config import Config
import torch as th
import pytorch_lightning as pl
from models import LSTMModel, BertBaseModel, GRUModel
from utils import natural_keys
import warnings
warnings.filterwarnings(action="ignore")


parser = argparse.ArgumentParser()

parser.add_argument(
    '--model-type',
    '-mt',
    type=str,
    default='lstm',
    required=True,
    help='Type of model architechture to use, one of lstm, gru, bert'
)


parser.add_argument(
    '--version',
    '-v',
    type=str,
    required=True,
    help='Version of experiment'
)

parser.add_argument(
    '--base',
    '-b',
    type=str,
    required=True,
    help='Base architecture'
)

if __name__ == '__main__':
    args = parser.parse_args()

    print('[INFO] Building model architecture')
    if args.model_type.lower() == 'lstm':
        model = LSTMModel()
    elif args.model_type.lower() == 'gru':
        model = GRUModel()
    else:
        model = BertBaseModel()

    print('[INFO] Model built')
    print('[INFO] Fetching model checkpoint')

    matching_models = [f for f in os.listdir(
        Config.models_dir) if args.base in f and args.model_type in f and args.version in f]

    matching_models.sort(key=natural_keys)
    print(f'[INFO] ({len(matching_models)}) matching models found')
    # print(matching_models)
    # sys.exit()
    # load the last saved checkpoint from the matching models
    mname = matching_models[-1]
    print(f'[INFO] Loading {mname} weights since it is the last one')
    try:
        model.load_from_checkpoint(
            checkpoint_path=os.path.join(Config.models_dir, mname)
        )
        print(f'[INFO] Converting model to scriptmodule')

        fn = f'arabizi-sentiments-{Config.base_model}-{args.model_type}-version-{args.version}.pth'
        th.jit.save(
            model.to_torchscript(),
            os.path.join(
                Config.models_dir,
                fn
            )
        )
        print(f'[INFO] Model saved as {fn}')

    except Exception as e:
        print('[ERROR] Not able to load the requested .ckpt file', e)
