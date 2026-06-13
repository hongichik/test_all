import os
import tempfile

if os.environ.get('NCS_SMOKE'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from logging import getLogger
from pathlib import Path

import sys

_CORE_ROOT = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from ncs_numpy_compat import apply_numpy_recbole_compat

apply_numpy_recbole_compat()

try:
    import pandas as pd

    pd.options.mode.copy_on_write = False
except ImportError:
    pass

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_trainer, init_seed, set_color

from core_ave import COREave
from core_trm import COREtrm


def _smoke_data_path(dataset: str) -> str:
    """Tạo bản .inter nhỏ (train/valid/test) cho smoke test."""
    n = int(os.environ.get("NCS_SMOKE_SAMPLES", "2000"))
    base = Path(tempfile.mkdtemp(prefix="core_smoke_"))
    dst = base / dataset
    dst.mkdir(parents=True)
    src = Path(__file__).resolve().parent / "dataset" / dataset
    train_lines = (src / f"{dataset}.train.inter").read_text(encoding="utf-8").splitlines()
    subset = "\n".join(train_lines[: n + 1]) + "\n"
    for split in ("train", "valid", "test"):
        (dst / f"{dataset}.{split}.inter").write_text(subset, encoding="utf-8")
    return str(base) + "/"


def run_single_model(args):
    # configurations initialization
    config_files = [
        str(_CORE_ROOT / 'props/overall.yaml'),
        str(_CORE_ROOT / f'props/core_{args.model}.yaml'),
    ]
    config_dict = {'data_path': str(_CORE_ROOT / 'dataset') + '/'}
    if os.environ.get('NCS_SMOKE'):
        config_files.append(str(_REPO / 'config' / 'smoke_1epoch.yaml'))
        config_dict = {
            'data_path': _smoke_data_path(args.dataset),
            'use_gpu': False,
            'gpu_id': '',
            'train_neg_sample_args': None,
            'alias_of_item_id': ['item_id_list'],
        }
    elif os.environ.get('NCS_EPOCH1'):
        config_files.append(str(_REPO / 'config' / 'full_1epoch.yaml'))
    config = Config(
        model=COREave if args.model == 'ave' else COREtrm,
        dataset=args.dataset,
        config_file_list=config_files,
        config_dict=config_dict,
    )
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    device = 'cpu' if os.environ.get('NCS_SMOKE') else config['device']
    if args.model == 'ave':
        model = COREave(config, train_data.dataset).to(device)
    elif args.model == 'trm':
        model = COREtrm(config, train_data.dataset).to(device)
    else:
        raise ValueError('model can only be "ave" or "trm".')
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='trm', help='ave or trm')
    parser.add_argument('--dataset', type=str, default='diginetica', help='diginetica, nowplaying, retailrocket, tmall, yoochoose')
    args, _ = parser.parse_known_args()

    run_single_model(args)
