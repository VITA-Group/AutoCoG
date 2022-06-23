import argparse
import gc
import logging
import math
import os
import os.path as osp
import random
import statistics
import sys
import warnings
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.utils as tgu
import yaml
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import homophily
from tqdm import tqdm

from lib.config import cfg
from lib.config import convert_to_dict
from lib.config import update_config
from lib.config.state_space import MODEL as org_nas_cfg
from lib.core.functions import test
from lib.core.functions import train
from lib.core.functions import validate
from lib.core.functions import validate_test
from lib.model import *
from lib.util.util import logger_setup
from lib.util.util import model_setup
from lib.util.util import seed
# from lib.model.dart import Model
# from lib.model.mask import Mask

args = None
cfg = cfg
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str,
                        default=-1)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument("--logging", type=str, default="INFO")
    parser.add_argument("--search", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--run_iter", type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--id", type=str, default='')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--tune_iter', type=int, default=1)
    parser.add_argument("--p_stages", type=int, default=3, 
                       help="")
    parser.add_argument("--increase_layers", type=int, default=3,
                       help="")
    parser.add_argument("--p_cutoff", type=int, default=1,
                       help="")
    parser.add_argument("--target_n_layers", type=int, default=32)
    parser.add_argument("--p_selection_type", type=str, default='strict',
                        help="should be among [strict|random|stagger]")
    parser.add_argument('--retain-skip-edge', '-rse', action='store_true')
    parser.add_argument('--edge-finetuning', action='store_true')
    parser.add_argument('--edge-k',  type=int, default=1)
    parser.add_argument("--disable-model", action='store_true')
    parser.add_argument("--noise", type=float, default=0.0, help="for robustness aba")
    # general
    parser.add_argument('--cfg',
                        help='experiment cfg file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify cfg options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--disable-dst', action='store_true')
    args = parser.parse_args()
    return args


def check_cfg():
    global cfg
    if cfg.MODEL.HIDDEN_DIM != max(cfg.MODEL.NAS.HIDDEN_DIM):
        cfg.defrost()
        cfg.MODEL.HIDDEN_DIM = max(cfg.MODEL.NAS.HIDDEN_DIM)
    if cfg.MODEL.N_HEADS != max(cfg.MODEL.NAS.NUM_HEADS):
        cfg.defrost()
        cfg.MODEL.N_HEADS = max(cfg.MODEL.NAS.NUM_HEADS)
    cfg.freeze()

def main():
    global args, cfg
    args = parse_args()
    update_config(cfg, args)
    check_cfg()
    #
    seed(args.seed)
    cfg.defrost()
    if args.resume:
        cfg.RESUME = True
    cfg.freeze()
    #
    final_output_dir = Path(cfg.SAVE_DIR)
    final_output_dir.mkdir(parents=True, exist_ok=True)
    #
    logger = logger_setup(cfg.SAVE_DIR, args.logging)
    logger.info(f'=> creating {cfg.SAVE_DIR}')
    logger.info(cfg)
    gpu = int(args.gpu)
    if gpu >= 0:
        torch.cuda.set_device(gpu)


    run_search_train()



def get_model(model_fn,
              x=None,
              edge_index=None,
              edge_weight=None,
              arch=None,
              **kwargs):
    global cfg, args
    return model_setup(model_fn, Architecture, validate, Mask, arch, cfg, args)


def run_search_train():
    global cfg, args
    logger = logging.getLogger()
    save_search_cfg("test.yaml")
    # warmup
    seeds = [random.randint(0, 9999) for _ in range(args.run_iter)]
    seeds[0] = args.seed
    best_results = [0, 0]
    results = []

    # load seeded architecture
    edge_weights = None
    if not args.search:
        if cfg.MODEL.RANDOM_SEARCH:
            architecture = ''
        else:
            architecture = cfg.MODEL.ARCHITECTURE
            if cfg.MODEL.EDGE_WEIGHT != '':
                edge_weights = torch.load(cfg.MODEL.EDGE_WEIGHT).cuda()
            load_search_cfg(cfg.MODEL.NAS_CFG_FILE)

    master_dir = cfg.SAVE_DIR
    overall_stats = {}

    original_dict = []
    for k, v in org_nas_cfg.items():
        original_dict.append(f"MODEL.NAS.{k}")
        original_dict.append(v)
    original_dict.append("MODEL.NUM_LAYERS")
    original_dict.append(cfg.MODEL.NUM_LAYERS)

    for i in range(0, args.run_iter):
        seed(seeds[i])
        overall_stats[seeds[i]] = {}
        # change save dir
        cfg.defrost()
        cfg.SAVE_DIR = osp.join(master_dir, f"architecture_seed-{seeds[i]}")
        cfg.freeze()
        if not osp.isdir(cfg.SAVE_DIR):
            os.mkdir(cfg.SAVE_DIR)
        #
        if args.search:
            architecture, edge_index, edge_weights = model_search(
                run_iter=i,
                subscript='model-search')

        if args.tune:
            cfg.defrost()
            prev_setting = cfg.GRAPH.WITH_TRAIN
            cfg.GRAPH.WITH_TRAIN = False
            cfg.freeze()
            tune_results = []
            for j in range(0, args.tune_iter):
                seed(j)
                res, _, _ = model_train(
                    arch=architecture,
                    run_iter=i,
                    subscript=f'tune-{j}',
                    edge_weights=edge_weights,
                    edge_index=edge_index,
                )
                tune_results.append(res)
            cfg.defrost()
            cfg.GRAPH.WITH_TRAIN = prev_setting
            cfg.freeze()
            ################33
            avg_res = sum(tune_results) / len(tune_results) * 100
            max_res = max(tune_results) * 100
            min_res = min(tune_results) * 100
            stdev = statistics.stdev(tune_results) * 100
            overall_stats[seeds[i]]['avg_res'] = avg_res
            overall_stats[seeds[i]]['max_res'] = max_res
            overall_stats[seeds[i]]['min_res'] = min_res
            overall_stats[seeds[i]]['stdev'] = stdev
            ##################

        logger.info(f"performance with seed {seeds[i]}")
        for k, v in overall_stats[seeds[i]].items():
            logger.info(f"{k}: {v}")
        # restore org cfg
        if args.search:
            cfg.defrost()
            cfg.merge_from_list(original_dict)
            cfg.MODEL.HIDDEN_DIM = max(cfg.MODEL.NAS.HIDDEN_DIM)
            cfg.MODEL.N_HEADS = max(cfg.MODEL.NAS.NUM_HEADS)
            cfg.freeze()

    logger.info("summary of the entire process")
    best_avg_res = 0
    best_seed_arch = False
    best_seed_stat = False
    for no, stats in overall_stats.items():
        logger.info(f"performance with seed {no}")
        for k, v in stats.items():
            logger.info(f"{k}: {v}")
        if best_avg_res < stats['avg_res']:
            best_avg_res = stats['avg_res']
            best_seed_arch = no
            best_seed_stat = stats
    ##########################
    logger.info("##########################")
    logger.info(f"best seed belong to : {best_seed_arch}")
    for k, v in best_seed_stat.items():
        logger.info(f"{k}: {v}")

    torch.cuda.empty_cache()
    gc.collect()


def model_search(**kwargs):
    global cfg, args, edge_index
    logger = logging.getLogger()
    cfg.defrost()
    cfg.MODEL.IS_SEARCH = True
    cfg.freeze()
    starting_layer = cfg.MODEL.NUM_LAYERS
    edge_weights = None
    edge_index = None
    num_edges = None
    mask = None #Mask(cfg, pkg[0].x.size(1)).cuda()
    for i in range(args.p_stages):
        pkg = get_model(Model, **kwargs)
        if not mask:
            mask = Mask(cfg, pkg[0].x.size(1)).cuda()
        if num_edges is None:
            num_edges = pkg[0].edge_index.size(1)
        architecture, edge_weights, cos_sim = loop_search(*pkg,
                                                          subscript=f"{kwargs['subscript']}_stage-{i}",
                                                          epoch=cfg.TRAIN.SEARCH_EPOCH,
                                                          run_iter=kwargs['run_iter'],
                                                          patience=cfg.TRAIN.SEARCH_PATIENCE,
                                                          edge_weights=edge_weights,
                                                          edge_index=edge_index,
                                                          mask=mask
                                                          )

        if args.disable_dst:
            edge_weights = None  # @adj_matrixes[-1]
            edge_index = None
        else:
            if edge_index is None:
                edge_index = pkg[0].edge_index
            logger.info(f"homophily before grow: {homophily(edge_index, pkg[0].y)}")
            # edge_weights[0:num_edges] = (edge_weights[0:num_edges] + 1)/2
            edge_weights, edge_index = cut(edge_weights, edge_index, p=0.05)
            edge_weights, edge_index = grow(edge_weights, edge_index, cos_sim, pkg[0].distance, k=args.edge_k)

            logger.info(f"homophily after grew: {homophily(edge_index, pkg[0].y)}")
            logger.info(f"growing our edges!! total edges is: {edge_weights.size(0)}, original: {num_edges}")

        if filter_nas_settings(architecture):  # directly modify global nas configuration
            save_search_cfg(f"stage-{i}_cfg.yaml")
            cfg.defrost()
            cfg.MODEL.NUM_LAYERS = min(starting_layer + (i + 1) * args.increase_layers, args.target_n_layers)
            cfg.freeze()
        else:
            break
        del pkg, architecture
        torch.cuda.empty_cache()
        gc.collect()
    save_search_cfg("final_cfg.yaml")  # saving final search space as yaml
    cfg.defrost()
    cfg.MODEL.NUM_LAYERS = args.target_n_layers
    cfg.freeze()
    pkg = get_model(Model, **kwargs)
    # mask = Mask(cfg, pkg[0].x.size(1)).cuda()
    if mask is None:
        mask = Mask(cfg, pkg[0].x.size(1)).cuda()

    if args.disable_dst:
        edge_weights = None
        edge_index = None
    architecture, edge_weights, cos_sim = loop_search(*pkg,
                                                      subscript=f"{kwargs['subscript']}_final_stage",
                                                      epoch=cfg.TRAIN.SEARCH_EPOCH,
                                                      run_iter=kwargs['run_iter'],
                                                      patience=cfg.TRAIN.SEARCH_PATIENCE,
                                                      edge_weights=edge_weights,
                                                      edge_index=edge_index,
                                                      mask=mask
                                                      )
    if edge_index is None:
        edge_index = pkg[0].edge_index
    if not args.disable_dst:
        logger.info(f"final homophily rate {homophily(edge_index, pkg[0].y)}")
        logger.info(f"Final edge count is {edge_weights.size(0)}")
    else:
        edge_weights = pkg[0].edge_weight
        edge_index = pkg[0].edge_index
    #
    torch.save({'edge_weights': edge_weights, 'edge_index': edge_index},
               osp.join(cfg.SAVE_DIR, 'binary_edge_weights.pth.tar'))
    return architecture, edge_index, edge_weights


def loop_search(data, model, architecture, optimizer, loss_fn, run_iter, epoch, emb=None, subscript='', patience=100,
                mask=None, edge_index=None, edge_weights=None):
    global cfg, args

    dst = cfg.SAVE_DIR
    logger = logging.getLogger()
    is_search = True
    settings = cfg.TRAIN
    best_result = 0
    desc = "model train" if not is_search else "model search"
    pbar = tqdm(range(epoch), desc=f"{desc}-{subscript}",
                ncols=100) if not args.verbose else range(epoch)
    patience_cnt = patience
    best_results = float('inf')
    best_test_results = 0
    if edge_index is not None:
        assert edge_weights is not None
        assert edge_weights.size(0) == edge_index.size(1)
        data.edge_index = edge_index
    if edge_weights is None:
        edge_weights = data.edge_weight.clone()
    for cur_epoch in pbar:
        if architecture and cur_epoch >= settings.BEGIN_EPOCH:
            architecture.decay()
        info_dump = {}
        if not args.disable_dst:
            edge_weights = mask(data.edge_index, data.x)
            mask.optimizer.zero_grad()
            model_loss = train(cfg, model, optimizer, loss_fn, data, cur_epoch, edge_weights, architecture,
                               mask_loss=mask.get_loss())
            if cfg.MODEL.CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm(mask.parameters(),
                                              cfg.MODEL.CLIP_NORM)
            mask.optimizer.step()
        else:
            edge_weights = data.edge_weight
            model_loss = train(cfg, model, optimizer, loss_fn, data, cur_epoch, edge_weights, architecture)
            mask.optimizer.step()

        with torch.no_grad():
            model.eval()
            mask.eval()
            if not args.disable_dst:
                edge_weights = mask(data.edge_index, data.x)
            else:
                edge_weights = data.edge_weight
            acc_valid, acc_test, loss_val, loss_test = validate_test(cfg, model, data, edge_weights, loss_fn)
            if loss_val < best_results:
                best_test_results = acc_test
                best_results = loss_val
                patience_cnt = patience
                p = osp.join(dst, f'best_architecture_{run_iter}_{subscript}.pth.tar')
                plogging(f"saving best architecture at: {p} | val loss: {loss_val}, test acc:{acc_test}")
                if not args.disable_model:
                    architecture.save_architecture(p)
                torch.save(model.state_dict(), p)
                torch.save(edge_weights, osp.join(dst, f'best_edge_weights_{run_iter}_{subscript}.pth.tar'))
                if args.disable_dst:
                    cos_sim = get_cosine_sim(model, data, edge_weights)
                else:
                    mask(data.edge_index, data.x)
                    cos_sim = torch.sigmoid(mask.__z__ @ mask.__z__.t())

            # best_results = max(acc_valid, best_results)
            model.train()
            mask.train()
        # total_edge_weights = new_edge_weight.detach().clone()
        msg = f"Epoch {cur_epoch}/{epoch}: train loss: {model_loss:.5f}"
        if is_search and not args.disable_model:
            status = architecture.status()
            for k, v in status.items():
                msg = msg + f'  {k}: {v:.5f}'
        msg = msg + f" test acc: {acc_test:.5f}"
        msg = msg + f" valid  acc: {acc_valid:.5f}"
        msg = msg + f" edges: {data.edge_index.shape[1]}"
        if not args.disable_dst:
            msg = msg + f"mask loss: {mask.__loss__.item()}"
        if cur_epoch % cfg.PRINT_FREQ == 0:
            plogging(msg)
        info_dump.update({
            'epoch': cur_epoch,
            f'train_loss_{subscript}': model_loss,
            f'test_acc_{subscript}': acc_test,
        })
        if architecture and cur_epoch >= cfg.TRAIN.BEGIN_EPOCH:
            loss_valid = architecture.loss_valid
            info_dump[f'valid_acc_{subscript}'] = acc_valid
            info_dump[f'valid_loss_{subscript}'] = loss_valid.item()
            info_dump.update(architecture.get_entropy())
        info_dump[f'best_acc_{subscript}'] = best_result

        if cur_epoch % cfg.CHECKPOINT_FREQ == 0:

            if is_search:
                p = osp.join(dst, 'checkpoint_search.pth.tar')
            else:
                p = osp.join(dst, 'checkpoint_train.pth.tar')
            plogging(f"saving checkpoint at: {p}")
            ckp = {}
            ckp['model'] = model.state_dict()
            ckp['optimizer'] = optimizer.state_dict()
            ckp['current_epoch'] = cur_epoch
            ckp['best_result'] = best_result
            if architecture is not None:
                ckp['architecture'] = architecture.state_dict()
            torch.save(ckp, p)
        patience_cnt -= 1
        if patience_cnt <= 0:
            break
    logger.info(f"best test results: {best_test_results}")
    if not args.disable_model:
        p = osp.join(dst, f'final_architecture_{run_iter}_{subscript}.pth.tar')
        architecture.save_architecture(p)
        torch.save(edge_weights, osp.join(dst, f'final_edge_weights_{run_iter}_{subscript}.pth.tar'))

        p = osp.join(dst, f'best_architecture_{run_iter}_{subscript}.pth.tar')
        architecture.load_architecture(p)
        architecture.summary()
    else:
        p = ''
    ##########################
    ###########################
    return p, torch.load(osp.join(dst, f'best_edge_weights_{run_iter}_{subscript}.pth.tar')), cos_sim


def loop_train(data, model, architecture, optimizer, loss_fn, run_iter, epoch, emb=None, subscript='',
               edge_weights=None, edge_index=None, mask=None, **kwargs):
    global cfg, args
    dst = cfg.SAVE_DIR
    logger = logging.getLogger()
    is_search = False
    settings = cfg.TRAIN
    best_result = float('inf')
    desc = "model train" if not is_search else "model search"
    pbar = tqdm(range(epoch), desc=f"{desc}-{subscript}",
                ncols=100) if not args.verbose else range(epoch)
    patience = cfg.TRAIN.TRAIN_PATIENCE
    patience_cnt = cfg.TRAIN.TRAIN_PATIENCE
    best_results = float('inf')
    best_test_results = 0
    if edge_index is not None:
        assert edge_weights is not None
        data.edge_index = edge_index
        print(f"training with #{edge_index.size(1)} edges")
    if edge_weights is None:
        edge_weights = data.edge_weight.clone()
    for cur_epoch in pbar:
        info_dump = {}
        if mask is not None:
            new_edge_weight = mask(data.edge_index, data.x, edge_weights)
            # random drop
            # new_edge_weight = mask.random_drop(new_edge_weight)
            #
            mask.optimizer.zero_grad()
            model_loss = train(cfg, model, optimizer, loss_fn, data, cur_epoch, new_edge_weight, architecture)
            if cfg.MODEL.CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm(mask.parameters(),
                                              cfg.MODEL.CLIP_NORM)
            mask.optimizer.step()
        else:
            new_edge_weight = edge_weights
            model_loss = train(cfg, model, optimizer, loss_fn, data, cur_epoch, new_edge_weight, architecture)

        with torch.no_grad():
            model.eval()
            if mask is not None:
                mask.eval()
                new_edge_weight = mask(data.edge_index, data.x, edge_weights)
                mask.train()
            else:
                new_edge_weight = edge_weights
            acc_valid, acc_test, loss_val, loss_test = validate_test(cfg, model, data, new_edge_weight, loss_fn)
            if loss_val < best_results:
                # print(f"best results: valid: {acc_valid}, test: {acc_test}")
                best_test_results = acc_test
                best_results = loss_val
            # best_results = max(acc_valid, best_results)
                patience_cnt = patience
                best_result = acc_valid
                p = osp.join(dst, f'best_model_{run_iter}_{subscript}.pth.tar')
                plogging(f"saving best architecture at: {p} | val loss: {loss_val}, test acc:{acc_test}")
                torch.save(model.state_dict(), p)
            model.train()

        msg = f"Epoch {cur_epoch}/{epoch}: train loss: {model_loss:.5f}"

        msg = msg + f" test acc: {acc_test:.5f}"
        msg = msg + f" valid  acc: {acc_valid:.5f}"
        msg = msg + f" edges: {data.edge_index.shape[1]}"
        if cur_epoch % cfg.PRINT_FREQ == 0:
            plogging(msg)
        info_dump.update({
            'epoch': cur_epoch,
            f'train_loss_{subscript}': model_loss,
            f'test_acc_{subscript}': acc_test,
        })

        info_dump[f'best_acc_{subscript}'] = best_result

        if cur_epoch % cfg.CHECKPOINT_FREQ == 0:

            if is_search:
                p = osp.join(dst, 'checkpoint_search.pth.tar')
            else:
                p = osp.join(dst, 'checkpoint_train.pth.tar')
            plogging(f"saving checkpoint at: {p}")
            ckp = {}
            ckp['model'] = model.state_dict()
            ckp['optimizer'] = optimizer.state_dict()
            ckp['current_epoch'] = cur_epoch
            ckp['best_result'] = best_result
            if architecture is not None:
                ckp['architecture'] = architecture.state_dict()
            torch.save(ckp, p)
        patience_cnt -= 1
        if patience_cnt <= 0:
            break
    logger.info(f"best test results: {best_test_results}")

    p = osp.join(dst, f'final_model_{run_iter}_{subscript}.pth.tar')
    # logger.info("saving final model")
    torch.save(model.state_dict(), p)
    #
    best_index = data.edge_index
    best_weight = data.edge_weight
    return best_test_results, best_index, best_weight  # data.edge_index, data.edge_weight.detach(


def filter_nas_settings(architecture):
    global cfg, args
    if args.disable_model:
        return True
    logger = logging.getLogger()
    logger.info("##############################")
    if isinstance(architecture, str):
        weights = torch.load(architecture)
    elif isinstance(architecture, Tensor):
        weights = architecture
    cfg.defrost()
    nas_settings = cfg.MODEL.NAS
    weights = {k: v for k, v in weights.items() if 'arch' in k}
    if args.p_selection_type == 'strict':
        keys_to_remove = weights.keys()
    elif args.p_selection_type == 'random':
        keys = weights.keys()
        keys_to_remove = random.sample(keys, len(keys) // 2)
    elif args.p_selection_type == 'stagger':
        keys = [k for k in weights.keys() if k != 'arch_residual']
        max_l = max([v.size(1) for k, v in weights.items() if k != 'arch_residual'])
        keys_to_remove = list(filter(lambda x: weights[x].size(1) >= max_l, keys))
    else:
        raise NotImplementedError

    new_nas_config = []
    for k in keys_to_remove:
        v = weights[k]
        if args.retain_skip_edge and k.lower() == 'arch_first_to_skip':
            continue
        else:
            name = "_".join(k.split("_")[1::]).upper()
            options = getattr(nas_settings, name)
            v = v.mean(0)
            _, idx = v.topk(v.shape[0])
            idx = idx.tolist()
            #
            if len(options) > args.p_cutoff:
                # a special consideration for data policy
                if k == 'arch_data_policy' and v.shape[0] <= 2:
                    continue
                else:
                    removed_option = [options[i] for i in idx[-args.p_cutoff::]]
                    sel_idx = idx[0:-args.p_cutoff]
                    sel_idx.sort()
                    new_options = [options[i] for i in sel_idx]
                    logger.info(f"{name} -> new options: {new_options} || removed : {removed_option}")
                    new_nas_config.append(f"MODEL.NAS.{name}")
                    new_nas_config.append(new_options)

    cfg.merge_from_list(new_nas_config)
    cfg.MODEL.HIDDEN_DIM = max(cfg.MODEL.NAS.HIDDEN_DIM)
    cfg.MODEL.N_HEADS = max(cfg.MODEL.NAS.NUM_HEADS)
    cfg.freeze()
    return len(new_nas_config) > 0


def save_search_cfg(name):
    global cfg
    cfg_dict = convert_to_dict(cfg.MODEL.NAS)
    save_dir = osp.join(cfg.SAVE_DIR, name)
    with open(save_dir, 'w') as fp:
        yaml.dump(cfg_dict, fp, width=float('inf'))


def load_search_cfg(name):
    global cfg
    save_dir = name  # osp.join(cfg.SAVE_DIR, name)
    with open(save_dir, 'r') as fp:
        nas_dict = yaml.load(fp)
        nas_list = []
        for k, v in nas_dict.items():
            nas_list.append(f"MODEL.NAS.{k}")
            nas_list.append(v)
        cfg.defrost()
        cfg.merge_from_list(nas_list)
        cfg.MODEL.HIDDEN_DIM = max(cfg.MODEL.NAS.HIDDEN_DIM)
        cfg.MODEL.N_HEADS = max(cfg.MODEL.NAS.NUM_HEADS)
        cfg.MODEL.NUM_LAYERS = args.target_n_layers
        cfg.freeze()


def model_train(**kwargs):  # arch , edges, edge_weight, run_iter=0):
    global cfg, args
    cfg.defrost()
    cfg.MODEL.IS_SEARCH = False
    cfg.freeze()
    pkg = get_model(Model, **kwargs)
    if args.edge_finetuning:
        mask = Mask(cfg, pkg[0].x.size(1)).cuda()
    else:
        mask = None
    epoch = kwargs.get('epoch', cfg.TRAIN.TRAIN_EPOCH)
    return loop_train(*pkg,
                      epoch=epoch,
                      mask = mask,
                      **kwargs)


def plogging(msg):
    global cfg, args
    logger = logging.getLogger()
    if args.verbose:
        logger.info(msg)


def get_graph_info(edges, attr, num_nodes, y):
    with torch.no_grad():
        info = {}
        info['isolated_nodes'] = tgu.contains_isolated_nodes(edges, num_nodes)
        info['homophily'] = tgu.homophily(edges, y)
        info['sparsity'] = edges.size(1)
        adj = tgu.to_dense_adj(edges, edge_attr=attr).squeeze(0)
        info.update(get_row_entropy(adj))
        info.update(get_row_norm(adj))
        return info


def get_row_entropy(adj):
    entropies = {}
    running_mean = 0
    prob = F.softmax(adj, dim=-1)
    log_prob = F.log_softmax(adj, dim=-1)
    e = -(log_prob * prob).sum(-1, keepdim=False)
    entropies[f'graph_entropy_mean'] = e.mean()
    # e = e.tolist()
    # for row_i, e_i in enumerate(e):
    #     entropies[f'graph_entropy_{row_i}'] = e_i
    return entropies


def get_row_norm(adj):
    norm = {}
    # running_mean = 0
    adj_row_norm = adj.norm(dim=-1)
    norm['graph_norm_mean'] = adj_row_norm.mean()
    # adj_row_norm = adj_row_norm.tolist()
    # for row_i, n in enumerate(adj_row_norm):
    #     norm[f'graph_norm_{row_i}'] = n #adj[row_i,:].norm()
    # norm['graph_norm_mean'] = running_mean
    return norm


@torch.no_grad()
def get_cosine_sim(model, data, edge_weights, eps=1e-8):
    model.eval()
    model(data.x, data.edge_index, edge_weights)
    p = model.final_embeddings
    n1 = p.norm(p=2, dim=1, keepdim=True)
    # n2 = y.norm(p=2, dim=1, keepdim=True)
    return p / n1.clamp(min=eps) @ (p / n1.clamp(min=eps)).t()


def replenish_edge(mask, adj, similarities, pkg):
    global args, cfg
    edge_weights = mask.binarize(adj, cfg.MODEL.MASK_P)
    adj = adj[edge_weights > 0]
    print(
        f"total edges: {edge_weights.sum()} | exiting edges: {pkg[0].edge_index.size(1)}| loss %: {edge_weights.sum() / pkg[0].edge_index.size(1) * 100}")
    ##### filter zero value edges
    edge_index = pkg[0].edge_index
    src_nodes_to_replace = torch.unique(edge_index[0, edge_weights == 0])
    edge_index = edge_index[:, edge_weights > 0]

    if args.replenish_edges:
        avg_sim = sum(similarities) / len(similarities)
        avg_sim = (1 + avg_sim) / pkg[0].distance
        avg_sim.fill_diagonal_(0)
        avg_sim[edge_index[0], edge_index[1]] = 0
        n_new_edges = edge_weights.size(0) - edge_weights.sum()
        edges_per_node = n_new_edges // src_nodes_to_replace.size(0)  # data.x
        edges_per_node = int(edges_per_node.item())
        values, index = avg_sim[src_nodes_to_replace].topk(edges_per_node, dim=-1)
        new_edges = torch.stack([
            src_nodes_to_replace.tile(edges_per_node, 1).t().contiguous().view(-1).to(index.device),
            index.view(-1)
        ])
        edge_index = torch.cat([edge_index, new_edges], dim=-1)
        edge_weights = torch.ones(edge_index.size(1)).to(edge_index.device)
        print(
            f"Replenished edges: {edge_weights.sum()} | exiting edges: {pkg[0].edge_index.size(1)}| % change: {edge_weights.sum() / pkg[0].edge_index.size(1) * 100}")
    else:
        edge_weights = torch.ones(edge_index.size(1)).to(edge_index.device)

    return edge_index, edge_weights

def grow(weight, edge_index, sim, distance, k):
    sim = (1+sim)/ torch.log(distance)
    sim.fill_diagonal_(0)
    sim[edge_index[0], edge_index[1]] = 0
    values, index = sim.topk(k, dim=-1)
    new_edges = torch.stack([
        torch.arange(sim.size(0)).tile(k, 1).t().contiguous().view(-1).to(index.device),
            # if not args.disable_loop:
            #     edge_weights = mask(data.edge_index, data.x, edge_weights)
            # else:
        index.view(-1)
    ])
    edge_index = torch.cat([edge_index, new_edges], dim=-1)
    weight = torch.cat([weight, 0.5 * torch.ones(new_edges.size(1)).to(weight.device)], dim=0)
    return weight, edge_index


def cut(weight, index, p):
    mask = weight.detach().clone().cpu().numpy()
    threshold = np.percentile(mask, math.floor(p*100))
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    mask = torch.tensor(mask).cuda()
    index = index[:, mask==1]
    weight = weight[mask==1]
    return weight, index


if __name__ == "__main__":
    main()
