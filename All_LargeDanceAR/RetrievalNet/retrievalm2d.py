import sys

sys.path.append(sys.path[0]+r"/../")
import functools
import json
import multiprocessing
import pickle as pkl
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime
from os.path import join as pjoin

import ipdb
import numpy as np
import torch
from configs import get_config
from datasets import (EvaluatorModelWrapper, get_dataset_motion_loader,
                      get_motion_loader)
from models import *
from tqdm import tqdm
from utils.metrics import *
from utils.plot_script import *
from utils.utils import *

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

torch.multiprocessing.set_sharing_strategy('file_system')
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists
def sort_by_dist(data_list, reverse=False):
    sortlist =  sorted(data_list, key=lambda x: x['dist'], reverse=reverse)
    for one in sortlist:
        del one['dist']
    return sortlist

def evaluate_matching_score(args, motion_loaders, audio_batch, eval_wrapper):
    match_score_dict = OrderedDict({}) 
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    audio_embeddings = eval_wrapper.get_co_embeddings(audio_batch, "audio")
    audio_embeddings = audio_embeddings.cpu().numpy()
    # print(motion_loaders.keys())
    print('========== Evaluating MM Distance ==========')
    res_list = [] 
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        
        # print(motion_loader_name)
        with torch.no_grad():
            ttttnum = 0
            for idx, motion_batch in tqdm(enumerate(motion_loader)):
                try:
                    seqnames, _, _ = motion_batch
                except:
                    seqnames = motion_batch
                # ipdb.set_trace()
                # if idx == 0:
                # motion_embeddings = eval_wrapper.get_co_embeddings(motion_batch, "motion")
                # for motionidx in range(motion_embeddings.shape[0]):
                #     # print(os.path.join('/data1/hzy/AllDataset/motionembeding', seqnames[motionidx]+'.npy'))
                #     time.sleep(0.01 )
                #     np.save(os.path.join('/data1/hzy/AllDataset/motionembeding', seqnames[motionidx]+'.npy'), motion_embeddings[motionidx].cpu().numpy())
                #     ttttnum = ttttnum+1
                # print('ttttnumttttnumttttnumttttnumttttnumttttnum', ttttnum) 
                # sys.exit(0)           
                motion_embeddings = []
                for motionidx in range(len(seqnames)):
                    # 93服务器
                    # motion_embeddings.append(np.load(os.path.join('/data1/hzy/AllDataset/motionembeding', seqnames[motionidx]+'.npy')) )
                    # 45服务器
                    motion_embeddings.append(np.load(os.path.join('/data2/hzy/InfiniteDance/InfiniteDanceData/dance/motionembeding', seqnames[motionidx]+'.npy')) )
                        
                motion_embeddings = np.stack(motion_embeddings, axis=0)
                # motion_embeddings = torch.from_numpy(motion_embeddings).to(audio_embeddings)
                # dist_mat = euclidean_distance_matrix_torch(audio_embeddings, motion_embeddings)
                # argsmax = torch.argsort(dist_mat, dim=1).detach().cpu().numpy() 
                # dist_mat = dist_mat.detach().cpu().numpy()
                        
                    
                dist_mat = euclidean_distance_matrix(audio_embeddings, motion_embeddings) #.cpu().numpy() .cpu().numpy()
                argsmax = np.argsort(dist_mat, axis=1)
                
                
                # 记录每个batch中的最match的10个
                top10_idx = argsmax[:,:10]
                for jdx, nameidx in enumerate(top10_idx):
                    for _ in nameidx:
                        datadict = {}
                        datadict['muidx'] = jdx
                        datadict['name'] = seqnames[_]
                        datadict['dist'] = dist_mat[jdx,_]
                        res_list.append(datadict)
                
                # top_k_mat = calculate_top_k(argsmax, top_k=2)
                # top_k_count += top_k_mat.sum(axis=0)
                # all_size += audio_embeddings.shape[0]
         
                
            sorted_music = []
            for kdx in range(audio_embeddings.shape[0]):
                onemusic_info = []
                for one_res in res_list: 
                    if one_res['muidx'] == kdx:
                        onemusic_info.append(one_res)
                
                onemusic_info = sort_by_dist(onemusic_info)
                sorted_music.append(onemusic_info)         
            
        #     mm_dist = mm_dist_sum / all_size
        #     R_precision = top_k_count / all_size
        #     match_score_dict[motion_loader_name] = mm_dist
        #     R_precision_dict[motion_loader_name] = R_precision

        # print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        # print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')

        # line = f'---> [{motion_loader_name}] R_precision: '
        # for i in range(len(R_precision)):
        #     line += '(top %d): %.4f ' % (i+1, R_precision[i])
        # print(line)
        # print(line)

    return match_score_dict, R_precision_dict, activation_dict, sorted_music



def sliding_windows_with_repeat_padding(audio_data, win_size=384, stride=96):
    windows = []
    n = len(audio_data)
    for start in range(0, n, stride):
        end = start + win_size
        if end <= n:
            window = audio_data[start:end]
            windows.append(window)
        else:
            # needed = end - n
            # tail = audio_data[start:]
            # # Repeat the tail enough times and trim to the needed length
            # padding = np.tile(tail, (needed // len(tail) + 1))[:needed]
            # window = np.concatenate([tail, padding])
            tail = audio_data[start:]
            pad_length = win_size - len(tail)
            padded_tail = np.pad(tail, ((0, pad_length), (0, 0)), mode='wrap')
            windows.append(padded_tail)
            break
    return windows

def retrieval_motion_from_audio(input_audio_file, win_size=384, slide=96, device=None, args=None, eval_motion_loaders=None, eval_wrapper=None):
    audio_data = np.load(input_audio_file).astype(np.float32)
    audio_list = sliding_windows_with_repeat_padding(audio_data, win_size, slide)
    audio_data = np.stack(audio_list, axis=0)
    audio_data = torch.from_numpy(audio_data).to(device).to(torch.float32)
        
    # audio_data = torch.from_numpy(audio_data).unsqueeze(0).to(device).to(torch.float32)
    audio_batch = os.path.basename(input_audio_file), audio_data, torch.zeros([audio_data.shape[0], 264]).to(audio_data)
                    
    mat_score_dict, R_precision_dict, acti_dict, sorted_music = evaluate_matching_score(args, eval_motion_loaders, audio_batch, eval_wrapper)
        
    return sorted_music

def process_onefile(input_audio_file, args, win_size=384, slide=96, eval_motion_loaders=None, eval_wrapper=None):
    device = f"cuda:{args.gpu}"
    sorted_music = retrieval_motion_from_audio(input_audio_file, win_size=win_size, slide=slide, device=device, args=args, eval_motion_loaders=eval_motion_loaders, eval_wrapper=eval_wrapper)
    
    def save_info(args, input_audio_file, sorted_music, format='json'):
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(i) for i in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.generic,)):  # 包括 np.float32、np.int64 等
                return obj.item()
            else:
                return obj

        if args.savedir != "None":
            os.makedirs(args.savedir, exist_ok=True)
            if format == 'json':
                with open(os.path.join(args.savedir, os.path.basename(input_audio_file).split('.')[0] + '.json'), 'w', encoding='utf-8') as f:
                    json.dump(clean_for_json(sorted_music), f, ensure_ascii=False, indent=4)
            elif format == 'pkl':
                with open(os.path.join(args.savedir, os.path.basename(input_audio_file).split('.')[0] + '.pkl'), 'wb') as f:
                    pkl.dump(sorted_music, f)


    save_info(args, input_audio_file, sorted_music)
            
if __name__ == '__main__':
    # batch_size is fixed to 96!!
    parser = ArgumentParser()
    parser.add_argument(
        "--audiopath",  # 参数名
        type=str,  # 参数类型
        default='/data2/hzy/InfiniteDance/InfiniteDanceData/music/musicfeature_55_allmusic_pure'
        # default='/data1/hzy/AllDataset/musicfeature_55_allmusic_pure'   # /Latin-Samba61.npy
    )
    parser.add_argument(
        "--batch_size",  # 参数名
        type=int,  # 参数类型
        default=256
    )
    parser.add_argument(
        "--gpu",  # 参数名
        type=int,  # 参数类型
        default=0
    )
    parser.add_argument(
        "--range",  # 参数名
        type=str,  # 参数类型
        default="all"
    )
    parser.add_argument(
        "--savedir",  # 参数名
        type=str,  # 参数类型
        default="None",
    )
    args = parser.parse_args()
    device = f"cuda:{args.gpu}"
    batch_size = args.batch_size
    
    eval_motion_loaders = {}
    data_cfg = get_config("./checkpoints/AInterClip_Audio55Motion264/0512/train/bc256_s100l384_drop0.2_lr1e-4/datasets45.yaml").largedance    #_test    #
    evalmodel_cfg = get_config("./checkpoints/AInterClip_Audio55Motion264/0512/train/bc256_s100l384_drop0.2_lr1e-4/InterCLIP.yaml")
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size)
    eval_motion_loaders['gt'] = gt_loader
    
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)  # InterClip
   
    if os.path.isfile(args.audiopath):
        process_onefile(args.audiopath, args=args, win_size=384, slide=192, eval_motion_loaders=eval_motion_loaders, eval_wrapper=eval_wrapper)
    else:
        filelist = os.listdir(args.audiopath)
        filelist.sort()
        selected_list = []
        for idx, file in enumerate(filelist):
            if not file.endswith('.npy'):
                continue
            if args.range == "all":
                pass
                print('all samples')
            elif args.range == "last":
                pass
                print('lasted samples')
                if os.path.exists( os.path.join(args.savedir, file.replace('npy', 'json'))):
                    temp= os.path.join(args.savedir, file.replace('npy', 'json'))
                    print(f' file exists:{temp}')
                    continue
            else:       # args.range 为 1/3，2/3，3/3，表示数据分为三份，当前处理第几份
                # assert len(args.range) == 3
                range0 = int(args.range.split('/')[0]) -1
                range1 = int(args.range.split('/')[0])
                fenmu = int(args.range.split('/')[-1])
                if int(idx) < range0*len(filelist)/fenmu or int(idx) > range1*len(filelist)/fenmu:
                    continue
            selected_list.append(file)
            print('selected_list, ', len(selected_list))
        
        
        selected_list_2 = []
        for one in selected_list:
            if not one.endswith('.npy'):
                continue
            selected_list_2.append( os.path.join(args.audiopath, one) )
        
        for onefile in tqdm(selected_list_2):
            process_onefile(onefile, args=args, win_size=384, slide=192, eval_motion_loaders=eval_motion_loaders, eval_wrapper=eval_wrapper)
        # multiprocessing.set_start_method('spawn', force=True)
        # process = functools.partial(process_onefile, args=args, win_size=384, slide=96, eval_motion_loaders=eval_motion_loaders, eval_wrapper=eval_wrapper)
        # with multiprocessing.Pool(8) as pool:
        #     for _ in tqdm(pool.imap_unordered(process, selected_list_2), total=len(selected_list_2)):
        #         pass
            
    # ipdb.set_trace()
''''
tmux new -s r3/4
conda activate t2m2 && CUDA_VISIBLE_DEVICES=5 python retrievalm2d.py --range 3/4 --savedir /data1/hzy/AllDataset/retrieval_s100_l384 


tmux new -s infer192_384
conda activate t2m2 && CUDA_VISIBLE_DEVICES=3 python retrievalm2d.py --range last --savedir  /data2/hzy/InfiniteDance/InfiniteDanceData/dance/retrieval_s192_l384

'''        
    
    

    
    