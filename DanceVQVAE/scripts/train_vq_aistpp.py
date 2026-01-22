# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import json
import os
import time
import warnings
from datetime import datetime, timedelta
from os.path import join as pjoin

import models.vqvae as vqvae
import options.option_vq as option_vq
import torch
import torch.optim as optim
import utils.losses as losses
import utils.utils_model as utils_model
from dataset import dataset_TM_eval, dataset_VQ
from models.evaluator_wrapper import EvaluatorModelWrapper
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import paramUtil

warnings.filterwarnings('ignore')
import ast
import subprocess
import warnings
import weakref

import numpy as np
from utils.word_vectorizer import WordVectorizer

# 忽略特定的警告
warnings.filterwarnings("ignore", message=".*weakref.*")
torch.cuda.set_device(4)
warnings.filterwarnings('ignore')

def getfid(checkpoint_path):
    result = subprocess.run(
        [
            "python", "/data1/hzy/HumanMotion/T2M-GPT_mofea264/Inferrence_Reconstruct_momask_all.py",
            "--checkpoint_path", checkpoint_path,
        ],
        capture_output=True,
        text=True,
        check=True
    )
    print("Output:", result.stdout)
    metrics = ast.literal_eval(result.stdout)
    fid = metrics['fid_k'].real
    return fid

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
if args.vel_decoder:
    args.out_dir = '/data1/hzy/HumanMotion/T2M-GPT_mofea264/output/exp_momask_alldata_mofea264_250503_2043/vel_decoder'
elif args.rot_decoder:
    args.out_dir = '/data1/hzy/HumanMotion/T2M-GPT_mofea264/output/exp_momask_alldata_mofea264_250503_2043/rot_decoder'

    

   
current_time = (datetime.now() + timedelta(hours=7.92)).strftime(f"25%m%d_%H%M")
args.out_dir = os.path.join(args.out_dir, f'exp_momask_alldata_mofea264_{current_time}')
os.makedirs(args.out_dir, exist_ok=True)
args_dict = vars(args)

# 保存为 JSON 文件
json_file_path = os.path.join(args.out_dir, 'args.json')
with open(json_file_path, 'w') as f:
    json.dump(args_dict, f, indent=4)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

# 处理数据集的输入
if args.dataname == 'kit':
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 21
elif args.dataname == 't2m':
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22
elif args.dataname == 'all':
    dataset_opt_path = 'checkpoints/all/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

train_loader, train_dataset = dataset_VQ.DATALoader(args.dataname,
                                                   args.batch_size,
                                                   sfile=args.split_file_train,
                                                   window_size=args.window_size,
                                                   unit_length=2**args.down_t)
train_loader_iter = dataset_VQ.cycle(train_loader)

eval_dataset = dataset_VQ.MotionDataset(args, split_file=args.split_file_eval, data_root=args.data_root)
eval_loader = DataLoader(eval_dataset, batch_size=128, drop_last=True, num_workers=8,
                         shuffle=True, pin_memory=True)

dim_pose = 264
net = vqvae.RVQVAE(args, 
                   dim_pose,
                   args.nb_code,
                   args.code_dim,
                   args.output_emb_width,
                   args.down_t,
                   args.stride_t,
                   args.width,
                   args.depth,
                   args.dilation_growth_rate,
                   args.vq_act,
                   args.vq_norm)

if args.resume_pth:
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=False)

net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

Loss = losses.ReConsLoss(args, args.recons_loss, args.nb_joints, args.w_motion, args.w_joints, args.w_vel, args.w_acc, args.w_foot)

##### ------ warm-up ------- #####
avg_all, avg_perplexity, avg_commit = 0., 0., 0.

print("begin warm up", args.warm_up_iter)
for nb_iter in range(1, args.warm_up_iter):
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float()  # (batch_size, 64, dim)
    
    pred_motion, pred_motion_xz, pred_motion_rot, loss_commit, perplexity, utilization_stats = net(gt_motion)
    
    if args.vel_decoder:
        pred_motion[:, :, 3:5] = pred_motion_xz
    if args.rot_decoder:
        pred_motion[:, :, [1, 2, *range(68, 194)]] = pred_motion_rot
    
    loss_motion = Loss(pred_motion, gt_motion)
    loss = loss_motion + args.commit * loss_commit
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_all += loss.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()

    if nb_iter % args.print_iter == 0:
        for i, stats in enumerate(utilization_stats):
            print(f"nb_iter {nb_iter}")
            print(f"Quantizer {i}:")
            print(f"Non-zero Codes: {stats['non_zero_codes']} / {stats['total_codes']}")
            writer.add_scalar(f'Warmup/Quantizer_{i}/Usage_Mean', stats['mean'], nb_iter)
            writer.add_scalar(f'Warmup/Quantizer_{i}/Usage_Std', stats['std'], nb_iter)
            writer.add_scalar(f'Warmup/Quantizer_{i}/Non_Zero_Codes', stats['non_zero_codes'], nb_iter)
        
        avg_all /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        logger.info("Warmup loss:{0}".format(loss))
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_all:.5f}")
        
        avg_all,avg_perplexity, avg_commit = 0., 0., 0.

##### ---- Training ---- #####
avg_all, avg_perplexity, avg_commit, avg_recons = 0., 0., 0., 0.
print("begin train")
loss_eval_min = 10000
train_loss_min = 10000
loss_eval_2 = 10000
fid_min = 10000
for nb_iter in range(1, args.total_iter + 1):
    net.train()

    gt_motion = next(train_loader_iter)
    gt_motion = gt_motion.cuda().float()  # bs, nb_joints, joints_dim, seq_len
    
    pred_motion, pred_motion_xz, pred_motion_rot, loss_commit, perplexity, utilization_stats = net(gt_motion)
    if args.vel_decoder:
        pred_motion[:, :, 3:5] = pred_motion_xz
    if args.rot_decoder:
        pred_motion[:, :, [1, 2, *range(68, 194)]] = pred_motion_rot
    
    loss_all, motion_loss, xz_vel_loss, xz_acc_loss, rot_loss, rot_acc_loss, foot_contact_loss = Loss.forward_all(pred_motion, gt_motion)
    loss = loss_all + 0.05 * loss_commit
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_all += loss.item()
    avg_recons += motion_loss.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()

    if nb_iter % args.print_iter == 0:
        for i, stats in enumerate(utilization_stats):
            print(f"Quantizer {i}:")
            print(f"  Non-zero Codes: {stats['non_zero_codes']} / {stats['total_codes']}")
            writer.add_scalar(f'Train/Quantizer_{i}/Usage_Mean', stats['mean'], nb_iter)
            writer.add_scalar(f'Train/Quantizer_{i}/Usage_Std', stats['std'], nb_iter)
            writer.add_scalar(f'Train/Quantizer_{i}/Non_Zero_Codes', stats['non_zero_codes'], nb_iter)

        avg_all /= args.print_iter
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter

        # TensorBoard 记录
        writer.add_scalar('Train/Recons', avg_recons, nb_iter)
        writer.add_scalar('Train/Foot_Contact', foot_contact_loss.item(), nb_iter)
        if args.vel_decoder:
            writer.add_scalar('Train/XZ_Vel_Loss', xz_vel_loss.item(), nb_iter)
            writer.add_scalar('Train/XZ_Acc_Loss', xz_acc_loss.item(), nb_iter)
        if args.rot_decoder:
            writer.add_scalar('Train/Rot_Loss', rot_loss.item(), nb_iter)
            writer.add_scalar('Train/Rot_Acc_Loss', rot_acc_loss.item(), nb_iter)

        if avg_all < train_loss_min:
            train_loss_min = avg_all
            print('avg_all < train_loss_min, changing..')
            print(train_loss_min)
            print('saving best loss')
            torch.save({'net': net.state_dict()}, os.path.join(args.out_dir, 'net_best_loss_train.pth'))
            check = os.path.join(args.out_dir, 'net_best_loss_train.pth')
            if os.path.exists(check):
                fid = getfid(check)
                logger.info(f"fid  {fid_min}")
                if fid < fid_min:
                    fid_min = fid
                    logger.info(f"fid update {fid_min}")
                    print('fid min', fid_min)
                    print('loss', avg_all)
                    print('saving best fid_net')
                    torch.save({'net': net.state_dict()}, os.path.join(args.out_dir, 'net_best_fid.pth'))
        
        writer.add_scalar('./Train/L1', avg_all, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        if args.vel_decoder:
            logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t all.  {avg_all:.5f}\t recons. {avg_recons:.5f} \t foot. {foot_contact_loss:.5f} \t xz_vel_loss .{xz_vel_loss:.5f}\t xz_acc_loss .{xz_acc_loss:.5f}")
            print(f"Train. Iter {nb_iter}: \tCommit: {avg_commit:.5f} \tPPL: {avg_perplexity:.2f} \tAll: {avg_all:.5f} \tRecons: {avg_recons:.5f}\t foot. {foot_contact_loss:.5f} \tXZ_Vel_Loss: {xz_vel_loss:.5f} \t XZ_Acc_Loss: {xz_acc_loss:.5f}") 
        elif args.rot_decoder:
            logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t all.  {avg_all:.5f}\t recons.  {avg_recons:.5f} \t foot. {foot_contact_loss:.5f} \t rot_loss .{rot_loss:.5f}\t rot_acc_loss .{rot_acc_loss:.5f}")
            print(f"Train. Iter {nb_iter}: \tCommit: {avg_commit:.5f} \tPPL: {avg_perplexity:.2f} \tAll: {avg_all:.5f} \tRecons: {avg_recons:.5f} \t foot. {foot_contact_loss:.5f} \trot_loss: {rot_loss:.5f} \t rot_acc_loss: {rot_acc_loss:.5f}")        
        else:
            logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t all.  {avg_all:.5f}\t recons.  {avg_recons:.5f} \t ")
            print(f"Train. Iter {nb_iter}: \tCommit: {avg_commit:.5f} \tPPL: {avg_perplexity:.2f} \tAll: {avg_all:.5f} \tRecons: {avg_recons:.5f} \t")                 
        
        avg_all, avg_perplexity, avg_commit, avg_recons = 0., 0., 0., 0.
    
    if nb_iter % args.eval_iter == 0:
        print('begin eval')
        net.eval()
        
        loss_all_eval = []
        loss_commit_all = []
        perplexity_all = []
        with torch.no_grad():
            for i, batch_data in enumerate(eval_loader):
                gt_motion_eval = batch_data
                gt_motion_eval = gt_motion_eval.cuda().float()  # bs, nb_joints, joints_dim, seq_len

                pred_motion_eval, pred_motion_xz, pred_motion_rot, loss_commit_eval, perplexity_eval, _ = net(gt_motion_eval)
                if args.vel_decoder:
                    pred_motion_eval[:, :, 3:5] = pred_motion_xz
                if args.rot_decoder:
                    pred_motion_eval[:, :, [1, 2, *range(68, 194)]] = pred_motion_rot

                loss_motion_eval = Loss(pred_motion_eval, gt_motion_eval)
                loss_all, motion_loss, xz_vel_loss, xz_acc_loss, rot_loss, rot_acc_loss, foot_contact_loss = Loss.forward_all(pred_motion_eval, gt_motion_eval)

                loss_eval = args.commit * loss_commit_eval + loss_all
                loss_all_eval.append(loss_eval.item())
                loss_commit_all.append(loss_commit_eval.item())
                perplexity_all.append(perplexity_eval.item())
        
        loss_eval_2 = sum(loss_all_eval) / len(loss_all_eval)
        loss_commit_avg = sum(loss_commit_all) / len(loss_commit_all)
        perplexity_avg = sum(perplexity_all) / len(perplexity_all)
        
        # TensorBoard 记录（评估阶段）
        writer.add_scalar('Eval/Recons', motion_loss.item(), nb_iter)
        if args.vel_decoder:
            writer.add_scalar('Eval/XZ_Vel_Loss', xz_vel_loss.item(), nb_iter)
            writer.add_scalar('Eval/XZ_Acc_Loss', xz_acc_loss.item(), nb_iter)
        if args.rot_decoder:
            writer.add_scalar('Eval/Rot_Loss', rot_loss.item(), nb_iter)
            writer.add_scalar('Eval/Rot_Acc_Loss', rot_acc_loss.item(), nb_iter)

        if loss_eval_2 < loss_eval_min and avg_all < train_loss_min:
            loss_eval_min = loss_eval_2
            print('loss_eval_2 < loss_eval_min, changing..')
            print(loss_eval_min)
            print('saving best loss')
            torch.save({'net': net.state_dict()}, os.path.join(args.out_dir, 'net_best_loss.pth'))
            check1 = os.path.join(args.out_dir, 'net_best_loss.pth')
        
        writer.add_scalar('./Eval/Loss', loss_eval_2, nb_iter)
        writer.add_scalar('./Eval/Commit', loss_commit_avg, nb_iter)
        writer.add_scalar('./Eval/PPL', perplexity_avg, nb_iter)
        
        print('saving latest')
        torch.save({'net': net.state_dict()}, os.path.join(args.out_dir, 'latest.pth'))

writer.close()  # 关闭 TensorBoard writer