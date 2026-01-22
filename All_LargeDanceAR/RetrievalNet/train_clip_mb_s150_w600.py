'''
本文件和train_clip类似，只是针对music55和joints22的retrieval做了配置

'''
import os
import sys

sys.path.append(sys.path[0] + r"/../")
import time
from collections import OrderedDict
from os.path import join as pjoin

import pytorch_lightning as pl
import torch
import torch.optim as optim
from configs import get_config
from datasets import DataModule
from interclip_rp import InterCLIP_AudioJoints, InterCLIP_RP
from models import *
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from utils.utils import print_current_loss

# sys.path.append('/data1/hzy/HumanMotion/utilsds')
from pytorch3d.transforms import matrix_to_axis_angle
# os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from pytorch_lightning.strategies import DDPStrategy

torch.set_float32_matmul_precision('medium')

class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg
        self.mode = cfg.TRAIN.MODE

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model

        self.writer = SummaryWriter(self.log_dir)

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        # scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        scheduler = MultiStepLR(optimizer, milestones=[self.cfg.TRAIN.milestone_epoch], gamma=0.2)

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        seqname, audio, joints = batch_data
        # motion1 = motion1.detach().float()  # .to(self.device)
        # motion2 = motion2.detach().float()  # .to(self.device)

        # B, T = motion1.shape[:2]

        batch = OrderedDict({})
        batch["seqname"] = seqname
        batch["audio"] = audio.to(torch.float32)
        batch["joints"] = joints.to(torch.float32)
        # batch["motions"] = flame.to(torch.float32)

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()


    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()

        return {"loss": loss,
            "loss_logs": loss_logs}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                self.writer.add_scalar(tag, mean_loss[tag], self.it)
            self.logs = OrderedDict()
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_epoch_end(self):
        # pass
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()


    def save(self, file_name):
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return

def build_clip_models(cfg):
    model = InterCLIP(cfg)
    checkpoint = torch.load(pjoin('checkpoints/interclip.ckpt'),map_location="cpu")
    # checkpoint = torch.load(pjoin('checkpoints/interclip/model/5.ckpt'),map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return model

if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    data_cfg      = get_config("configs/largedance/musicbody/datasets45.yaml").largedance
    train_cfg     = get_config("configs/largedance/musicbody/train.yaml")
    evalmodel_cfg = get_config("configs/largedance/musicbody/InterCLIP.yaml")
    print('1')
    if evalmodel_cfg.NAME == 'InterCLIP_AudioJoints':
        model = InterCLIP_AudioJoints(evalmodel_cfg)    
    else:
        raise
        model = InterCLIP_RP(evalmodel_cfg)
    datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    print('2')
    if train_cfg.TRAIN.RESUME:
        ckpt = torch.load(train_cfg.TRAIN.RESUME, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("checkpoint state loaded!")
    litmodel = LitTrainModel(model, train_cfg)
    print('3')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
                                                       save_top_k=-1)
    print('4')
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices=[0], accelerator='gpu',      # 
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=32,
        callbacks=[checkpoint_callback],
    )
    print('5')
    trainer.fit(model=litmodel, datamodule=datamodule)
