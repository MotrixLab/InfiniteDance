import copy
from collections import OrderedDict
from os.path import join as pjoin

import numpy as np
import torch
# from datasets import InterHumanDataset
from RetrievalNet.datasets.audio55mofea264 import AudioMotion_mb
from RetrievalNet.datasets.audiomotion import AudioMotion
# from datasets.evaluator_models import InterCLIP
from RetrievalNet.interclip_rp import InterCLIP_AudioJoints, InterCLIP_RP
from RetrievalNet.models import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class EvaluationDataset(Dataset):
    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        # Pre-process all target captions
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                name, text, motion1, motion2, motion_lens, action = data
                batch = {}
                if i in mm_idxs:
                    batch["text"] = list(text) * mm_num_repeats
                    batch["action"] = list(action) * mm_num_repeats
                else:
                    batch["text"] = list(text)
                    batch["action"] = list(action)
                batch["motion_lens"] = motion_lens

                batch = self.model.forward_test(batch)
                motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())

                # motions_output[..., :22 * 3] = filters.gaussian_filter1d(motions_output[..., :22 * 3], 1, axis=0, mode='nearest')
                # motions_output[..., 22 * 3:22 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 3:22 * 6], 0.1, axis=0, mode='nearest')
                # motions_output[..., 22 * 6:22 * 6 + 21 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 6:22 * 6 + 21 * 6], 0.5, axis=0, mode='nearest')

                B,T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length


                sub_dict = {'motion1': motions_output[0, :,0],
                            'motion2': motions_output[0, :,1],
                            'motion_lens': motion_lens[0],
                            'text': text[0],
                            'action': action[0]}
                generated_motions.append(sub_dict)
                if i in mm_idxs:
                    mm_sub_dict = {'mm_motions': motions_output,
                                'motion_lens': motion_lens[0],
                                    'text': text[0],
                                    'action': action[0]}
                    mm_generated_motions.append(mm_sub_dict)


        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text, action = data['motion1'], data['motion2'], data['motion_lens'], data['text'], data['action']
        return "generated", text, motion1, motion2, motion_lens, action


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        motion_lens = data['motion_lens']
        mm_motions1 = mm_motions[:,:,0]
        mm_motions2 = mm_motions[:,:,1]
        text = data['text']
        action = data['action']
        motion_lens = np.array([motion_lens]*mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens, action


def get_dataset_motion_loader(opt, batch_size):
    opt = copy.deepcopy(opt)
    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.NAME == 'interhuman':
        print('Loading dataset %s ...' % opt.NAME)

        dataset = InterHumanDataset(opt)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    elif opt.NAME == 'interx':
        print('Loading dataset %s ...' % opt.NAME)

        dataset = InterXDataset(opt)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)    
    elif opt.NAME == 'largedance':
        print('Loading dataset %s ...' % opt.NAME)

        dataset = AudioMotion(**opt)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)  
    elif opt.NAME == 'largedance_mb':
        print('Loading dataset %s ...' % opt.NAME)

        dataset = AudioMotion_mb(**opt)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)    
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset




def get_motion_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats):
    # Currently the configurations of two datasets are almost the same
    dataset = EvaluationDataset(model, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader




# def build_models(cfg):
#     model = InterCLIP(cfg)

#     checkpoint = torch.load(pjoin('RetrievalNet/checkpoints/AInterClip_rp/0301/train/origin_interhuman/swap_fea516_bc128_lr1e-4/model/epoch=1999-step=418000.ckpt'),map_location="cpu")
#     # checkpoint = torch.load(pjoin('checkpoints/interclip/model/5.ckpt'),map_location="cpu")
#     for k in list(checkpoint["state_dict"].keys()):
#         if "model" in k:
#             checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
#     model.load_state_dict(checkpoint["state_dict"], strict=True)

#     return model


def build_InterCLIP_RP(cfg):
    model = InterCLIP_RP(cfg)
    # checkpoint = torch.load(pjoin('checkpoints/AInterClip_rp/0301/train/origin_interhuman/swap_fea516_bc128_lr1e-4/model/epoch=1999-step=418000.ckpt'),map_location="cpu")
    # checkpoint = torch.load(pjoin('checkpoints/AInterClip_rp/0423/train/bc512_lr1e-3/model/epoch=49-step=2600.ckpt'),map_location="cpu")
    checkpoint = torch.load(pjoin('checkpoints/AInterClip_rp/0424/all/bc128_lr1e-4/model/epoch=599-step=132600.ckpt'),map_location="cpu")

    for k in list(checkpoint["state_dict"].keys()):
        if "model." == k[:6]:
            # checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
            checkpoint["state_dict"][k[6:]] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    return model




def build_InterCLIP_AudioJoints(cfg):
    model = InterCLIP_AudioJoints(cfg)
    # checkpoint = torch.load(pjoin('/data1/hzy/HumanMotion/RetrievalNet/checkpoints/AInterClip_Audio55Motion264/0512/train/bc256_s100l384_drop0.2_lr1e-4/model/epoch=999-step=241000.ckpt'),map_location="cpu")
    checkpoint = torch.load(pjoin('/data2/hzy/InfiniteDance/All_LargeDanceAR/RetrievalNet/checkpoints/AInterClip_Audio55Motion264/0512/train/bc256_s100l384_drop0.2_lr1e-4/model/epoch=999-step=241000.ckpt'),map_location="cpu")
    # checkpoint = torch.load(pjoin('/data2/hzy/InfiniteDance/All_LargeDanceAR/RetrievalNet/checkpoints/AInterClip_Audio55Motion264/0930/train/bc256_s150l600_drop0.2_lr1e-4/model/epoch=999-step=119000.ckpt'),map_location="cpu")

    for k in list(checkpoint["state_dict"].keys()):
        if "model." == k[:6]:
            # checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
            checkpoint["state_dict"][k[6:]] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    return model


class EvaluatorModelWrapper(object):

    def __init__(self, cfg, device):
        if cfg.NAME == 'InterCLIP_AudioJoints':
            self.model = build_InterCLIP_AudioJoints(cfg)
        elif cfg.NAME == 'InterCLIP_RP':
            self.model = build_InterCLIP_RP(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()


    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data, only="None"):
        with torch.no_grad():
            # breakpoint()
            if self.cfg.NAME == 'InterCLIP_RP':
                seqname, audio, joints, flame = batch_data

                batch = OrderedDict({})
                batch["seqname"] = seqname
                batch["audio"] = audio.to(torch.float32)
                batch["joints"] = joints.to(torch.float32)
                batch["motions"] = flame.to(torch.float32)
            elif self.cfg.NAME == 'InterCLIP_AudioJoints':
                seqname, audio, joints = batch_data

                batch = OrderedDict({})
                batch["seqname"] = seqname
                batch["audio"] = audio.to(torch.float32)
                batch["joints"] = joints.to(torch.float32)

            if only == "None" or only == "motion":
                '''Motion Encoding'''
                motion_embedding = self.model.encode_motion(batch)['motion_emb']

            if only == "None" or only == "audio":
                '''Text Encoding'''
                text_embedding = self.model.encode_cond(batch)['cond_emb']
        if only == "None":
            return text_embedding, motion_embedding
        elif only == "motion":
            return motion_embedding
        elif only == "audio":
            return text_embedding

        

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            if self.cfg.NAME == 'InterCLIP_RP':
                seqname, audio, joints, flame = batch_data

                batch = OrderedDict({})
                batch["seqname"] = seqname
                batch["audio"] = audio.to(torch.float32)
                batch["joints"] = joints.to(torch.float32)
                batch["motions"] = flame.to(torch.float32)
            elif self.cfg.NAME == 'InterCLIP_AudioJoints':
                seqname, audio, joints = batch_data

                batch = OrderedDict({})
                batch["seqname"] = seqname
                batch["audio"] = audio.to(torch.float32)
                batch["joints"] = joints.to(torch.float32)
    
            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding
