import copy
import sys
from os.path import join as pjoin

from datasets import InterHumanDataset
from datasets.evaluator_models import InterCLIP
from datasets.interclip_rp import InterCLIP_RP
from models import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils.preprocess import load_motion, process_motion_np, rigid_transform

promotiondir = '/data2/lrh/dataset/HHI/InterHuman/motions_processed/'


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
        # mm_idxs = idxs[:]
        print('mm_idxs', len(mm_idxs))

        generated_motions = []
        mm_generated_motions = []
        # Pre-process all target captions
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                name, text, motion1, motion2, motion_lens = data
                batch = {}
                if i in mm_idxs:
                    batch["text"] = list(text) * mm_num_repeats
                else:
                    batch["text"] = list(text)
                batch["motion_lens"] = motion_lens

                batch = self.model.forward_test(batch)
                motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())

                # motions_output[..., :22 * 3] = filters.gaussian_filter1d(motions_output[..., :22 * 3], 1, axis=0, mode='nearest')
                # motions_output[..., 22 * 3:22 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 3:22 * 6], 0.1, axis=0, mode='nearest')
                # motions_output[..., 22 * 6:22 * 6 + 21 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 6:22 * 6 + 21 * 6], 0.5, axis=0, mode='nearest')
                
                savedir = "/data2/lrh/project/hhi/InterGen/checkpoints/evalorigin"
                le_name = 'le_n#' + name[0] + 'person1@None.npy'
                fo_name = 'fo_n#' + name[0] + 'person2@None.npy'
                np.save(os.path.join(savedir, le_name), motions_output[0,:,0])
                np.save(os.path.join(savedir, fo_name), motions_output[0,:,1]) 

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
                            'text': text[0]}
                generated_motions.append(sub_dict)
                if i in mm_idxs:
                    mm_sub_dict = {'mm_motions': motions_output,
                                   'motion_lens': motion_lens[0],
                                    'text': text[0]}
                    mm_generated_motions.append(mm_sub_dict)


        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens


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
        motion_lens = np.array([motion_lens]*mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens


def get_dataset_motion_loader(opt, batch_size):
    opt = copy.deepcopy(opt)
    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.NAME == 'interhuman':
        print('Loading dataset %s ...' % opt.NAME)

        dataset = InterHumanDataset(opt)
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

#     checkpoint = torch.load(pjoin('checkpoints/origin/interclip.ckpt'),map_location="cpu")
#     # checkpoint = torch.load(pjoin('checkpoints/interclip/model/5.ckpt'),map_location="cpu")
#     for k in list(checkpoint["state_dict"].keys()):
#         if "model" in k:
#             checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
#     model.load_state_dict(checkpoint["state_dict"], strict=True)

#     # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
#     return model



def build_models(cfg):
    model = InterCLIP_RP(cfg)
    # swap 258 fea
    checkpoint = torch.load(pjoin('/data2/lrh/project/hhi/InterGen/checkpoints/AInterClip_rp/0301/train/origin_interhuman/swap_fea516_bc128_lr1e-4/model/epoch=349-step=28700.ckpt'),map_location="cpu")
    
    for k in list(checkpoint["state_dict"].keys()):
        if "model." == k[:6]:
            # checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
            checkpoint["state_dict"][k[6:]] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    return model


class EvaluatorModelWrapper(object):

    def __init__(self, cfg, device):

        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()


    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data

            for _ in range(motion2.shape[0]):
                na = name[_]
                file_a = os.path.join(promotiondir, 'person1', na+'.npy')
                file_b = os.path.join(promotiondir, 'person2', na+'.npy')
                promotion1, motion1_swap = load_motion(file_a, 1, swap=False)
                promotion2, motion2_swap = load_motion(file_b, 1, swap=False)
                length = promotion2.shape[0]
                if length > 300:
                    idx = random.choice(list(range(0, length - 300, 1)))
                    gt_length = 300
                    promotion1 = promotion1[idx:idx + gt_length]
                    promotion2 = promotion2[idx:idx + gt_length]
                else:
                    idx = 0
                    gt_length = min(length - idx, 300 )
                    promotion1 = promotion1[idx:idx + gt_length]
                    promotion2 = promotion2[idx:idx + gt_length]
                if np.random.rand() > 0.5:
                    promotion1, promotion2 = promotion2, promotion1
                promotion1, root_quat_init1, root_pos_init1 = process_motion_np(promotion1, 0.001, 0, n_joints=22)
                promotion2, root_quat_init2, root_pos_init2 = process_motion_np(promotion2, 0.001, 0, n_joints=22)
                r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
                angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
                xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
                relative = np.concatenate([angle, xz], axis=-1)[0]
                promotion2 = rigid_transform(relative, promotion2)
                gt_length = len(promotion1)
                if gt_length < 300:
                    padding_len = 300 - gt_length
                    D = promotion1.shape[1]
                    padding_zeros = np.zeros((padding_len, D))
                    promotion1 = np.concatenate((promotion1, padding_zeros), axis=0)
                    promotion2 = np.concatenate((promotion2, padding_zeros), axis=0)
               
                promotion1 = torch.from_numpy(promotion1).to(motion1)
                promotion2 = torch.from_numpy(promotion2).to(motion1)
                if np.random.rand() > 0.5:
                    promotion1, promotion2 = promotion2, promotion1
                motion1_ = promotion1.unsqueeze(0)
                motion2_ = promotion2.unsqueeze(0)

                if _ == 0:
                    motion1 = motion1_
                    motion2 = motion2_
                else:
                    motion1 = torch.cat([motion1, motion1_], dim=0)
                    motion2 = torch.cat([motion2, motion2_], dim=0)
            

            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

            '''Text Encoding'''
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data

            for _ in range(motion2.shape[0]):
                na = name[_]
                file_a = os.path.join(promotiondir, 'person1', na+'.npy')
                file_b = os.path.join(promotiondir, 'person2', na+'.npy')
                promotion1, motion1_swap = load_motion(file_a, 1, swap=False)
                promotion2, motion2_swap = load_motion(file_b, 1, swap=False)
                length = promotion2.shape[0]
                if length > 300:
                    idx = random.choice(list(range(0, length - 300, 1)))
                    gt_length = 300
                    promotion1 = promotion1[idx:idx + gt_length]
                    promotion2 = promotion2[idx:idx + gt_length]
                else:
                    idx = 0
                    gt_length = min(length - idx, 300 )
                    promotion1 = promotion1[idx:idx + gt_length]
                    promotion2 = promotion2[idx:idx + gt_length]
                if np.random.rand() > 0.5:
                    promotion1, promotion2 = promotion2, promotion1
                promotion1, root_quat_init1, root_pos_init1 = process_motion_np(promotion1, 0.001, 0, n_joints=22)
                promotion2, root_quat_init2, root_pos_init2 = process_motion_np(promotion2, 0.001, 0, n_joints=22)
                r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
                angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
                xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
                relative = np.concatenate([angle, xz], axis=-1)[0]
                promotion2 = rigid_transform(relative, promotion2)
                gt_length = len(promotion1)
                if gt_length < 300:
                    padding_len = 300 - gt_length
                    D = promotion1.shape[1]
                    padding_zeros = np.zeros((padding_len, D))
                    promotion1 = np.concatenate((promotion1, padding_zeros), axis=0)
                    promotion2 = np.concatenate((promotion2, padding_zeros), axis=0)
               
                promotion1 = torch.from_numpy(promotion1).to(motion1)
                promotion2 = torch.from_numpy(promotion2).to(motion1)
                if np.random.rand() > 0.5:
                    promotion1, promotion2 = promotion2, promotion1
                motion1_ = promotion1.unsqueeze(0)
                motion2_ = promotion2.unsqueeze(0)


                if _ == 0:
                    motion1 = motion1_
                    motion2 = motion2_
                else:
                    motion1 = torch.cat([motion1, motion1_], dim=0)
                    motion2 = torch.cat([motion2, motion2_], dim=0)

            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding
