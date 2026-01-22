import os

import numpy as np

audio_dir = '/data1/hzy/AllDataset/allmusic_librosa55'
mofea264_dir = '/data1/hzy/HumanMotion/All_mofea/alldata_new_joint_vecs264'

for file in os.listdir(audio_dir):
    mufile = os.path.join(audio_dir, file)
    mofile = os.path.join(mofea264_dir, file)
    if  not os.path.exists(mofile):
        continue
    
    music = np.load(mufile)
    motion = np.load(mofile)
    print('music', music.shape)
    print('motion', motion.shape)