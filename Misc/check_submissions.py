import numpy as np, os, matplotlib.pyplot as plt, sys
os.chdir('/media/karsten_dl/QS2/standard_liverlesion_segmentation/Misc')
sys.path.insert(0,os.getcwd()+'/../Utilities')
sys.path.insert(0,os.getcwd()+'/../Network_Zoo')
import network_zoo as netlib
import General_Utilities as gu
import nibabel as nib
data_path = '/media/karsten_dl/QS2/standard_liverlesion_segmentation/SAVEDATA/Test_Segmentations/Test_Submissions'
old_data_path = '/media/karsten_dl/QS2/standard_liverlesion_segmentation/SAVEDATA/Test_Segmentations/Old_Test_Submissions'
dp = '/media/karsten_dl/QS2/standard_liverlesion_segmentation/LOADDATA/Test_Data_2D/Volumes'

%gui qt
import pyqtgraph as pg

vol_n = 30
or_vol   = gu.normalize(np.stack([np.load(dp+'/test-volume-{}/'.format(vol_n)+x) for x in sorted(os.listdir(dp+'/test-volume-{}'.format(vol_n)),key=lambda x: int(x.split('-')[-1].split('.')[0]))]))
vol_info = nib.load(data_path+'/test-segmentation-{}.nii'.format(vol_n))
vol      = np.array(vol_info.dataobj)
old_vol_info = nib.load(old_data_path+'/test-segmentation-{}.nii'.format(vol_n))
old_vol      = np.array(vol_info.dataobj)
print('Shape:',vol.shape)
print('Shape:',or_vol.shape)
print('Shape:',old_vol.shape)

vol.shape
pg.image(or_vol+vol.transpose(2,0,1).astype(float))
pg.image(or_vol+old_vol.transpose(2,0,1).astype(float))

import pickle as pkl

network_setup = '/media/karsten_dl/QS2/standard_liverlesion_segmentation/SAVEDATA/Standard_Liver_Networks/vUnet2D_liver_full_equipment_prime'
network_opt = pkl.load(open(network_setup+'/hypa.pkl','rb'))

network     = netlib.Scaffold_UNet(network_opt)
import torch
checkpoint  = torch.load(network_setup+'/checkpoint_best_val.pth.tar')
network.load_state_dict(checkpoint['network_state_dict'])
device = torch.device('cuda')
_ = network.to(device)


from tqdm import trange, tqdm

c = []
or_vol = list(or_vol)
with torch.no_grad():
    for i in trange(len(or_vol)):
        extra_ch  = network_opt.Network['channels']//2
        low_bound = np.clip(i-extra_ch,0,None).astype(int)
        low_diff  = extra_ch-i
        up_bound  = np.clip(i+extra_ch+1,None,len(or_vol)).astype(int)
        up_diff   = i+extra_ch+1-len(or_vol)

        vol_slices = or_vol[low_bound:up_bound]

        if low_diff>0:
            extra_slices    = or_vol[low_bound+1:low_bound+1+low_diff][::-1]
            vol_slices      = extra_slices+vol_slices
        if up_diff>0:
            extra_slices    = or_vol[up_bound-up_diff-1:up_bound-1][::-1]
            vol_slices      = vol_slices+extra_slices

        d_slice = np.stack(vol_slices)
        out = network(torch.from_numpy(np.expand_dims(d_slice,axis=0)).type(torch.FloatTensor).to(device))[0]
        c.append(out.cpu().detach().numpy())

cp = np.stack(c)
pg.image(cp[:,0,1,:])
