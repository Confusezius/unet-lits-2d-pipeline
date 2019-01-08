import numpy as np, os, matplotlib.pyplot as plt, sys
os.chdir('/media/karsten_dl/QS2/standard_liverlesion_segmentation/Misc')
sys.path.insert(0,os.getcwd()+'/../Utilities')
import General_Utilities as gu
import nibabel as nib
data_path = '/media/karsten_dl/QS2/standard_liverlesion_segmentation/SAVEDATA/Test_Segmentations/Test_Submissions'

vol_info = nib.load(data_path+'/test-segmentation-40.nii')
vol      = np.array(vol_info.dataobj)
print('Shape:',vol.shape)
