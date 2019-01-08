import numpy as np, os, matplotlib.pyplot as plt, sys
os.chdir('/media/karsten_dl/QS2/standard_liverlesion_segmentation/Misc')
sys.path.insert(0,os.getcwd()+'/../Utilities')
import General_Utilities as gu
data_path = '/media/karsten_dl/QS2/standard_liverlesion_segmentation/LOADDATA/Training_Data_2D'
### Image Weightmaps
for i in range(445,473):
    volume, slicev = 'volume-10', 'slice-{}.npy'.format(i)
    vol    = np.load(data_path+'/Volumes'+'/'+volume+'/'+slicev)
    liv    = np.load(data_path+'/LiverMasks'+'/'+volume+'/'+slicev)
    les    = np.load(data_path+'/LesionMasks'+'/'+volume+'/'+slicev)
    b_liv  = np.load(data_path+'/BoundaryMasksLiver'+'/'+volume+'/'+slicev)
    b_les  = np.load(data_path+'/BoundaryMasksLesion'+'/'+volume+'/'+slicev)

    f,ax = plt.subplots(1,5)
    ax[0].imshow(gu.normalize(vol))
    ax[1].imshow(liv, cmap='Greys')
    ax[2].imshow(b_liv.astype(float), cmap='Greys')
    ax[3].imshow(les, cmap='Reds')
    ax[4].imshow(b_les.astype(float), cmap='Reds')
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[2].set_xticks([])
    ax[3].set_xticks([])
    ax[4].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    ax[4].set_yticks([])
    f.set_size_inches(15,3)
    f.tight_layout()
    plt.show()

    f.savefig(os.getcwd()+'/weightmap_example_{}_slice{}.png'.format(volume,i), transparent=True)
    plt.close()
