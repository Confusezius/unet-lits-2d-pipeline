"""
This script contains all functions to aid with loading and reading the LiTS-Dataset.
It includes the Dataset-functions to be loaded into the Pytorch DataLoader.
@author:Karsten Roth - Heidelberg University, 07/11/2017
"""

"""=================================="""
"""====== Load Basic Libraries ======"""
"""=================================="""
import warnings
warnings.filterwarnings("ignore")

import os, sys, time, csv, itertools, copy
import numpy as np, matplotlib, pickle as pkl, nibabel as nib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch

import scipy.ndimage as ndi
import scipy.ndimage.interpolation as sni
import scipy.ndimage.filters       as snf
import scipy.ndimage.measurements  as snm
import skimage.transform as st


from datetime import datetime


"""==================================================="""
"""============== Basic Parameter Values ============="""
"""==================================================="""
MIN_BOUND = -100.0 #Everything below: Water
MAX_BOUND = 400.0  #Everything above corresponds to bones
# Mean/Sd after normalization over full dataset
PIXEL_MEAN = {"orig":0.1021}
PIXEL_STD  = {"orig":0.19177}





"""======================================"""
"""========== Basic Utilities ==========="""
"""======================================"""
def set_bounds(image,MIN_BOUND,MAX_BOUND):
    """
    Clip image to lower bound MIN_BOUND, upper bound MAX_BOUND.
    """
    return np.clip(image,MIN_BOUND,MAX_BOUND)

def normalize(image,use_bd=True,zero_center=True,unit_variance=True,supply_mode="orig"):
    """
    Perform standardization/normalization, i.e. zero_centering and Setting
    the data to unit variance.
    Input Arguments are self-explanatory except for:
    supply_mode: Describes the type of LiTS-Data, i.e. whether it has been
                 rescaled/resized or not. See >Basic_Parameter_Values<
    """
    if not use_bd:
        MIN_BOUND = np.min(image)
        MAX_BOUND = np.max(image)
    else:
        MIN_BOUND = -100.0 #Everything below: Water
        MAX_BOUND = 400.0
        image = set_bounds(image,MIN_BOUND,MAX_BOUND)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image = np.clip(image,0.,1.)
    if zero_center:
        image = image - PIXEL_MEAN[supply_mode]
    if unit_variance:
        image = image/PIXEL_STD[supply_mode]
    return image

def torch_generate_onehot_matrix(matrix_mask, data_shape):
    """
    Function to convert a mask array of shape BS,W,H(,D) with values
    in 0...C-1 to an array of shape BS,C,W,H(,D). Works with torch tensors.

    Arguments:
        matrix_mask:    Mask to convert.
        data_shape:     Reference shape the array should be converted to.
    """
    bs,n_dim = data_shape[:2]
    onehot_matrix = torch.zeros(data_shape).view(-1,n_dim).scatter_(1,matrix_mask.view(-1,1).type(torch.LongTensor),1)
    onehot_matrix = onehot_matrix.view(bs,int(np.prod(data_shape[2:])),n_dim).transpose(1,2).contiguous().view(*data_shape)
    return onehot_matrix

def numpy_generate_onehot_matrix(matrix_mask, ndim):
    """
    Function to convert a mask array of shape W,H(,D) with values
    in 0...C-1 to an array of shape C,W,H(,D). Works with numpy arrays.

    Arguments:
        matrix_mask:    Mask to convert.
        ndim:           Number of additional one-hot dimensions.
    """
    onehot_matrix = np.eye(ndim)[matrix_mask.reshape(-1).astype('int')].astype('int')
    data_shape    = list(matrix_mask.shape)
    data_shape[0] = ndim
    onehot_matrix = np.fliplr(np.flipud(onehot_matrix).T).reshape(*data_shape)
    return onehot_matrix



"""======================================"""
"""============ Augmentation ============"""
"""======================================"""
##############################################################################################
def rotate_2D(to_aug, rng=np.random.RandomState(1)):
    """
    Perform standard 2D-per-slice image rotation.
    Arguments:
    to_aug:     List of files that should be deformed in the same way. Each element
                must be of standard Torch_Tensor shape: (C,W,H,...).
                Deformation is done equally for each channel, but differently for
                each image in a batch if N!=1.
    rng:        Random Number Generator that can be provided for the Gaussian filter means.
    copy_files: If True, copies the input files before transforming. Ensures that the actual
                input data remains untouched. Otherwise, it is directly altered.

    Function only returns data when copy_files==True.
    """
    angle = (rng.rand()*2-1)*10
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            #actually perform rotation
            aug_file[ch,:]    = ndi.interpolation.rotate(aug_file[ch,:].astype(np.float32), angle, reshape=False, order=0, mode="nearest")
    return to_aug, angle


##############################################################################################
def zoom_2D(to_aug, rng=np.random.RandomState(1)):
    """
    Perform standard 2D per-slice zooming/rescaling.
    Arguments:
    to_aug:     List of files that should be deformed in the same way. Each element
                must be of standard Torch_Tensor shape: (N,C,W,H,...).
                Deformation is done equally for each channel, but differently for
                each image in a batch if N!=1.
    rng:        Random Number Generator that can be provided for the Gaussian filter means.
    copy_files: If True, copies the input files before transforming. Ensures that the actual
                input data remains untouched. Otherwise, it is directly altered.

    Function only returns data when copy_files==True.
    Note: Should also work for 3D, but has not been tested for that.
    """
    magnif         = rng.uniform(0.825,1.175)
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            sub_img     = aug_file[ch,:]
            # sub_mask    = aug_file[ch,:]
            img_shape   = np.array(sub_img.shape)
            new_shape   = [int(np.round(magnif*shape_val)) for shape_val in img_shape]
            zoomed_shape= (magnif,)*(sub_img.ndim)

            if magnif<1:
                how_much_to_clip    = [(x-y)//2 for x,y in zip(img_shape, new_shape)]
                idx_cornerpix       = tuple(-1 for _ in range(sub_img.ndim))
                idx_zoom            = tuple(slice(x,x+y) for x,y in zip(how_much_to_clip,new_shape))
                zoomed_out_img      = np.ones_like(sub_img)*sub_img[idx_cornerpix]
                zoomed_out_img[idx_zoom] = ndi.interpolation.zoom(sub_img.astype(np.float32),zoomed_shape,order=0,mode="nearest")
                aug_file[ch,:]        = zoomed_out_img

            if magnif>1:
                zoomed_in_img       = ndi.interpolation.zoom(sub_img.astype(np.float32),zoomed_shape,order=0,mode="nearest")
                rounding_correction = [(x-y)//2 for x,y in zip(zoomed_in_img.shape,img_shape)]
                rc_idx              = tuple(slice(x,x+y) for x,y in zip(rounding_correction, img_shape))
                aug_file[ch,:]   = zoomed_in_img[rc_idx]

    return to_aug


##############################################################################################
def hflip_2D(to_aug, rng=np.random.RandomState(1)):
    """
    Perform standard 2D per-slice horizontal_flipping.
    Arguments:
    to_aug:     List of files that should be deformed in the same way. Each element
                must be of standard Torch_Tensor shape: (N,C,W,H,...).
                Deformation is done equally for each channel, but differently for
                each image in a batch if N!=1.
    rng:        Random Number Generator that can be provided for the Gaussian filter means.
    copy_files: If True, copies the input files before transforming. Ensures that the actual
                input data remains untouched. Otherwise, it is directly altered.

    Function only returns data when copy_files==True.
    Note: Should also work for 3D, but has not been tested for that.
    """
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            aug_file[ch,:]  = np.fliplr(aug_file[ch,:])

    return to_aug


##############################################################################################
def vflip_2D(to_aug, rng=np.random.RandomState(1)):
    """
    Perform standard 2D per-slice vertical flipping.
    Arguments:
    to_aug:     List of files that should be deformed in the same way. Each element
                must be of standard Torch_Tensor shape: (N,C,W,H,...).
                Deformation is done equally for each channel, but differently for
                each image in a batch if N!=1.
    rng:        Random Number Generator that can be provided for the Gaussian filter means.
    copy_files: If True, copies the input files before transforming. Ensures that the actual
                input data remains untouched. Otherwise, it is directly altered.

    Function only returns data when copy_files==True.
    Note: Should also work for 3D, but has not been tested for that.
    """
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            aug_file[ch,:]  = np.flipud(aug_file[ch,:])

    return to_aug


##############################################################################################
def augment_2D(to_aug, mode_dict=["rot","zoom"], copy_files=False, return_files=False, seed=1, is_mask=[0,1,0]):
    """
    Combine all augmentation methods to perform data augmentation (in 2D). Selection is done randomly.
    Arguments:
    to_aug:     List of files that should be deformed in the same way. Each element is a list with
                Arrays of standard Torch_Tensor shape: (C,W,H,...).
                Augmentation is done equally for each channel, but differently for
                each image in a batch if N!=1.
    mode_dict:  List of augmentation methods that should be used.
    rng:        Random Number Generator that can be provided for the Gaussian filter means.
    copy_files: If True, copies the input files before transforming. Ensures that the actual
                input data remains untouched. Otherwise, it is directly altered.

    Function only returns data when copy_files==True.
    """
    rng = np.random.RandomState(seed)
    modes = []

    if rng.randint(2) and "rot" in mode_dict:
        modes.append('rot')
        to_aug, rotation_angle = rotate_2D(to_aug,rng)
    if rng.randint(2) and "zoom" in mode_dict:
        modes.append('zoom')
        to_aug = zoom_2D(to_aug,rng)
    if rng.randint(2) and "hflip" in mode_dict:
        modes.append('hflip')
        to_aug = hflip_2D(to_aug,rng)
    if rng.randint(2) and "vflip" in mode_dict:
        modes.append('vflip')
        to_aug = vflip_2D(to_aug,rng)

    return to_aug







"""================================================="""
"""============ Cropping for DataLoader ============"""
"""================================================="""
def get_crops_per_batch(batch_to_crop, idx_batch=None, crop_size=[128,128], n_crops=1, seed=1):
    """
    Function to crop from input images.
    Takes as input a list of same-shaped 3D/4D-arrays with Ch,W,H(,D). If an index-file
    is supplied, crops will only be taken in and around clusters in the index file. If the index-file
    contains no clusters, then a random crop will be taken.

    Arguments:
    batch_to_crop:      list of batches that need to be cropped. Note that cropping is performed independently for
                        each image of a batch.
    idx_batch:          Batch of same size as input batches. Contains either clusters (i.e. ones) from which a
                        cluster-center will be sampled or None. In this case, the center will be randomly selected.
                        If not None, prov_coords must be None. The idx_image should ahve shape (1,W,H).
    prov_coords:        If we have precomputed indices where we simply want to crop around, pass with prov_coords-argument.
                        In this case, idx_batch must be None! When passed, prov_coords should be a list of lists/arrays containing
                        the coordinate suggestions and should be of length batch_size!
                        It is assumed that all cooridnates are already adjusted to viable ranges per volume.
    crop_size:          Size of the crops to take -> len(crop_size) = input_batch.ndim-1, i.e. ignore batchdimension.
    n_crops:            Number of crops to take per image. Ensure that this coincides with your chosen batchsize during training.
    """
    rng = np.random.RandomState(seed)

    # assert (idx_batch is not None and prov_coords is None) or \
    #        (idx_batch is None and prov_coords is not None) or \
    #        (idx_batch is None and prov_coords is None), "Error when passing arguments for idx_batch and/or prov_coords!"
    #
    # assert all((np.array(batch_to_crop[0].shape[-len(crop_size):])-np.array(crop_size))>0), "Crop size chosen to be bigger than volume!"

    sup                     = list(1-np.array(crop_size)%2)
    bl_len                  = len(batch_to_crop)
    batch_list_to_return    = []

    ### Provide idx-list
    batch_list_to_return_temp   = [[] for i in range(len(batch_to_crop))]

    if idx_batch is not None:
        all_crop_idxs           = np.where(idx_batch[0,:]==1) if np.sum(idx_batch[0,:])!=0 else [[]]
    else:
        all_crop_idxs           = [[]]

    if len(all_crop_idxs[0])>0:
        if idx_batch is not None:
            crop_idx = [np.clip(rng.choice(ax),crop_size[i]//2-1,batch_to_crop[0][:].shape[i+1]-crop_size[i]//2-1) for i,ax in enumerate(all_crop_idxs)]
    else:
        crop_idx = [rng.randint(crop_size[i]//2-1,np.array(batch_to_crop[0].shape[i+1])-crop_size[i]//2-1) for i in range(batch_to_crop[0].ndim-1)]
    # if prov_coords is not None:
    # slice_list = [slice(0,None)]+[slice(center-crop_size[i]//2+mv,center+crop_size[i]//2+1) for i,(center,mv) in enumerate(zip(crop_idx,sup))]
    # else:
    slice_list = [slice(0,None)]+[slice(center-crop_size[i]//2+mv,center+crop_size[i]//2+1) for i,(center,mv) in enumerate(zip(list(crop_idx),sup))]

    for i in range(bl_len):
        batch_list_to_return.append(batch_to_crop[i][slice_list])

    return tuple(batch_list_to_return)




"""======================================"""
"""============ Visualisation ==========="""
"""======================================"""
def progress_plotter(x, train_loss, train_metric, val_metric=None, savename='result.svg', title='No title'):
    plt.style.use('ggplot')

    f,ax = plt.subplots(1)
    ax.plot(x, train_loss,'b--',label='Training Loss')
    axt = ax.twinx()
    axt.plot(x, train_metric, 'b', label='Training Dice')

    if val_metric is not None: axt.plot(x, val_metric, 'r', label='Validation Dice')

    ax.set_title(title)
    ax.legend(loc=0)
    axt.legend(loc=2)

    f.suptitle('Metrics')
    f.set_size_inches(15,10)
    f.savefig(savename)

    plt.close()



"""=============================="""
"""========== Loggers ==========="""
"""=============================="""
class CSVlogger():
    def __init__(self, logname, header_names):
        self.header_names = header_names
        self.logname      = logname
        with open(logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(header_names)
    def write(self, inputs):
        with open(self.logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(inputs)




"""============================================"""
"""=== Setup Save Folder And Summary Files ===="""
"""============================================"""
def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str


def logging_setup(opt):
    #Generate Save Directory Name with Timestamp
    save_dir_name = opt.Network['Network_name']
    if not opt.no_date:
        dt = datetime.now()
        dt = '{}-{}-{}-{}-{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        save_dir_name += '_SetupIter-'+str(opt.iter_idx)+'_Date-'+dt
        save_dir_name += '_'+opt.Training['savename'] if len(opt.Training['savename']) else ''
    else:
        save_dir_name += '_'+opt.Training['savename'] if len(opt.Training['savename']) else ''

    save_path     = opt.Paths['Save_Path']+"/"+save_dir_name

    #Check if a folder with that name exists. If so, append
    #an index to distinguish.
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        count = 1
        while os.path.exists(save_path):
            count       += 1
            svn          = save_dir_name+"__V"+str(count)
            save_path    = opt.Paths['Save_Path']+"/"+svn
        save_dir_name = svn
        os.makedirs(save_path)
    opt.Paths['Save_Path'] = save_path
    opt.Training['Save_Dir_Name'] = save_dir_name

    #Save setup parameters to text-file and pickle
    with open(save_path+'/Parameter_Info.txt','w') as f:
        f.write(gimme_save_string(opt))
    pkl.dump(opt,open(save_path+"/hypa.pkl","wb"))






"""==============================================="""
"""========= PLOT EXAMPLE SEGMENTATIONS =========="""
"""==============================================="""
import random
def generate_example_plots_2D(net, t_dataset, v_dataset, opt, has_crop=True, name_append="end_of_epoch", seeds=[0,6], n_plots=20, is_hnm=False):
    _ = net.eval()

    for i,seed in enumerate(seeds):
        random.seed(seed)
        np.random.seed(seed)

        ### Get Random Choice of Example Slices
        t_slices, t_crops, t_gts, t_preds = [],[],[],[None]*n_plots
        v_slices, v_crops, v_gts, v_preds = [],[],[],[None]*n_plots

        for _ in range(n_plots):
            coin = np.random.randint(0,3)
            vol, idx = random.choice(t_dataset.input_samples['Neg']) if coin==0 or not len(t_dataset.input_samples['Pos']) else random.choice(t_dataset.input_samples['Pos'])


            t_slices.append(t_dataset.volume_details[vol]['Input_Image_Paths'][idx])
            if has_crop: t_crops.append(t_dataset.volume_details[vol]['RefMask_Paths'][idx])
            t_gts.append(t_dataset.volume_details[vol]['TargetMask_Paths'][idx])

            vol, idx = random.choice(v_dataset.input_samples['Pos'])
            v_slices.append(v_dataset.volume_details[vol]['Input_Image_Paths'][idx])
            if has_crop: v_crops.append(v_dataset.volume_details[vol]['RefMask_Paths'][idx])
            v_gts.append(v_dataset.volume_details[vol]['TargetMask_Paths'][idx])


        ### Compute Example Segmentations on Slices of Training Dataset
        for j,(sub_slices,sub_gt) in enumerate(zip(t_slices,t_gts)):
            if opt.Training['no_standardize']:
                net_input = normalize(np.concatenate([np.expand_dims(np.expand_dims(np.load(sub_slice),0),0) for sub_slice in sub_slices],axis=1), unit_variance=False, zero_center=False)
                t_slices[j] = normalize(np.load(sub_slices[len(sub_slices)//2]), unit_variance=False, zero_center=False)
            else:
                net_input = normalize(np.concatenate([np.expand_dims(np.expand_dims(np.load(sub_slice),0),0) for sub_slice in sub_slices],axis=1))
                t_slices[j] = normalize(np.load(sub_slices[len(sub_slices)//2]))
            net_input = torch.from_numpy(net_input).type(torch.FloatTensor).to(opt.device)

            pred = net(net_input)[0].data.cpu().squeeze(0).numpy()
            init_pred_size = pred.shape[0]
            pred = np.round(pred)[0,:] if pred.shape[0]==1 else np.argmax(pred,axis=0)
            # pred = np.round(pred)[0,:] if pred.shape[0]==1 else np.argmax(pred,axis=0)
            if is_hnm: pred = pred%4>(0+int(init_pred_size>2))

            t_preds[j]  = pred if not has_crop else pred*np.load(t_crops[j])
            t_gts[j]    = np.load(sub_gt)
            if is_hnm: t_gts[j] = t_gts[j]%4>(0+int(init_pred_size>2))


        ### Compute Example Segmentations on Slices of Validation Dataset
        for j,(sub_slices,sub_gt) in enumerate(zip(v_slices,v_gts)):
            if opt.Training['no_standardize']:
                net_input = normalize(np.concatenate([np.expand_dims(np.expand_dims(np.load(sub_slice),0),0) for sub_slice in sub_slices],axis=1), unit_variance=False, zero_center=False)
                v_slices[j] = normalize(np.load(sub_slices[len(sub_slices)//2]), unit_variance=False, zero_center=False)
            else:
                net_input = normalize(np.concatenate([np.expand_dims(np.expand_dims(np.load(sub_slice),0),0) for sub_slice in sub_slices],axis=1))
                v_slices[j] = normalize(np.load(sub_slices[len(sub_slices)//2]))
            net_input = torch.from_numpy(net_input).type(torch.FloatTensor).to(opt.device)

            pred = net(net_input)[0].data.cpu().squeeze(0).numpy()
            init_pred_size = pred.shape[0]
            pred = np.round(pred)[0,:] if pred.shape[0]==1 else np.argmax(pred,axis=0)
            if is_hnm: pred = pred%4>(0+int(init_pred_size>2))

            v_preds[j]  = pred if not has_crop else pred*np.load(v_crops[j])
            v_gts[j]     = np.load(sub_gt)


        ### Generate Plots
        # Training
        f,ax = plt.subplots(10,n_plots//10*3)
        axs = ax.reshape(-1)
        for idx in range(0,len(axs),3):
            axs[idx].imshow(t_slices[idx//3])
            axs[idx+1].imshow(t_gts[(idx+1)//3])
            axs[idx+2].imshow(t_preds[(idx+2)//3],vmin=0,vmax=1)
        f.set_size_inches(15,20)
        f.tight_layout()
        f.savefig(opt.Paths['Save_Path']+'/training_samples_'+name_append+'_'+str(i+1)+'.svg')
        plt.close()
        # Validation
        f,ax = plt.subplots(10,n_plots//10*3)
        axs = ax.reshape(-1)
        for idx in range(0,len(axs),3):
            axs[idx].imshow(v_slices[idx//3])
            axs[idx+1].imshow(v_gts[(idx+1)//3])
            axs[idx+2].imshow(v_preds[(idx+2)//3],vmin=0,vmax=1)
        f.set_size_inches(15,20)
        f.tight_layout()
        f.savefig(opt.Paths['Save_Path']+'/validation_samples_'+name_append+'_'+str(i+1)+'.svg')
        plt.close()




"""================================================="""
"""===== Read Parameters From TxT to Namespace ====="""
"""================================================="""
import pandas as pd, itertools as it, ast
### Function to extract setup info from text file ###
def extract_setup_info(opt):
    """
    Structure information for network_base_setup_file and network_variation_setup_file:
        [1] network_base_setup_file:
                Comments: %,=
                Dict-Entries: #+Name
                Entries into resp. dicts: key+':'+items
        [2] network_variation_setup_file:
                Comments: %
                Sub-Gridsearches: Use = to divide
                Dict-Entries: #+Name
                Gridsearches on above parameter: key+':'+[var_1, ..., var_n]
    """

    baseline_setup = pd.read_table(opt.base_setup, header=None)
    baseline_setup = [x for x in baseline_setup[0] if '%' not in x and '=' not in x]
    sub_setups     = [x.split('#')[-1].replace(' ','') for x in np.array(baseline_setup) if '#' in x]
    vals           = [x for x in np.array(baseline_setup)]
    set_idxs       = [i for i,x in enumerate(np.array(baseline_setup)) if '#' in x]+[len(vals)]
    settings = {}
    for i in range(len(set_idxs)-1):
        settings[sub_setups[i]] = [[y.replace(" ","") for y in x.split(':')] for x in vals[set_idxs[i]+1:set_idxs[i+1]]]

    d_opt = vars(opt)
    for key in settings.keys():
        d_opt[key] = {subkey:ast.literal_eval(x) for subkey,x in settings[key]}

    d_opt['iter_idx'] = 0
    if opt.search_setup == '':
        return [opt]


    variation_setup = pd.read_table(opt.search_setup, header=None)
    variation_setup = [x for x in variation_setup[0] if '%' not in x]
    sub_grid_div_idxs = [i for i,x in enumerate(variation_setup) if '=' in x]+[len(variation_setup)]

    sub_grid_searches = [variation_setup[sub_grid_div_idxs[i]+1:sub_grid_div_idxs[i+1]] for i in range(len(sub_grid_div_idxs)-1)]
    setup_collection = []
    for variation_setup in sub_grid_searches:
        sub_setups      = [x.split('#')[-1].replace(' ','') for x in np.array(variation_setup) if '#' in x]
        vals            = [x for x in np.array(variation_setup)]
        set_idxs        = [i for i,x in enumerate(np.array(variation_setup)) if '#' in x]+[len(vals)]
        settings = {}
        for i in range(len(set_idxs)-1):
            settings[sub_setups[i]] = []
            for x in vals[set_idxs[i]+1:set_idxs[i+1]]:
                y = x.split(':')
                settings[sub_setups[i]].append([[y[0].replace(" ","")], ast.literal_eval(y[1].replace(" ",""))])
            settings

        all_c = []
        for key in settings.keys():
            sub_c = []
            for s_i in range(len(settings[key])):
                sub_c.append([[key]+list(x) for x in list(it.product(*settings[key][s_i]))])
            all_c.extend(sub_c)


        training_options = list(it.product(*all_c))
        for i,variation in enumerate(training_options):
            base_opt   = copy.deepcopy(opt)
            base_d_opt = vars(base_opt)
            for sub_variation in variation:
                # if sub_variation[0] not in base_d_opt.keys(): base_d_opt[sub_variation[0]] = {}
                base_d_opt[sub_variation[0]][sub_variation[1]] = sub_variation[2]
            base_d_opt['iter_idx'] = i
            setup_collection.append(base_opt)


    return setup_collection



"""================================"""
"""===== Create Network Graph ====="""
"""================================"""
def save_graph(network_output, savepath, savename, view=False):
    from graphviz import Digraph
    print('Creating Graph... ', end='')
    def make_dot(var, savename, params=None):
        """
        Generate a symbolic representation of the network graph.
        """
        if params is not None:
            assert all(isinstance(p, Variable) for p in params.values())
            param_map = {id(v): k for k, v in params.items()}

        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='6',
                         ranksep='0.1',
                         height='0.6',
                         width='1')
        dot  = Digraph(node_attr=node_attr, format='svg', graph_attr=dict(size="40,10", rankdir='LR', rank='same'))
        seen = set()

        def size_to_str(size):
            return '('+(', ').join(['%d' % v for v in size])+')'

        def add_nodes(var):
            replacements  = ['Backward', 'Th', 'Cudnn']
            color_assigns = {'Convolution':'orange',
                             'ConvolutionTranspose': 'lightblue',
                             'Add': 'red',
                             'Cat': 'green',
                             'Softmax': 'yellow',
                             'Sigmoid': 'yellow',
                             'Copys':   'yellow'}
            if var not in seen:
                op1 = torch.is_tensor(var)
                op2 = not torch.is_tensor(var) and str(type(var).__name__)!='AccumulateGrad'

                text = str(type(var).__name__)
                for rep in replacements:
                    text = text.replace(rep, '')
                color = color_assigns[text] if text in color_assigns.keys() else 'gray'

                if 'Pool' in text: color = 'lightblue'

                if op1 or op2:
                    if hasattr(var, 'next_functions'):
                        count = 0
                        for i, u in enumerate(var.next_functions):
                            if str(type(u[0]).__name__)=='AccumulateGrad':
                                if count==0: attr_text = '\nParameter Sizes:\n'
                                attr_text += size_to_str(u[0].variable.size())
                                count += 1
                                attr_text += ' '
                        if count>0: text += attr_text


                if op1:
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
                if op2:
                    dot.node(str(id(var)), text, fillcolor=color)

                seen.add(var)

                if op1 or op2:
                    if hasattr(var, 'next_functions'):
                        for u in var.next_functions:
                            if u[0] is not None:
                                if str(type(u[0]).__name__)!='AccumulateGrad':
                                    dot.edge(str(id(u[0])), str(id(var)))
                                    add_nodes(u[0])
                    if hasattr(var, 'saved_tensors'):
                        for t in var.saved_tensors:
                            dot.edge(str(id(t)), str(id(var)))
                            add_nodes(t)

        add_nodes(var.grad_fn)
        dot.save(savename)
        return dot

    if not os.path.exists(savepath+"/Network_Graphs"):
        os.makedirs(savepath+"/Network_Graphs")
    viz_graph = make_dot(network_output, savepath+"/Network_Graphs"+"/"+savename)
    print('Done.')
    if view: viz_graph.view()
