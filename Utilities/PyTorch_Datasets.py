"""========================================================================="""
### LIBRARIES
import os, sys, time, csv, itertools, copy, numpy as np, pandas as pd, itertools as it, torch, copy
import scipy.ndimage.measurements as snm, skimage.transform as st
from tqdm import tqdm, trange

import General_Utilities as gu, Network_Utilities as nu




"""========================================================================="""
### META FUNCTION TO RETURN ADJUSTED DATASETS, e.g. train-val-split --- 2D
def Generate_Required_Datasets(opt, type_flag='base'):
    rng = np.random.RandomState(opt.Training['seed'])
    vol_info = {}
    vol_info['volume_slice_info'] = pd.read_csv(opt.Paths['Training_Path']+'/Assign_2D_Volumes.csv',     header=0)
    vol_info['target_mask_info']  = pd.read_csv(opt.Paths['Training_Path']+'/Assign_2D_LesionMasks.csv', header=0) if opt.Training['data'] == 'lesion' else pd.read_csv(opt.Paths['Training_Path']+'/Assign_2D_LiverMasks.csv', header=0)

    if opt.Training['data']=='lesion':  vol_info['ref_mask_info']     = pd.read_csv(opt.Paths['Training_Path']+'/Assign_2D_LiverMasks.csv', header=0)
    if opt.Training['use_weightmaps']:  vol_info['weight_mask_info']  = pd.read_csv(opt.Paths['Training_Path']+'/Assign_2D_LesionWmaps.csv', header=0) if opt.Training['data'] == 'lesion' else pd.read_csv(opt.Paths['Training_Path']+'/Assign_2D_LiverWmaps.csv', header=0)

    available_volumes = sorted(list(set(np.array(vol_info['volume_slice_info']['Volume']))), key=lambda x: int(x.split('-')[-1]))
    rng.shuffle(available_volumes)

    percentage_data_len = int(len(available_volumes)*opt.Training['perc_data'])
    train_val_split     = int(percentage_data_len*opt.Training['train_val_split'])
    training_volumes    = available_volumes[:percentage_data_len][:train_val_split]
    validation_volumes  = available_volumes[:percentage_data_len][train_val_split:]


    training_dataset   = Basic_Image_Dataset_2D(vol_info, training_volumes, opt)
    validation_dataset = Basic_Image_Dataset_2D(vol_info, validation_volumes, opt, is_validation=True)
    return training_dataset, validation_dataset




"""========================================================================="""
### BASE DATASET CLASS IN 2D
class Basic_Image_Dataset_2D(torch.utils.data.Dataset):
    def __init__(self, vol_info, volumes, opt, is_validation=False):
        self.pars          = opt
        self.vol_info      = vol_info

        self.is_validation = is_validation
        self.rng           = np.random.RandomState(opt.Training['seed'])

        self.available_volumes = volumes
        self.volume_details    = {key:{'Wmap_Paths':[], 'TargetMask_Paths':[], 'Input_Image_Paths':[], 'RefMask_Paths':[], 'Has Target Mask':[], 'Has Ref Mask':[]} for key in volumes}

        self.rvic = opt.Training['Training_ROI_Vicinity']
        self.channel_size = opt.Network['channels']

        self.div_in_volumes = {key:{'Input_Image_Paths':[],'Has Target Mask':[], 'Has Ref Mask':[], 'Wmap_Paths':[], 'RefMask_Paths':[], 'TargetMask_Paths':[]} for key in self.available_volumes}
        for i,vol in enumerate(vol_info['volume_slice_info']['Volume']):
            if vol in self.div_in_volumes.keys():
                self.div_in_volumes[vol]['Input_Image_Paths'].append(vol_info['volume_slice_info']['Slice Path'][i])
                self.div_in_volumes[vol]['Has Target Mask'].append(vol_info['target_mask_info']['Has Mask'][i])
                if opt.Training['use_weightmaps']: self.div_in_volumes[vol]['Wmap_Paths'].append(vol_info['weight_mask_info']['Slice Path'][i])
                self.div_in_volumes[vol]['TargetMask_Paths'].append(vol_info['target_mask_info']['Slice Path'][i])
                if opt.Training['data']=='lesion': self.div_in_volumes[vol]['Has Ref Mask'].append(vol_info['ref_mask_info']['Has Mask'][i])
                if opt.Training['data']=='lesion': self.div_in_volumes[vol]['RefMask_Paths'].append(vol_info['ref_mask_info']['Slice Path'][i])


        self.input_samples = {'Neg':[], 'Pos':[]}
        for i,vol in enumerate(self.div_in_volumes.keys()):
            for j in range(len(self.div_in_volumes[vol]['Input_Image_Paths'])):
                crop_condition = np.sum(self.div_in_volumes[vol]['Has Ref Mask'][int(np.clip(j-self.rvic, 0, None)):j+self.rvic])
                if opt.Training['data']=='liver': crop_condition=True

                if crop_condition:
                    extra_ch  = self.channel_size//2
                    low_bound, low_diff = np.clip(j-extra_ch,0,None).astype(int), extra_ch-j
                    up_bound, up_diff   = np.clip(j+extra_ch+1,None,len(self.div_in_volumes[vol]["Input_Image_Paths"])).astype(int), j+extra_ch+1-len(self.div_in_volumes[vol]["Input_Image_Paths"])

                    vol_slices = self.div_in_volumes[vol]["Input_Image_Paths"][low_bound:up_bound]

                    if low_diff>0:
                        extra_slices    = self.div_in_volumes[vol]["Input_Image_Paths"][low_bound+1:low_bound+1+low_diff][::-1]
                        vol_slices      = extra_slices+vol_slices
                    if up_diff>0:
                        extra_slices    = self.div_in_volumes[vol]["Input_Image_Paths"][up_bound-up_diff-1:up_bound-1][::-1]
                        vol_slices      = vol_slices+extra_slices

                    self.volume_details[vol]['Input_Image_Paths'].append(vol_slices)
                    self.volume_details[vol]['TargetMask_Paths'].append(self.div_in_volumes[vol]['TargetMask_Paths'][j])

                    if opt.Training['data']!='liver':  self.volume_details[vol]['RefMask_Paths'].append(self.div_in_volumes[vol]['RefMask_Paths'][j])
                    if opt.Training['use_weightmaps']: self.volume_details[vol]['Wmap_Paths'].append(self.div_in_volumes[vol]['Wmap_Paths'][j])

                    type_key = 'Pos' if self.div_in_volumes[vol]['Has Target Mask'][j] or self.is_validation else 'Neg'
                    self.input_samples[type_key].append((vol, len(self.volume_details[vol]['Input_Image_Paths'])-1))

        self.n_files  = np.sum([len(self.input_samples[key]) for key in self.input_samples.keys()])
        self.curr_vol = self.input_samples['Pos'][0][0] if len(self.input_samples['Pos']) else self.input_samples['Neg'][0][0]




    def __getitem__(self, idx):
        #Choose a positive example with 50% change if training.
        #During validation, 'Pos' will contain all validation samples.
        #Note that again, volumes without lesions/positive target masks need to be taken into account.
        type_choice = not idx%self.pars.Training['pos_sample_chance'] or self.is_validation
        modes       = list(self.input_samples.keys())
        type_key    = modes[type_choice] if len(self.input_samples[modes[type_choice]]) else modes[not type_choice]

        type_len = len(self.input_samples[type_key])

        next_vol,_ = self.input_samples[type_key][(idx+1)%type_len]
        vol, idx   = self.input_samples[type_key][idx%type_len]

        vol_change = next_vol!=vol
        self.curr_vol   = vol

        intvol = self.volume_details[vol]["Input_Image_Paths"][idx]
        intvol = intvol[len(intvol)//2]

        input_image  = np.concatenate([np.expand_dims(np.load(sub_vol),0) for sub_vol in self.volume_details[vol]["Input_Image_Paths"][idx]],axis=0)
        #Perform data standardization
        if self.pars.Training['no_standardize']:
            input_image  = gu.normalize(input_image, zero_center=False, unit_variance=False, supply_mode="orig")
        else:
            input_image  = gu.normalize(input_image)

        #Lesion/Liver Mask to output
        target_mask = np.load(self.volume_details[vol]["TargetMask_Paths"][idx])
        target_mask = np.expand_dims(target_mask,0)


        #Liver Mask to use for defining training region of interest
        crop_mask = np.expand_dims(np.load(self.volume_details[vol]["RefMask_Paths"][idx]),0) if self.pars.Training['data']=='lesion' else None
        #Weightmask to output
        weightmap = np.expand_dims(np.load(self.volume_details[vol]["Wmap_Paths"][idx]),0).astype(float) if self.pars.Training['use_weightmaps'] else None


        #Generate list of all files that would need to be crop, if cropping is required.
        files_to_crop  = [input_image, target_mask]
        is_mask        = [0,1]
        if weightmap is not None:
            files_to_crop.append(weightmap)
            is_mask.append(0)
        if crop_mask is not None:
            files_to_crop.append(crop_mask)
            is_mask.append(1)

        #First however, augmentation, if required, is performed (i.e. on fullsize images to remove border artefacts in crops).
        if len(self.pars.Training['augment']) and not self.is_validation:
            # Old: copyFiles needs to be True.
            files_to_crop = list(gu.augment_2D(files_to_crop, mode_dict = self.pars.Training['augment'],
                                               seed=self.rng.randint(0,1e8), is_mask = is_mask))

        #If Cropping is required, we crop now.
        if len(self.pars.Training['crop_size']) and not self.is_validation:
            #Add imaginary batch axis in gu.get_crops_per_batch
            crops_for_picked_batch  = gu.get_crops_per_batch(files_to_crop, crop_mask, crop_size=self.pars.Training['crop_size'], seed=self.rng.randint(0,1e8))
            input_image     = crops_for_picked_batch[0]
            target_mask     = crops_for_picked_batch[1]
            weightmap       = crops_for_picked_batch[2] if weightmap is not None else None
            crop_mask       = crops_for_picked_batch[-1] if crop_mask is not None else None


        #If a one-hot encoded target mask is required:
        one_hot_target = gu.numpy_generate_onehot_matrix(target_mask, self.pars.Training['num_classes']) if self.pars.Training['require_one_hot'] else None

        #If we use auxiliary inputs to input additional information into the network, we compute respective outputs here.
        auxiliary_targets, auxiliary_wmaps, one_hot_auxiliary_targets   = None, None, None
        if not self.is_validation and self.pars.Network['use_auxiliary_inputs']:
            auxiliary_targets, auxiliary_wmaps, one_hot_auxiliary_targets   = [], [], []
            for val in range(len(self.pars.Network['structure'])-1):
                aux_level = 2**(val+1)
                aux_img = np.round(st.resize(target_mask,(target_mask.shape[0], target_mask.shape[1]//aux_level,target_mask.shape[2]//aux_level),order=0, mode="reflect", preserve_range=True))
                auxiliary_targets.append(aux_img)
                if self.pars.Training['require_one_hot']:
                    one_hot_auxiliary_targets.append(gu.numpy_generate_onehot_matrix(aux_img, self.pars.Training['num_classes']))
                if weightmap is not None:
                    aux_img = st.resize(weightmap,(weightmap.shape[0], weightmap.shape[1]//aux_level,weightmap.shape[2]//aux_level),order=0, mode="reflect", preserve_range=True)
                    auxiliary_wmaps.append(aux_img)

        #Final Output Dictionary
        return_dict = {"input_images":input_image.astype(float), "targets":target_mask.astype(float),
                       "crop_option":crop_mask.astype(float) if crop_mask is not None else None,
                       "weightmaps":weightmap.astype(float) if weightmap is not None else None,
                       "one_hot_targets":one_hot_target,
                       "aux_targets":auxiliary_targets, "one_hot_aux_targets": one_hot_auxiliary_targets,
                       "aux_weightmaps": auxiliary_wmaps, 'internal_slice_name':intvol, 'vol_change':vol_change}

        return_dict = {key:item for key,item in return_dict.items() if item is not None}
        return return_dict


    def __len__(self):
        return self.n_files
