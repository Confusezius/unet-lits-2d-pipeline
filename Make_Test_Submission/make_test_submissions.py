# @author: Karsten Roth - Heidelberg University, 07/11/2017


"""======================================="""
"""========= TEST DATASET ================"""
"""======================================="""
import torch.utils.data as data
class TestDataset(data.Dataset):
    def __init__(self, test_data_folder, path_2_liver_segmentations, opt, channel_size=1):
        self.pars = opt
        self.test_volumes       = [x for x in os.listdir(test_data_folder+'/Volumes')]
        self.test_volume_slices = {key:[test_data_folder+'/Volumes/'+key+'/'+x for x in sorted(os.listdir(test_data_folder+'/Volumes/'+key),key=lambda x: [int(y) for y in x.split('-')[-1].split('.')[0].split('_')])] for key in self.test_volumes}
        self.test_liver_slices  = {key:[path_2_liver_segmentations+'/'+key+'/'+x for x in sorted(os.listdir(path_2_liver_segmentations+'/'+key),key=lambda x: [int(y) for y in x.split('-')[-1].split('.')[0].split('_')])] for key in self.test_volumes}

        self.channel_size   = channel_size
        self.liver_seg_info = pkl.load(open(path_2_liver_segmentations+'/liver_bound_dict.pkl','rb'))
        self.recon_info     = pkl.load(open(test_data_folder+'/volume_nii_info.pkl','rb'))

        self.iter_data, slice_cluster_collect, liv_cluster_collect = [],[],[]
        for vol in self.test_volumes:
            for i in range(len(self.test_volume_slices[vol])):
                extra_ch  = self.channel_size//2
                low_bound = np.clip(i-extra_ch,0,None).astype(int)
                low_diff  = extra_ch-i
                up_bound  = np.clip(i+extra_ch+1,None,len(self.test_volume_slices[vol])).astype(int)
                up_diff   = i+extra_ch+1-len(self.test_volume_slices[vol])

                vol_slices = self.test_volume_slices[vol][low_bound:up_bound]
                liv_slices = self.test_liver_slices[vol][low_bound:up_bound]

                if low_diff>0:
                    extra_slices    = self.test_volume_slices[vol][low_bound+1:low_bound+1+low_diff][::-1]
                    vol_slices      = extra_slices+vol_slices
                    extra_slices    = self.test_liver_slices[vol][low_bound+1:low_bound+1+low_diff][::-1]
                    liv_slices      = extra_slices+liv_slices
                if up_diff>0:
                    extra_slices    = self.test_volume_slices[vol][up_bound-up_diff-1:up_bound-1][::-1]
                    vol_slices      = vol_slices+extra_slices
                    extra_slices    = self.test_liver_slices[vol][up_bound-up_diff-1:up_bound-1][::-1]
                    liv_slices      = liv_slices+extra_slices

                slice_cluster_collect.append(vol_slices)
                liv_cluster_collect.append(liv_slices)
                self.iter_data.append((vol,i))

            self.test_volume_slices[vol] = slice_cluster_collect
            self.test_liver_slices[vol]  = liv_cluster_collect

            slice_cluster_collect, liv_cluster_collect = [],[]


        self.n_files = len(self.iter_data)
        self.vol_slice_idx = 0
        self.curr_vol = 0
        self.curr_vol_name = self.test_volumes[0]
        self.curr_vol_size = self.liver_seg_info[self.test_volumes[self.curr_vol]+'.nii'][2]

        self.total_num_slices = len(self.test_volume_slices[self.curr_vol_name])

    def __getitem__(self, idx):
        VOI, SOI = self.iter_data[idx]
        V2O  = np.concatenate([np.expand_dims(np.load(vol),0) for vol in self.test_volume_slices[VOI][SOI]],axis=0)

        if self.pars.Training['no_standardize']:
            V2O  = gu.normalize(V2O, zero_center=False, unit_variance=False, supply_mode="orig")
        else:
            V2O  = gu.normalize(V2O)

        LMSK  = np.concatenate([np.expand_dims(np.load(vol),0) for vol in self.test_liver_slices[VOI][SOI]],axis=0)

        if self.vol_slice_idx==len(self.test_volume_slices[self.test_volumes[self.curr_vol]])-1:
            self.vol_slice_idx    = 0
            self.curr_vol_name    = self.test_volumes[self.curr_vol]
            self.total_num_slices = len(self.test_volume_slices[self.curr_vol_name])
            self.curr_vol     += 1
            return_data = {'VolSlice':V2O,'end_volume':True}
        else:
            return_data = {'VolSlice':V2O,'end_volume':False}
            self.vol_slice_idx+=1
        return_data['LivMsk'] = LMSK

        return return_data

    def __len__(self):
        return self.n_files





"""======================================="""
"""============= MAIN FUNCTIONALITY ======"""
"""======================================="""
def main(opt):
    ############ Init Network ###################
    network_list = []
    for i,network_setup in enumerate(opt.networks_to_use):
        network_opt = pkl.load(open(network_setup+'/hypa.pkl','rb'))
        network     = netlib.Scaffold_UNet(network_opt)
        checkpoint  = torch.load(network_setup+'/checkpoint_best_val.pth.tar')
        network.load_state_dict(checkpoint['network_state_dict'])
        network_list.append({'network':network, 'settings':network_opt})
        del network, network_opt, checkpoint

    back_device = torch.device('cpu')
    up_device   = torch.device('cuda')



    ############# Set Dataloader ##################
    max_channels    = np.max([item['settings'].Network['channels'] for item in network_list])
    test_dataset    = TestDataset(opt.test_data, opt.path_2_liv_seg, network_list[0]['settings'], channel_size=max_channels)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

    input_slices, vol_segs, volume_info, liver_masks= [],[],{},[]
    n_vols = len(test_dataloader.dataset.test_volumes)
    vol_count = 1

    data_iter = tqdm(test_dataloader, position=0)


    ############# Run Test Mask Generation ##################
    for idx,data in enumerate(data_iter):
        ### Getting input slices ###
        data_iter.set_description('Reading... [Vol {}/{}]'.format(vol_count, n_vols))
        input_slices.append(data['VolSlice'])
        liver_masks.append(data['LivMsk'].numpy()[0,0,:])
        end_volume = data['end_volume'].numpy()[0]

        if end_volume:
            prev_vol     = test_dataloader.dataset.test_volumes[test_dataloader.dataset.curr_vol-1]
            liver_region = test_dataloader.dataset.liver_seg_info[prev_vol+'.nii'][2]

            with torch.no_grad():
                out_mask     = np.zeros((512,512,len(input_slices)))

                for net_count,net_dict in enumerate(network_list):

                    network = net_dict['network']
                    network.to(up_device)

                    ### Computing Segmentation ###
                    data_iter.set_description('Segmenting... [Vol {}/{} | Net {}/{}]'.format(vol_count, n_vols, net_count+1, len(opt.networks_to_use)))


                    for slice_idx, input_slice in enumerate(tqdm(input_slices, desc='Running Segmentation...', position=1)):
                        if slice_idx in range(np.clip(liver_region[0],0,None),liver_region[1]):

                            n_ch = net_dict['settings'].Network['channels']
                            input_slice = input_slice[:,max_channels//2-n_ch//2:max_channels//2+n_ch//2+1,:]
                            input_slice = input_slice.type(torch.FloatTensor).to(up_device)

                            seg_pred    = network(input_slice)[0]

                            out_mask[:,:,slice_idx] += seg_pred.detach().cpu().numpy()[0,-1,:]

                out_mask    = np.round(out_mask/len(opt.networks_to_use))
                out_mask    = out_mask.astype(np.uint8)

                liver_masks = np.stack(liver_masks,axis=-1).astype(np.uint8)

                ### (optional) Running Post-Processing by removing tiny lesion speckles
                # data_iter.set_description('Post-Processing... [Vol {}/{}]'.format(vol_count, n_vols))
                # out_mask = snm.label(np.round(out_mask))[0].astype(np.uint8)
                # for i in np.unique(out_mask)[1:]:
                #     eqs = out_mask==i
                #     n_zs = len(list(set(list(np.where(eqs)[-1]))))
                #     if n_zs==1:
                #         out_mask[eqs] = 0
                # out_mask = out_mask>0
                # out_mask = out_mask.astype(np.uint8)


                ### Saving Final Segmentation Mask
                data_iter.set_description('Saving... [Vol {}/{}]'.format(vol_count, n_vols))
                #temp_liver_masks = (liver_masks+out_mask)>0
                #labels   = snm.label(temp_liver_masks)
                #MainCC   = labels[0]==np.argmax(np.bincount(labels[0].flat)[1:])+1
                labels   = snm.label(np.round(out_mask/len(opt.networks_to_use)))

                #out_mask = out_mask*MainCC + MainCC
                out_mask = out_mask*liver_masks + liver_masks

                data_header = test_dataloader.dataset.recon_info[prev_vol]['header']
                affine      = test_dataloader.dataset.recon_info[prev_vol]['affine']
                nifti_save_image = nib.Nifti1Image(out_mask, affine=affine)
                nifti_save_image.header['pixdim'] = data_header['pixdim']
                nib.save(nifti_save_image, opt.save_folder+'/test-segmentation-'+prev_vol.split('-')[-1])
                del out_mask

                ### Reseting Parameters
                vol_segs, input_slices, liver_masks = [],[],[]
                vol_count+=1
                data_iter.set_description('Completed. [Vol {}/{}]'.format(vol_count, n_vols))







if __name__ == '__main__':
    """======================================="""
    ### LOAD BASIC LIBRARIES
    import os,json,sys,gc,time,datetime,imp,argparse,copy
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, os.getcwd()+'/../Utilities')
    sys.path.insert(0, os.getcwd()+'/../Network_Zoo')

    import numpy as np, pickle as pkl, nibabel as nib
    from tqdm import tqdm,trange

    import pandas as pd
    from collections import OrderedDict
    import scipy.ndimage.measurements as snm
    import scipy.ndimage.morphology as snmo

    import network_zoo as netlib
    import General_Utilities as gu, Network_Utilities as nu

    import PyTorch_Datasets as Data

    import torch
    from torch.utils.data import DataLoader

    ###NOTE: This line is necessary in case of the "too many open files"-Error!
    if int(torch.__version__.split(".")[1])>2:
        torch.multiprocessing.set_sharing_strategy('file_system')



    """=========================================="""
    ### GET INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_folder",    type=str, default='placeholder')
    parser.add_argument("--network_choice",    type=str, default='placeholder')
    parser.add_argument("--test_data",         type=str, default='placeholder')
    parser.add_argument("--save_folder",       type=str, default='placeholder')
    parser.add_argument("--path_2_liv_seg",    type=str, default='placeholder')
    parser.add_argument("--use_all",           action='store_true')
    opt  = parser.parse_args()


    assert opt.network_choice != 'placeholder' or opt.use_all, 'Please insert name of network to use for lesion segmentation!'



    """===================================="""
    ### RUN GENERATION
    if not opt.use_all:
        opt.networks_to_use = [os.getcwd()+'/../SAVEDATA/Standard_Lesion_Networks/'+opt.network_choice] if opt.network_folder=='placeholder' else [opt.network_folder+'/'+opt.network_choice]
    else:
        opt.networks_to_use = [os.getcwd()+'/../SAVEDATA/Standard_Lesion_Networks/'+x for x in os.listdir(os.getcwd()+'/../SAVEDATA/Standard_Lesion_Networks')]


    """===================================="""
    ### ADJUST PLACEHOLDER VALUES IF NOT SPECIFIED
    if 'placeholder' in opt.test_data:
        opt.test_data       = os.getcwd()+'/../LOADDATA/Test_Data_2D'
    if 'placeholder' in opt.save_folder:
        opt.save_folder     = os.getcwd()+'/../SAVEDATA/Test_Segmentations/Test_Submissions'
    if 'placeholder' in opt.path_2_liv_seg:
        opt.path_2_liv_seg  = os.getcwd()+'/../SAVEDATA/Test_Segmentations/Liver_Segmentations'
    if not os.path.exists(opt.save_folder): os.makedirs(opt.save_folder)

    """=========================================="""
    ### RUN TEST SEGMENTATIONS
    main(opt)
