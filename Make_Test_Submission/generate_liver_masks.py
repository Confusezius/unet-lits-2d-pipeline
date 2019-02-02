# @author: Karsten Roth - Heidelberg University, 07/11/2017


"""======================================="""
"""========= TEST DATASET ================"""
"""======================================="""
import torch.utils.data as data
class TestDataset(data.Dataset):
    def __init__(self, test_data_folder, opt, channel_size=1):
        self.pars = opt
        self.test_volumes = [x for x in os.listdir(test_data_folder+'/Volumes')]
        # self.test_volumes = sorted([x for x in os.listdir(test_data_folder)])
        ### Choose specific volumes
        # self.test_volumes = ['test-volume-42', 'test-volume-5']
        self.test_volume_slices = {key:[test_data_folder+'/Volumes/'+key+'/'+x for x in sorted(os.listdir(test_data_folder+'/Volumes/'+key),key=lambda x: [int(y) for y in x.split('-')[-1].split('.')[0].split('_')])] for key in self.test_volumes}
        self.channel_size = channel_size

        self.iter_data, slice_cluster_collect = [],[]
        for vol in self.test_volumes:
            for i in range(len(self.test_volume_slices[vol])):
                extra_ch  = self.channel_size//2
                low_bound = np.clip(i-extra_ch,0,None).astype(int)
                low_diff  = extra_ch-i
                up_bound  = np.clip(i+extra_ch+1,None,len(self.test_volume_slices[vol])).astype(int)
                up_diff   = i+extra_ch+1-len(self.test_volume_slices[vol])

                vol_slices = self.test_volume_slices[vol][low_bound:up_bound]

                if low_diff>0:
                    extra_slices    = self.test_volume_slices[vol][low_bound+1:low_bound+1+low_diff][::-1]
                    vol_slices      = extra_slices+vol_slices
                if up_diff>0:
                    extra_slices    = self.test_volume_slices[vol][up_bound-up_diff-1:up_bound-1][::-1]
                    vol_slices      = vol_slices+extra_slices

                slice_cluster_collect.append(vol_slices)
                self.iter_data.append((vol,i))

            self.test_volume_slices[vol] = slice_cluster_collect
            slice_cluster_collect = []


        self.n_files = len(self.iter_data)
        self.vol_slice_idx = 0
        self.curr_vol = 0
        self.curr_vol_name = self.test_volumes[0]

    def __getitem__(self, idx):
        VOI, SOI = self.iter_data[idx]
        V2O  = np.concatenate([np.expand_dims(np.load(vol),0) for vol in self.test_volume_slices[VOI][SOI]],axis=0)
        if self.pars.Training['no_standardize']:
            V2O  = gu.normalize(V2O, zero_center=False, unit_variance=False, supply_mode="orig")
        else:
            V2O  = gu.normalize(V2O)

        if self.vol_slice_idx==len(self.test_volume_slices[self.test_volumes[self.curr_vol]])-1:
            self.vol_slice_idx = 0
            self.curr_vol_name = self.test_volumes[self.curr_vol]
            self.curr_vol     += 1
            return_data = {'VolSlice':V2O,'end_volume':True}
        else:
            return_data = {'VolSlice':V2O,'end_volume':False}
            self.vol_slice_idx+=1
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
    test_dataset    = TestDataset(opt.test_data, network_list[0]['settings'], channel_size=max_channels)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=1, shuffle=False)

    input_slices, vol_segs, volume_info, volume_info_2 = [],[],{},{}
    n_vols = len(test_dataloader.dataset.test_volumes)
    vol_count = 1

    data_iter = tqdm(test_dataloader, position=0)


    ############# Run Test Mask Generation ##################
    for idx,data in enumerate(data_iter):
        ### Getting input slices ###
        data_iter.set_description('Reading... [Vol {}/{}]'.format(vol_count, n_vols))
        input_slices.append(data['VolSlice'])
        end_volume = data['end_volume'].numpy()[0]
        if end_volume:
            prev_vol = test_dataloader.dataset.test_volumes[test_dataloader.dataset.curr_vol-1]
            ### Computing Segmentation ###
            with torch.no_grad():
                vol_segs = np.zeros([512,512,len(input_slices)])
                for net_count,net_dict in enumerate(network_list):

                    network = net_dict['network']
                    network.to(up_device)

                    data_iter.set_description('Segmenting  [Vol {}/{} | Net {}/{}]'.format(vol_count, n_vols, net_count+1, len(opt.networks_to_use)))
                    for slice_idx, input_slice in enumerate(tqdm(input_slices, desc='Running Segmentation...', position=1)):

                        n_ch = net_dict['settings'].Network['channels']
                        input_slice = input_slice[:,max_channels//2-n_ch//2:max_channels//2+n_ch//2+1,:]
                        input_slice = input_slice.type(torch.FloatTensor).to(up_device)
                        seg_pred    = network(input_slice)[0]

                        vol_segs[:,:,slice_idx] += seg_pred.detach().cpu().numpy()[0,-1,:]
                        #if seg_pred.size()[1]==2:
                        #    seg_pred = np.argmax(seg_pred.detach().cpu().numpy(),axis=1)
                        #else:
                        #    seg_pred = seg_pred.detach().cpu().numpy()[0,:]
                        #vol_segs.append(seg_pred)

                    ### Moving Network back to cpu
                    network.to(back_device)
                    torch.cuda.empty_cache()

                ### Finding Biggest Connected Component
                data_iter.set_description('Finding CC... [Vol {}/{} | Net {}/{}]'.format(vol_count, n_vols, net_count+1, len(opt.networks_to_use)))
                #vol_segs = np.vstack(vol_segs).transpose(1,2,0)
                labels   = snm.label(np.round(vol_segs/len(opt.networks_to_use)))
                #labels   = snm.label(np.round(vol_segs/len(opt.networks_to_use)))
                MainCC   = labels[0]==np.argmax(np.bincount(labels[0].flat)[1:])+1
                ### (Optional) Remove thin connections to, most likely, noisy segmentations
                # MainCC = snmo.binary_erosion(MainCC, np.ones((4,4,4)))
                # labels   = snm.label(MainCC)
                # MainCC = labels[0]==np.argmax(np.bincount(labels[0].flat)[1:])+1
                # del labels

                ### (Optional) Resize CC and close minor segmentation holes
                MainCC = snmo.binary_dilation(MainCC, np.ones((5,5,3)))
                MainCC = snmo.binary_erosion(MainCC, np.ones((2,2,3)))

                ### Extract Mask Coordinates for Lesion Segmentation to reduce computations
                MainCC    = MainCC.astype(np.uint8)
                mask_info = np.where(MainCC==1)
                mask_info = {axis_idx:(np.min(x), np.max(x)) for axis_idx,x in enumerate(mask_info)}
                volume_info[prev_vol+'.nii'] = mask_info

                ### Saving Liver Segmentation
                data_iter.set_description('Saving... [Vol {}/{}]'.format(vol_count, n_vols))
                vol_save = opt.save_folder+'/'+prev_vol
                if not os.path.exists(vol_save):
                    os.makedirs(vol_save)

                for slice_i in trange(MainCC.shape[-1], desc='Saving to npy slice...', position=1):
                    np.save(vol_save+'/slice-'+str(slice_i)+'.npy', MainCC[:,:,slice_i])

                pkl.dump(volume_info, open(opt.save_folder+'/liver_bound_dict.pkl','wb'))

                ### Reseting Parameters
                del MainCC
                vol_segs, input_slices = [],[]
                vol_count+=1
                #data_iter.set_description('Completed. [Vol {}/{}]'.format(vol_count, n_vols))








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
    parser.add_argument("--use_all",           action='store_true')
    opt  = parser.parse_args()

    assert opt.network_choice != 'placeholder' or opt.use_all, 'Please insert name of network to use for liver segmentation!'



    """===================================="""
    ### RUN GENERATION
    if not opt.use_all:
        opt.networks_to_use = [os.getcwd()+'/../SAVEDATA/Standard_Liver_Networks/'+opt.network_choice] if opt.network_folder=='placeholder' else [opt.network_folder+'/'+opt.network_choice]
    else:
        opt.networks_to_use  = [os.getcwd()+'/../SAVEDATA/Standard_Liver_Networks/'+x for x in os.listdir(os.getcwd()+'/../SAVEDATA/Standard_Liver_Networks')]

    if 'placeholder' in opt.test_data:
        opt.test_data        = os.getcwd()+'/../LOADDATA/Test_Data_2D'

    if 'placeholder' in opt.save_folder:
        opt.save_folder      = os.getcwd()+'/../SAVEDATA/Test_Segmentations/Liver_Segmentations'


    """=========================================="""
    ### RUN TEST SEGMENTATIONS
    main(opt)
