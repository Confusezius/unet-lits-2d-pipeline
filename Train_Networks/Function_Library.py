# Training and validation functions for 2D/3D Liver/Lesion Segmentation Training
# of neural networks using the PyTorch framework.
# @author: Karsten Roth - Heidelberg University, 07/11/2017

"""==============================================================="""
### LIBRARIES
import warnings
warnings.filterwarnings("ignore")
import torch, random, numpy as np, sys, os, time
sys.path.insert(0, os.getcwd()+'/../Utilities')
import General_Utilities as gu, Network_Utilities as nu
from tqdm import tqdm, trange




"""==============================================================="""
### TRAIN ANY 2D/3D SEGMENTATION NETWORK FOR LIVER/LESION SEGMENTATION
def trainer(network_setup, data_loader, losses, opt, Metrics, epoch):
    network, optimizer = network_setup
    _ = network.train()

    base_loss_func, aux_loss_func = losses

    iter_preds_collect, iter_target_collect, iter_loss_collect = [],[],[]
    epoch_loss_collect, epoch_dice_collect = [],[]

    train_data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- | Dice: ---'.format(epoch)


    for slice_idx, file_dict in enumerate(train_data_iter):
        train_data_iter.set_description(inp_string)

        train_iter_start_time = time.time()

        training_slice  = file_dict["input_images"].type(torch.FloatTensor).to(opt.device)

        ### GET PREDICTION ###
        network_output, auxiliaries = network(training_slice)


        ### BASE LOSS ###
        feed_dict = {'inp':network_output}
        if base_loss_func.loss_func.require_single_channel_mask:
            feed_dict['target']         = file_dict['targets'].to(opt.device)
        if base_loss_func.loss_func.require_one_hot:
            feed_dict['target_one_hot'] = file_dict['one_hot_targets'].to(opt.device)
        if base_loss_func.loss_func.require_weightmaps:
            feed_dict['wmap']           = file_dict["weightmaps"].to(opt.device)
        loss_base = base_loss_func(**feed_dict)

        ### INJECTION LOSS ###
        loss_inj = torch.tensor(0).type(torch.FloatTensor).to(opt.device)

        ### AUXILIARY LOSS ###
        loss_aux = torch.tensor(0).type(torch.FloatTensor).to(opt.device)
        if opt.Network['use_auxiliary_inputs']:
            for aux_ix in range(len(auxiliaries)):
                feed_dict = {'inp':auxiliaries[aux_ix]}
                if aux_loss_func.loss_func.require_weightmaps:
                    feed_dict['wmap'] = file_dict['aux_weightmaps'][aux_ix].to(opt.device)
                if aux_loss_func.loss_func.require_one_hot:
                    feed_dict['target_one_hot'] = file_dict['one_hot_aux_targets'][aux_ix].to(opt.device)
                if aux_loss_func.loss_func.require_single_channel_mask:
                    feed_dict['target'] = file_dict['aux_targets'][aux_ix].to(opt.device)

                # loss_aux = loss_aux + aux_loss_func(**feed_dict)
                loss_aux = loss_aux + 1./(aux_ix+1)*aux_loss_func(**feed_dict)
            # loss_aux += 1/len(auxiliaries)

        ### COMBINE LOSS FUNCTIONS ###
        loss = loss_base+loss_aux


        ### RUN BACKPROP AND WEIGHT UPDATE ###
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### GET SCORES ###
        if network_output.shape[1]!=1:
            iter_preds_collect.append(np.argmax(network_output.detach().cpu().numpy(),axis=1))
        else:
            iter_preds_collect.append(np.round(network_output.detach().cpu().numpy()))

        iter_target_collect.append(file_dict['targets'].numpy())
        iter_loss_collect.append(loss.item())

        if slice_idx%opt.Training['verbose_idx']==0 and slice_idx!=0:
            ### Compute Dice Score of collected training samples
            verbose_dice = nu.Dice(np.vstack(iter_preds_collect),np.vstack(iter_target_collect))
            epoch_loss_collect.append(np.mean(iter_loss_collect))
            epoch_dice_collect.append(verbose_dice)

            ### Add Scores to metric collector
            ### Update tqdm string
            inp_string = 'Epoch {0} || Loss: {1:3.7f} | Dice: {2:2.5f}'.format(epoch, np.mean(epoch_loss_collect), np.mean(epoch_dice_collect))

            ### Reset mini collector lists
            iter_preds_collect, iter_target_collect, iter_loss_collect = [],[],[]


    Metrics["Train Dice"].append(np.mean(epoch_dice_collect))
    Metrics["Train Loss"].append(np.mean(epoch_loss_collect))






"""==============================================================="""
### VALIDATE ANY 2D SEGMENTATION NETWORK FOR LIVER/LESION SEGMENTATION
def validator(network, data_loader, opt, Metrics, epoch):
    _ = network.eval()

    iter_preds_collect, iter_target_collect = [],[]
    epoch_dice_collect = []

    validation_data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Dice: ---/Vol'.format(epoch)

    for slice_idx, file_dict in enumerate(validation_data_iter):
        validation_data_iter.set_description(inp_string)

        val_iter_start_time = time.time()
        validation_slice = file_dict["input_images"].type(torch.FloatTensor).to(opt.device)
        validation_mask  = file_dict["targets"].to(opt.device)
        if 'crop_option' in file_dict.keys(): validation_crop = file_dict['crop_option'].type(torch.FloatTensor).to(opt.device)
        network_output   = network(validation_slice)[0]
        if 'crop_option' in file_dict.keys(): network_output = network_output*validation_crop

        if opt.Training['num_out_classes']!=1:
            iter_preds_collect.append(np.argmax(network_output.detach().cpu().numpy(),axis=1))
        else:
            iter_preds_collect.append(np.round(network_output.detach().cpu().numpy()))
        iter_target_collect.append(validation_mask.detach().cpu().numpy())

        if file_dict['vol_change'] or slice_idx==len(data_loader)-1:
            mini_dice = nu.Dice(np.vstack(iter_preds_collect),np.vstack(iter_target_collect))
            epoch_dice_collect.append(mini_dice)

            inp_string = 'Epoch {0} || Dice: {1:2.5f}/Vol'.format(epoch,np.mean(epoch_dice_collect))
            validation_data_iter.set_description(inp_string)
            iter_preds_collect, iter_target_collect = [],[]

    Metrics['Val Dice'].append(np.mean(epoch_dice_collect))
