# Based on: Expandable script for Lesion Segmentation Base_Unet_Template.py.
# This Variant: Liver-Segmentation
# @author: Karsten Roth - Heidelberg University, 07/11/2017
"""==================================================================================================="""
"""======================= MAIN TRAINING FUNCTION/ALL FUNCTIONALITIES ================================"""
"""==================================================================================================="""
import os

def main(opt):
    """======================================================================================="""
    ### SET SOME DEFAULT PATHS
    if 'placeholder' in opt.Paths['Training_Path']:
        opt.Paths['Training_Path'] = os.getcwd()+'/../LOADDATA/Training_Data_2D'
    if 'placeholder' in opt.Paths['Save_Path']:
        foldername                 = 'Standard_Liver_Networks' if opt.Training['data']=='liver' else 'Standard_Lesion_Networks'
        opt.Paths['Save_Path']     = os.getcwd()+'/../SAVEDATA/'+foldername


    """======================================================================================="""
    ### REPRODUCIBILITY
    torch.manual_seed(opt.Training['seed'])
    torch.cuda.manual_seed(opt.Training['seed'])
    np.random.seed(opt.Training['seed'])
    random.seed(opt.Training['seed'])
    torch.backends.cudnn.deterministic = True



    """======================================================================================="""
    ### GPU SETUP
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.Training['gpu'])
    opt.device = torch.device('cuda')




    """======================================================================================="""
    if len(opt.Training['initialization']):
        try:
            ### NETWORK SETUP IF INITIALIZATION IS USED
            init_path   = opt.Training['initialization'] if not 'placeholder' in opt.Training['initialization'] else os.getcwd()+'/..'+opt.Training['initialization'].split('placeholder')[-1]
            network_opt = pkl.load(open(init_path+'/hypa.pkl','rb'))
            network     = netlib.NetworkSelect(network_opt)
            opt.Network = network_opt.Network
        except:
            raise Exception('Error when loading initialization weights! Please make sure that weights exist at {}!'.format(init_path))

    ### LOSS SETUP
    base_loss_func      = nu.Loss_Provider(opt)
    aux_loss_func       = nu.Loss_Provider(opt) if opt.Network['use_auxiliary_inputs'] else None
    opt.Training['use_weightmaps']  = base_loss_func.loss_func.require_weightmaps
    opt.Training['require_one_hot'] = base_loss_func.loss_func.require_one_hot
    opt.Training['num_out_classes'] = 1 if base_loss_func.loss_func.require_single_channel_input else opt.Training['num_classes']


    if len(opt.Training['initialization']):
        try:
            ### ONLY LOAD FEATURE WEIGHTS; FINAL LAYER IS SET UP ACCORDING TO USED LOSS
            checkpoint = torch.load(init_path+'/checkpoint_best_val.pth.tar')
            network.load_state_dict(checkpoint['network_state_dict'])
            if network.output_conv[0].out_channels != opt.Training['num_out_classes']:
                network.output_conv = torch.nn.Sequential(torch.nn.Conv2d(network.output_conv[0].in_channels, opt.Training['num_out_classes'], network.output_conv[0].kernel_size, network.output_conv[0].stride, network.output_conv[0].padding),
                                                          torch.nn.Sigmoid() if opt.Training['num_out_classes']==1 else torch.nn.Softmax(dim=1))

                if opt.Network['use_auxiliary_inputs']:
                    for i in range(len(network.auxiliary_preparators)):
                        in_channels = network.auxiliary_preparators[i].get_aux_output.in_channels
                        kernel_size = network.auxiliary_preparators[i].get_aux_output.kernel_size
                        stride      = network.auxiliary_preparators[i].get_aux_output.stride
                        network.auxiliary_preparators[i].get_aux_output = torch.nn.Conv2d(in_channels, opt.Training['num_out_classes'], kernel_size, stride)
                        network.auxiliary_preparators[i].out_act        = torch.nn.Sigmoid() if opt.Training['num_out_classes']==1 else torch.nn.Softmax(dim=1)
            del checkpoint
        except:
            raise Exception('Error when loading initialization weights! Please make sure that weights exist at {}!'.format(init_path))
    else:
        ### NETWORK SETUP WITHOUT INITIALIZATION
        network = netlib.NetworkSelect(opt)

    network.n_params = nu.gimme_params(network)
    opt.Network['Network_name'] = network.name
    _ = network.to(opt.device)



    """======================================================================================="""
    ### OPTIMIZER SETUP
    optimizer = torch.optim.Adam(network.parameters(), lr=opt.Training['lr'], weight_decay=opt.Training['l2_reg'])
    if isinstance(opt.Training['step_size'], list):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.Training['step_size'], gamma=opt.Training['gamma'])
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.Training['step_size'], gamma=opt.Training['gamma'])



    """======================================================================================="""
    ### TRAINING LOGGING SETUP
    # Set Logging Folder and Save Parameters
    imp.reload(gu)
    gu.logging_setup(opt)
    # Set Logging Dicts
    logging_keys    = ["Train Dice", "Train Loss", "Val Dice"]
    Metrics         = {key:[] for key in logging_keys}
    Metrics['Best Val Dice'] = 0
    # Set CSV Logger
    full_log  = gu.CSVlogger(opt.Paths['Save_Path']+"/log.csv", ["Epoch", "Time", "Training Loss", "Training Dice", "Validation Dice"])



    """======================================================================================="""
    ### TRAINING DATALOADER SETUP
    imp.reload(Data)
    train_dataset, val_dataset = Data.Generate_Required_Datasets(opt)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.Training['num_workers'], batch_size=opt.Training['batch_size'], pin_memory=False, shuffle=True)
    val_data_loader   = torch.utils.data.DataLoader(val_dataset,   num_workers=0, batch_size=1, shuffle=False)





    """======================================================================================="""
    ### START TRAINING
    full_training_start_time = time.time()
    epoch_iter = trange(0,opt.Training['n_epochs'],position=1)
    has_crop   = opt.Training['data']=='lesion'


    for epoch in epoch_iter:
        scheduler.step()
        epoch_iter.set_description("(#{}) Training [lr={}]".format(network.n_params, np.round(scheduler.get_lr(),8)))

        epoch_time = time.time()

        ###### Training ########
        flib.trainer([network,optimizer], train_data_loader, [base_loss_func, aux_loss_func], opt, Metrics, epoch)
        torch.cuda.empty_cache()


        ###### Validation #########
        epoch_iter.set_description('(#{}) Validating...'.format(network.n_params))
        flib.validator(network, val_data_loader, opt, Metrics, epoch)
        torch.cuda.empty_cache()


        ###### Save Training/Best Validation Checkpoint #####
        save_dict = {'epoch': epoch+1, 'network_state_dict':network.state_dict(), 'current_train_time': time.time()-full_training_start_time,
                     'optim_state_dict':optimizer.state_dict(), 'scheduler_state_dict':scheduler.state_dict()}
        # Best Validation Score
        if Metrics['Val Dice'][-1]>Metrics['Best Val Dice']:
            torch.save(save_dict, opt.Paths['Save_Path']+'/checkpoint_best_val.pth.tar')
            Metrics['Best Val Dice'] = Metrics['Val Dice'][-1]
            gu.generate_example_plots_2D(network, train_dataset, val_dataset, opt, has_crop=has_crop, name_append='best_val_dice', n_plots=20, seeds=[111,2222])

        # After Epoch
        torch.save(save_dict, opt.Paths['Save_Path']+'/checkpoint.pth.tar')


        ###### Logging Epoch Data ######
        epoch_iter.set_description('Logging to csv...')
        full_log.write([epoch, time.time()-epoch_time, Metrics["Train Loss"][-1], Metrics["Train Dice"][-1], Metrics["Val Dice"][-1]])


        ###### Generating Summary Plots #######
        epoch_iter.set_description('Generating Summary Plots...')
        sum_title = 'Max Train Dice: {0:2.3f} | Max Val Dice: {1:2.3f}'.format(np.max(Metrics["Train Dice"]), np.max(Metrics["Val Dice"]))
        gu.progress_plotter(np.arange(len(Metrics['Train Loss'])), \
                            Metrics["Train Loss"],Metrics["Train Dice"],Metrics["Val Dice"],
                            opt.Paths['Save_Path']+'/training_results.svg', sum_title)

        _ = gc.collect()

        ###### Generating Sample Plots #######
        epoch_iter.set_description('Generating Sample Plots...')
        gu.generate_example_plots_2D(network, train_dataset, val_dataset, opt, has_crop=has_crop, name_append='end_of_epoch', n_plots=20, seeds=[111,2222])
        torch.cuda.empty_cache()




"""==================================================================================================="""
"""============================= ___________MAIN_____________ ========================================"""
"""==================================================================================================="""
if __name__ == '__main__':


    """===================================="""
    ### LOAD BASIC LIBRARIES
    import warnings
    warnings.filterwarnings("ignore")

    import os,json,sys,gc,time,datetime,imp,argparse
    from tqdm import tqdm, trange
    import torch, random
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    sys.path.insert(0, os.getcwd()+'/../Utilities')
    sys.path.insert(0, os.getcwd()+'/../Network_Zoo')

    import numpy as np, matplotlib, pickle as pkl
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    import network_zoo as netlib

    import General_Utilities as gu
    import Network_Utilities as nu

    import PyTorch_Datasets as Data
    import Function_Library as flib


    """===================================="""
    ### GET TRAINING SETUPs ###
    #Read network and training setup from text file.
    parse_in = argparse.ArgumentParser()
    parse_in.add_argument('--base_setup',   type=str, default='Baseline_Parameters.txt',
                                            help='Path to baseline setup-txt which contains all major parameters that most likely will be kept constant during various grid searches.')
    parse_in.add_argument('--search_setup', type=str, default='',
                                            help='Path to search setup-txt, which contains (multiple) variations to the baseline proposed above.')
    parse_in.add_argument('--no_date',      action='store_true', help='Do not use date when logging files.')
    # opt = parse_in.parse_args(['--search_setup','Small_UNet_Lesion.txt'])
    opt = parse_in.parse_args()

    assert opt.search_setup!='', 'Please provide a Variation-Parameter Text File!'

    opt.base_setup   = os.getcwd()+'/Training_Setup_Files/'+opt.base_setup
    opt.search_setup = os.getcwd()+'/Training_Setup_Files/'+opt.search_setup

    training_setups = gu.extract_setup_info(opt)

    for training_setup in tqdm(training_setups, desc='Setup Iteration... ', position=0):
        main(training_setup)
