"""=================================================="""
### LOAD BASIC LIBRARIES
import argparse, numpy as np, os, sys, torch
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,os.getcwd()+'/../Utilities')
sys.path.insert(0,os.getcwd()+'/../Network_Zoo')
import General_Utilities as gu, Network_Utilities as nu
import network_zoo as netlib


"""=================================================="""
### GET NETWORK PARAMETERS
parse_in = argparse.ArgumentParser()
parse_in.add_argument('--base_setup',   type=str, default='Baseline_Parameters.txt',
                                        help='Path to baseline setup-txt which contains all major parameters that most likely will be kept constant during various grid searches.')
parse_in.add_argument('--search_setup', type=str, default='LiverNetwork_Parameters.txt',
                                        help='Path to search setup-txt, which contains (multiple) variations to the baseline proposed above.')
opt = parse_in.parse_args()
# opt = parse_in.parse_args(["--search_setup","Specific_Setup_Parameters_3D_LesionSegmentation_PC1.txt"])
opt.base_setup   = os.getcwd()+'/../Train_Networks/Training_Setup_Files/'+opt.base_setup
opt.search_setup = os.getcwd()+'/../Train_Networks/Training_Setup_Files/'+opt.search_setup

training_setups = gu.extract_setup_info(opt)
opt = training_setups[0]


"""================================================="""
### LOAD NETWORK
opt.Training['num_out_classes'] = 2
network = netlib.NetworkSelect(opt)
network.n_params = nu.gimme_params(network)
opt.Network['Network_name'] = network.name
device = torch.device('cuda')
_ = network.to(device)


### INPUT DATA
input_data   = torch.randn((1,opt.Network['channels'],256,256)).type(torch.FloatTensor).to(device)
network_pred = network(input_data)[0]


"""================================================="""
### SAVE COMPUTATION GRAPH
gu.save_graph(network_pred, os.getcwd(), opt.search_setup.split('.')[0], view=True)
