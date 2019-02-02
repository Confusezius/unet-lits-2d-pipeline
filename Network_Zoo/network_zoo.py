"""============================="""
"""========= Libraries ========="""
"""============================="""
import os, time, datetime, sys, copy
sys.path.insert(0, '../Utilities')

import torch, torch.nn as nn
from torchvision import models
import torch.nn.functional as F

import numpy as np
import Network_Utilities as nu



"""======================================================"""
### General UNet Scaffold
def NetworkSelect(opt):
    if opt.Training['network_type']=='unet':
        return Scaffold_UNet(opt)




"""======================================================"""
### Simple per-dim layer selection class
class LayerSet(object):
    def __init__(self, opt):
        self.conv  = nn.Conv2d
        self.tconv = nn.ConvTranspose2d
        self.norm  = nn.BatchNorm2d if opt.Network['use_batchnorm'] else nn.GroupNorm
        self.pool      = nn.MaxPool2d
        self.dropout   = nn.Dropout2d



####################################################################################
####################################################################################
############################# UNET #################################################
####################################################################################
####################################################################################
"""======================================================"""
### General UNet Scaffold
class Scaffold_UNet(nn.Module):
    def __init__(self, opt, stack_info=[]):
        super(Scaffold_UNet, self).__init__()

        ####################################### EXPLANATION ##########################################
        # [SU]:  Standard UNet Elements as described by Ronneberger et al.
        # [AU]:  Advanced UNet Elements - Variations to tweak and possibly improve the network performance.
        #        This includes the following options:
        #           - Input Distribution
        #           - Pyramid-style Pooling
        #           - Auxiliary Inputs
        #           - Multitask Injection
        #           - Residual or Dense connections
        # [kU]:  Elements required to use this class as subclass for a kUNet-stack.
        # [sqU]: Squeeze-and-Excite Element to recalibrate feature maps, as shown by Hu et al.


        ####################################### PRIOR SETUP ##########################################
        self.pars          = opt
        self.pars.fset     = LayerSet(opt)


        ####################################### BACKWARD COMPATIBILITY ###############################
        if 'block_type' not in self.pars.Network.keys():
            self.pars.Network['block_type'] = 'base'
        if 'dilation' not in self.pars.Network.keys():
            self.pars.Network['dilation'] = [1]*len(self.pars.Network['structure'])
        if 'dilation_up' not in self.pars.Network.keys():
            self.pars.Network['dilation_up'] = self.pars.Network['dilation']


        ############################# [SU] IF NO SPECIFIC DECODING STRUCTURE IS GIVEN, COPY ENCODING ####################
        if not 'structure_up' in opt.Network.keys():
            self.pars.Network['structure_up']    = self.pars.Network['structure']
        if not 'filter_start_up' in opt.Network.keys():
            self.pars.Network['filter_start_up'] = self.pars.Network['filter_start']


        ############################# [SU] INITIAL NUMBER OF FILTERS WITHOUT INPUT DISTR. ###############################
        self.pars.Network["filter_sizes"]    = [int(x) for x in self.pars.Network['filter_start']*2**np.array(list(range(0,len(self.pars.Network["structure"])+1)))]
        self.pars.Network["filter_sizes_up"] = [int(x) for x in self.pars.Network['filter_start_up']*2**np.array(list(range(0,len(self.pars.Network["structure"])+1)))]


        ####################################### UNIQUE NETWORK NAME ################################################
        self.name = "vUnet2D"


        ############### [SU] Create list of filter pairs corresponding to in- and outputfilters per block ###############
        down_filter_arrangements = list(zip(self.pars.Network["filter_sizes"][:-1],self.pars.Network["filter_sizes"][1:]))
        up_filter_arrangements   = list(zip(self.pars.Network["filter_sizes_up"][::-1][:-2],self.pars.Network["filter_sizes_up"][::-1][1:-1]))
        up_filter_arrangements[0] = (down_filter_arrangements[-1][-1],up_filter_arrangements[1][0])

        ############# [kU] ADJUST INPUT FILTERS IF USING STACK OF NETWORKS ##############################################
        if not len(stack_info): stack_info = [0]*len(down_filter_arrangements)
        if len(stack_info)<len(down_filter_arrangements): stack_info = stack_info+[0]*(len(down_filter_arrangements)-len(stack_info))


        ####################################### [SU] INPUT CONV #########################################################
        self.input_conv          = [self.pars.fset.conv(self.pars.Network["channels"]+stack_info[0]*self.pars.Training['num_out_classes'], self.pars.Network["filter_start"],3,1,1)]
        if self.pars.Network['use_batchnorm']:
            self.input_conv.append(self.pars.fset.norm(self.pars.Network['filter_start']))
        else:
            ### For the first layer, group norm reduces to instance norm
            self.input_conv.append(self.pars.fset.norm(self.pars.Network['filter_start'], self.pars.Network['filter_start']))
        self.input_conv.append(nn.LeakyReLU(0.05))
        self.input_conv          = nn.Sequential(*self.input_conv)


        ####################################### [AU] PYPOOL #############################################################
        kernel_padding_pars = [(1,1,0)]+[(2**(i+2), 2**(i+1), 2**i) for i in range(len(self.pars.Network['structure'])-1)]
        if self.pars.Network['use_pypool']:
            self.pypools             = nn.ModuleList([self.pars.fset.tconv(f[0], 1, setup[0], setup[1], setup[2]) for setup,f in zip(kernel_padding_pars[1:][::-1], up_filter_arrangements)])


        ####################################### [SU,AU] OUTPUT CONV ########################################################
        add = len(self.pypools) if self.pars.Network['use_pypool'] else 0
        outact                  = nn.Sigmoid() if self.pars.Training['num_out_classes']==1 else nn.Softmax(dim=1)
        self.output_conv        = nn.Sequential(self.pars.fset.conv(self.pars.Network["filter_start_up"]*2+add,self.pars.Training["num_out_classes"],1,1,0), outact)



        ####################################### [AU] AUXILIARY preparators ###############################################
        up_filter_4_aux     = self.pars.Network["filter_sizes_up"]
        up_filter_4_aux[-1] =  self.pars.Network["filter_sizes"][-1]
        self.auxiliary_preparators = nn.ModuleList([Auxiliary_Preparator(filter_in, self.pars, self.pars.Training['num_out_classes']) for n,filter_in in enumerate(up_filter_4_aux[::-1][:-2]) if n<len(self.pars.Network['structure'])-1]) if self.pars.Network['use_auxiliary_inputs'] else None


        ####################################### [SU,AU] PROJECTION TO LATENT SPACE ##########################################
        # self.downconv_blocks    = [UNetBaseBlock(self.pars["filter_start"],self.pars["filter_start"], self.pars, 0)]
        self.downconv_blocks    = [UNetBlockDown(f_in+f_out*stack_info[i]*(i>0),f_out, self.pars, self.pars.Network['structure'][i], self.pars.Network['dilation'][i]) for i,(f_in,f_out) in enumerate(down_filter_arrangements)]
        self.downconv_blocks    = nn.ModuleList(self.downconv_blocks)


        ####################################### [SU] RECONSTRUCTION FROM LATENT SPACE #####################################
        self.upconv_blocks      = [UNetBlockUp(f_t_in, f_in, f_out, self.pars, self.pars.Network['structure_up'][-(i+1)], self.pars.Network['dilation_up'][-(i+1)]) for i,((_,f_t_in),(f_in,f_out)) in enumerate(zip(down_filter_arrangements[::-1][1:],up_filter_arrangements))]
        self.upconv_blocks      = nn.ModuleList(self.upconv_blocks)


        ####################################### [SU] INITIALIZE PARAMETERS #################################################
        self.weight_init()




    def forward(self,net_layers):
        n_up_blocks   = len(self.upconv_blocks)

        ### [SU] INITIAL CONVOLUTION
        net_layers  = self.input_conv(net_layers)


        ############################################ DOWN ##################################################################
        ### ENCODING
        horizontal_connections = []
        for maxpool_iter in range(len(self.pars.Network["structure"])-1):
            ### [SU] STANDARD CONV
            net_layers, pass_layer      = self.downconv_blocks[maxpool_iter](net_layers)

            ### [SU] HORIZONTAL PASSES
            horizontal_connections.append(pass_layer)



        ############################################ BOTTLENECK ############################################################
        ### [SU] STANDARD CONV
        _, net_layers = self.downconv_blocks[-1](net_layers)

        ### [AU] PYPOOL
        if self.pars.Network['use_pypool']:
            pypool_inputs = [net_layers]



        ############################################ UP ##################################################################
        ### DECODING
        auxiliaries = [] if self.pars.Network['use_auxiliary_inputs'] and self.training else None
        for upconv_iter in range(n_up_blocks):
            ### [AU] AUXILIARY INPUTS
            if upconv_iter<n_up_blocks and self.pars.Network['use_auxiliary_inputs'] and self.training:
                auxiliaries.append(self.auxiliary_preparators[upconv_iter](net_layers))

            ### [SU] HORIZONTAL PASSES
            hor_pass    = horizontal_connections[::-1][upconv_iter]

            ### [SU] STANDARD UPCONV
            net_layers  = self.upconv_blocks[upconv_iter](net_layers, hor_pass)
            ### [AU] PYPOOL
            if self.pars.Network['use_pypool']:
                if upconv_iter<n_up_blocks-1:
                    pypool_inputs.append(net_layers)
                if upconv_iter==n_up_blocks-1:
                    pypool_inputs = [pypool_prep(pypool_input) for pypool_prep, pypool_input in zip(self.pypools, pypool_inputs)]
                    pypool_inputs = torch.cat(pypool_inputs, dim=1)
                    net_layers    = torch.cat([net_layers, pypool_inputs], dim=1)

        if self.pars.Network['use_auxiliary_inputs'] and self.training:
            auxiliaries = auxiliaries[::-1]



        #################################################### OUT ############################################################
        ### [SU] OUTPUT CONV
        net_layers = self.output_conv(net_layers)


        return net_layers, auxiliaries



    def weight_init(self):
        for net_segment in self.modules():
            if isinstance(net_segment, self.pars.fset.conv):
                if self.pars.Network['init_type']=="xavier_u":
                    torch.nn.init.xavier_uniform(net_segment.weight.data)
                elif self.pars.Network['init_type']=="he_u":
                    torch.nn.init.kaiming_uniform(net_segment.weight.data)
                elif self.pars.Network['init_type']=="xavier_n":
                    torch.nn.init.xavier_normal(net_segment.weight.data)
                elif self.pars.Network['init_type']=="he_n":
                    torch.nn.init.kaiming_normal(net_segment.weight.data)
                else:
                    raise NotImplementedError("Initialization {} not implemented.".format(init_type))

                torch.nn.init.constant(net_segment.bias.data, 0)





"""======================================================"""
### Basic ResnetBlock
class ResBlock(nn.Module):
    def __init__(self, filters_in, filters_out, pars, dilate_val, reduce=4):
        super(ResBlock, self).__init__()
        self.pars = pars
        self.net = nn.Sequential(self.pars.fset.conv(filters_in, filters_in//reduce, 1, 1, 0),
                                 self.pars.fset.norm(filters_in//reduce) if self.pars.Network['use_batchnorm'] else self.pars.fset.norm(filters_in//reduce//4, filters_in//reduce),
                                 nn.LeakyReLU(0.05),
                                 self.pars.fset.conv(filters_in//reduce, filters_in//reduce, 3, 1, dilate_val, dilation=dilate_val),
                                 self.pars.fset.norm(filters_in//reduce) if self.pars.Network['use_batchnorm'] else self.pars.fset.norm(filters_in//reduce//4, filters_in//reduce),
                                 nn.LeakyReLU(0.05),
                                 self.pars.fset.conv(filters_in//reduce, filters_out, 1, 1, 0))

    def forward(self,x):
        return self.net(x)


### Basic ResNeXtBlock
class ResXBlock(nn.Module):
    def __init__(self, filters_in, filters_out, dilate_val, pars):
        super(ResXBlock, self).__init__()
        self.pars = pars
        group_reduce, cardinality = filters_in//8, np.clip(filters_in//8,None,32).astype(int)
        self.blocks = nn.ModuleList([ResBlock(filters_in, filters_out, self.pars, dilate_val, reduce=group_reduce) for _ in range(cardinality)])

    def forward(self,x):
        for i,block in enumerate(self.blocks):
            if i==0:
                out = block(x)
            else:
                out = out + block(x)
        return out


### Basic Encoding Block
class UNetBlockDown(nn.Module):
    def __init__(self, filters_in, filters_out, pars, reps, dilate_val):
        super(UNetBlockDown, self).__init__()
        self.pars  = pars

        ### ADD OPTIONS FOR RESIDUAL/DENSE SKIP CONNECTIONS
        self.dense      = self.pars.Network['backbone']=='dense_residual'
        self.residual   = 'residual' in self.pars.Network['backbone']

        ### SET STANDARD CONVOLUTIONAL LAYERS
        self.convs, self.norms, self.dropouts, self.acts = [],[],[],[]

        for i in range(reps):
            f_in = filters_in if i==0 else filters_out

            if self.pars.Network['block_type']=='res':
                self.convs.append(ResBlock(f_in, filters_out, self.pars, dilate_val))
            elif self.pars.Network['block_type']=='resX':
                self.convs.append(ResXBlock(f_in, filters_out, self.pars, dilate_val))
            else:
                self.convs.append(self.pars.fset.conv(f_in, filters_out, 3, 1, dilate_val, dilation = dilate_val))


            if self.pars.Network['use_batchnorm']:
                #Set BatchNorm Filters
                self.norms.append(self.pars.fset.norm(f_in))
            else:
                #Set GroupNorm Filters
                self.norms.append(self.pars.fset.norm(f_in if f_in<self.pars.Network['filter_start']*2 else self.pars.Network['filter_start']*2, f_in))

            self.acts.append(nn.LeakyReLU(0.05))
            if self.pars.Network['dropout']: self.dropouts.append(self.pars.fset.dropout(self.pars.Network['dropout']))


        if self.pars.Network['se_reduction']: self.SE = SE_recalibration(filters_out, pars)

        self.convs, self.norms, self.dropouts, self.acts = nn.ModuleList(self.convs), nn.ModuleList(self.norms), nn.ModuleList(self.dropouts), nn.ModuleList(self.acts)


        ### ADD LAYERS THAT ADJUST THE CHANNELS OF THE INPUT LAYER TO THE BLOCK
        if filters_in!=filters_out and self.residual:
            self.adjust_channels = self.pars.fset.conv(filters_in, filters_out, 1, 1)
        else:
            self.adjust_channels = None

        ### ADD LAYERS FOR OPTIONAL CONVOLUTIONAL POOLING
        self.pool = self.pars.fset.conv(filters_out, filters_out, 3, 2, 1) if self.pars.Network["use_conv_pool"] else self.pars.fset.pool(kernel_size=3, stride=2, padding=1)



    def forward(self, net_layers):
        if self.dense:
            dense_list = []

        for i in range(len(self.convs)):
            ### NORMALIZE INPUT
            net_layers = self.norms[i](net_layers)

            ### ADJUST CHANNELS IF REQUIRED FOR RESIDUAL CONNECTIONS
            if self.residual:
                residual = self.adjust_channels(net_layers) if i==0 and self.adjust_channels is not None else net_layers
                if self.dense:
                    dense_list.append(residual)

            ### RUN SUBLAYER
            net_layers  = self.convs[i](net_layers)

            ### ADD SKIP CONNECTIONS TO OUTPUT
            if self.adjust_channels is not None:
                #dense connections
                if self.dense:
                    for residual in dense_list:
                        net_layers += residual
                #residual skip connections
                else:
                    net_layers += residual

            ### RUN THROUGH ACTIVATION
            net_layers = self.acts[i](net_layers)

            ### RUN THROUGH DROPOUT
            if self.pars.Network['dropout']: net_layers = self.dropouts[i](net_layers)

        ### RUN THROUGH SQUEEZE AND EXCITATION MODULE
        if self.pars.Network['se_reduction']: net_layers = self.SE(net_layers)

        ### LAYER TO BE USED FOR HORIZONTAL SKIP CONNECTIONS ACROSS U.
        pass_layer = net_layers

        ### CONVOLUTIONAL POOLING IF REQUIRED.
        net_layers = self.pool(net_layers)

        return net_layers, pass_layer





"""======================================================"""
### Horizontal UNet Block - Up
class UNetBlockUp(nn.Module):
    def __init__(self, filters_t_in, filters_in, filters_out, pars, reps, dilate_val):
        super(UNetBlockUp, self).__init__()

        self.pars       = pars
        upc, ups_mode   = self.pars.Network['up_conv_type'], 'nearest'
        self.conv_t     = self.pars.fset.tconv(filters_in, filters_out, upc[0], upc[1], upc[2]) if len(upc) else nn.Sequential(nn.Upsample(scale_factor=2, mode=ups_mode),self.pars.fset.conv(filters_in, filters_out, 1,1,0))
        # self.conv_t     = self.pars.fset.tconv(filters_in, filters_out, upc[0], upc[1], upc[2]) if len(upc) else nn.Sequential(nn.Upsample(scale_factor=2, mode=ups_mode),self.pars.fset.conv(filters_in, filters_out, 3,1,1))

        ### SET STANDARD CONVOLUTIONAL LAYERS
        self.convs, self.norms, self.dropouts, self.acts = [],[],[],[]


        for i in range(reps):
            f_in = filters_out+filters_t_in if i==0 else filters_out

            if self.pars.Network['block_type']=='res':
                self.convs.append(ResBlock(f_in, filters_out, self.pars, dilate_val))
            elif self.pars.Network['block_type']=='resX':
                self.convs.append(ResXBlock(f_in, filters_out, self.pars, dilate_val))
            else:
                self.convs.append(self.pars.fset.conv(f_in, filters_out, 3, 1, dilate_val, dilation = dilate_val))


            if self.pars.Network['use_batchnorm']:
                #Set BatchNorm Filters
                self.norms.append(self.pars.fset.norm(f_in))
            else:
                #Set GroupNorm Filters
                self.norms.append(self.pars.fset.norm(f_in if f_in<self.pars.Network['filter_start_up']*2 else self.pars.Network['filter_start_up']*2, f_in))

            self.acts.append(nn.LeakyReLU(0.05))
            if self.pars.Network['dropout']: self.dropouts.append(self.pars.fset.dropout(self.pars.Network['dropout']))

        if self.pars.Network['se_reduction']: self.SE = SE_recalibration(filters_out, pars)

        self.convs, self.norms, self.dropouts, self.acts = nn.ModuleList(self.convs), nn.ModuleList(self.norms), nn.ModuleList(self.dropouts), nn.ModuleList(self.acts)

        ### ADD OPTIONS FOR RESIDUAL/DENSE SKIP CONNECTIONS
        self.dense      = self.pars.Network['backbone']=='dense_residual'
        self.residual   = 'residual' in self.pars.Network['backbone']

        ### ADD LAYERS THAT ADJUST THE CHANNELS OF THE INPUT LAYER TO THE BLOCK
        if filters_in!=filters_out and self.residual:
            self.adjust_channels = self.pars.fset.conv(filters_out+filters_t_in, filters_out, 1, 1)
        else:
            self.adjust_channels = None


    def forward(self, net_layers, net_layers_hor_pass):
        net_layers      = self.conv_t(net_layers)
        net_layers      = torch.cat([net_layers, net_layers_hor_pass], dim=1)

        if self.dense: dense_list = []

        for i in range(len(self.convs)):
            ### NORMALIZE INPUT
            net_layers = self.norms[i](net_layers)

            ### ADJUST CHANNELS IF REQUIRED FOR RESIDUAL CONNECTIONS
            if self.residual: residual = self.adjust_channels(net_layers) if i==0 and self.adjust_channels is not None else net_layers
            if self.dense: dense_list.append(residual)

            ### RUN SUBLAYER
            net_layers  = self.convs[i](net_layers)

            ### ADD SKIP CONNECTIONS TO OUTPUT
            if self.adjust_channels is not None:
                #dense connections
                if self.dense:
                    for residual in dense_list:
                        net_layers += residual
                #residual skip connections
                else:
                    net_layers += residual


            ### RUN THROUGH ACTIVATION
            net_layers = self.acts[i](net_layers)

            ### RUN THROUGH DROPOUT
            if self.pars.Network['dropout']: net_layers = self.dropouts[i](net_layers)

        ### RUN THROUGH SQUEEZE AND EXCITATION MODULE
        if self.pars.Network['se_reduction']: net_layers = self.SE(net_layers)

        return net_layers




"""======================================================"""
### Layer to prepare auxiliary outputs
class Auxiliary_Preparator(nn.Module):
    def __init__(self, filters_in, pars, num_classes):
        super(Auxiliary_Preparator, self).__init__()
        self.pars = pars
        self.get_aux_output = self.pars.fset.conv(filters_in, num_classes, kernel_size=1, stride=1, padding=0)
        self.out_act        = nn.Softmax(dim=1) if num_classes>1 else nn.Sigmoid()

    def forward(self, x):
        return self.out_act(self.get_aux_output(x))




"""======================================================"""
### Squeeze-And-Excitation Layer
class SE_recalibration(nn.Module):
    def __init__(self, f_in, opt):
        super(SE_recalibration, self).__init__()
        self.pars = opt
        self.pool = nn.AdaptiveAvgPool2d(1) if opt.Training['mode']=='2D' else nn.AdaptiveAvgPool3d(1)
        self.recalib_weights = nn.Sequential(nn.Linear(f_in, int(f_in*opt.Network['se_reduction'])),
                                             nn.LeakyReLU(0.05), nn.Linear(int(f_in*opt.Network['se_reduction']), f_in),
                                             nn.Sigmoid())
    def forward(self, x):
        bs, ch = x.size()[:2]
        recalib_weights = self.pool(x).view(bs,ch)
        if self.pars.Training['mode']=='2D':
            recalib_weights = self.recalib_weights(recalib_weights).view(bs,ch,1,1)
        else:
            recalib_weights = self.recalib_weights(recalib_weights).view(bs,ch,1,1,1)
        return x*recalib_weights
