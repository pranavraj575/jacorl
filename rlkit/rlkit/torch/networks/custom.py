"""
Random networks
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer

from rlkit.torch.networks import LayerNorm
from rlkit.torch.pytorch_util import activation_from_string

class CumNet(PyTorchModule):
    def __init__(
            self,
            state_dim, # first state_dim elements of input is the state, rest is image (i.e. input[:state_dim] is the state vector)
            
            state_hidden_sizes, # list, is fully connected hidden layers applied to state vector 
                                #    (can be none, should prob be small, since the rest of the network is also fully connected)
            
            input_width, # for image
            input_height,
            input_channels, #image must be input[state_dim:].reshape(input_width,input_height,input_channels)
                            # as a result, input dimension must be state_dim + input_width*input_height*input_channels
            
            kernel_sizes, # list, is the kernel (x,y) for each layer of cnn
            n_channels,   # list, channel output for each layer of cnn
            strides,      # list, stride for each layer of cnn
            paddings,     # list, paddings for each layer of cnn
                          # all of these are lists of the same length, encodes the convolutional layers
            
            
            conv_output_size, # vector length for cnn to output
            
            combined_hidden_sizes, # list, fully connected hidden layers applied to combined 
            
            
            overall_output_size,
            
            # NN stuff
            init_w=3e-3,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            nn_hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
                          
            # CNN stuff
            pool_type='none', # 'max2d' if we want max pooling
            pool_sizes=None, # list, should be same length as convolution layers, is size of pool (x,y)
            pool_strides=None, # list, is stride for each pool layer
            pool_paddings=None, # list, is paddings for each pool layer
            
            conv_normalization_type='none', 
            cnn_hidden_init=nn.init.xavier_uniform_,
            
            image_hidden_sizes=None, # hidden layers to add to output of cnn
            image_fc_normalization_type='none',
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()
            
        #self.hidden_activation = hidden_activation #USED FOR BOTH IMAGE AND FNN 
            
        
        #STATE VECTOR NN
        
        self.state_dim = state_dim
        self.state_output_size = state_dim # gonna be last layer size
        
        
        
        self.layer_norm = layer_norm
        
        
        self.state_fcs =nn.ModuleList()
        self.state_layer_norms = nn.ModuleList()
        in_size = self.state_dim

        for i, next_size in enumerate(state_hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.state_output_size = next_size # this is just gonna be the last layer size
            nn_hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.state_fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.state_layer_norms.append(ln)
        #FINISHED STATE, entirely possible this is empty
        # IMAGE CNN
        
        if image_hidden_sizes is None:
            image_hidden_sizes = []
        
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        
        assert conv_normalization_type in {'none', 'batch', 'layer'}
        assert image_fc_normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
            
        
        
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        
        self.image_hidden_sizes=image_hidden_sizes
        
        
        self.conv_normalization_type = conv_normalization_type
        self.image_fc_normalization_type = image_fc_normalization_type
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.pool_type = pool_type
        
        
        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.image_fc_layers = nn.ModuleList()
        self.image_fc_norm_layers = nn.ModuleList()
        
        
        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            cnn_hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == 'max2d':
                self.pool_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_sizes[i],
                        stride=pool_strides[i],
                        padding=pool_paddings[i],
                    )
                )
        
        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.conv_normalization_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.conv_normalization_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none':
                test_mat = self.pool_layers[i](test_mat)

        self.conv_output_flat_size = int(np.prod(test_mat.shape))
        
        
        
        image_fc_input_size = self.conv_output_flat_size
        for idx, hidden_size in enumerate(image_hidden_sizes):
            fc_layer = nn.Linear(image_fc_input_size, hidden_size)
            fc_input_size = hidden_size
            #nn_hidden_init(fc_layer.weight)
            #fc_layer.bias.data.fill_(b_init_value)
            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)
            
            self.image_fc_layers.append(fc_layer)
                
            if self.image_fc_normalization_type == 'batch':
                self.image_fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
            if self.image_fc_normalization_type == 'layer':
                self.image_fc_norm_layers.append(nn.LayerNorm(hidden_size))
                
            
        self.image_last_fc = nn.Linear(image_fc_input_size, conv_output_size)
        self.image_last_fc.weight.data.uniform_(-init_w, init_w)
        self.image_last_fc.bias.data.uniform_(-init_w, init_w)
        
        
        #FINISHED IMAGE CNN
        
        #FINAL VECTOR NN
        self.final_vector_init_size=self.state_output_size + conv_output_size
        self.final_output_size = overall_output_size
        
        self.final_output_activation = output_activation
        
        self.final_fcs=nn.ModuleList()
        self.final_layer_norms=nn.ModuleList()
        in_size=self.final_vector_init_size
        
        
        for i, next_size in enumerate(combined_hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            nn_hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.final_fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.final_layer_norms.append(ln)
                
        
        self.last_fc = nn.Linear(in_size, overall_output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)
    
    def image_forward(self,input):
        cont=input.contiguous()
        # reshape from batch of flattened images into (batch, w, h, channels)
        h=cont.view((input.shape[0],
                      self.image_width,
                      self.image_height,
                      self.input_channels))
        # torch cnn needs in the order (batch, channels, h, w)
        h=h.transpose(1,3)
        
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.conv_normalization_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none':
                h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
        
        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        
        
        for i, layer in enumerate(self.image_fc_layers):
            h = layer(h)
            if self.image_fc_normalization_type != 'none':
                h = self.image_fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        
        
        return self.image_output_activation(self.image_last_fc(h))
        
        
    def state_forward(self,input):
        h=input
        for i, fc in enumerate(self.state_fcs):
            h = fc(h)
            if self.state_layer_norms and i < len(self.fcs) - 1:
                h = self.state_layer_norms[i](h)
            h = self.hidden_activation(h)
        return h

    def forward(self, input, return_preactivations=False):
        # input should be batch of flattened images/state 
        # dimension is M x N where N is flattend state space size
        h_state=input[:,:self.state_dim]
        h_image=input[:,self.state_dim:]
        
        h_state=self.state_forward(h_state)
        
        h_image=self.image_forward(h_image)
        
        h=torch.cat((h_state,h_image),1)
        
        
        for i, fc in enumerate(self.final_fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.final_layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.final_output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output



