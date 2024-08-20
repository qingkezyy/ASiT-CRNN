import torch.nn as nn
import torch
import yaml
import numpy as np
import torch.nn.functional as F
from desed_task.audiossl.methods.sed.patch_ssast_model import SSAST
from desed_task.audiossl.methods.sed.mae_ast_patch_model import MAE_AST
from desed_task.audiossl.methods.sed.clip_atst_model import ATST
from desed_task.audiossl.methods.sed.audioMAE_model import AudioMAE
from desed_task.audiossl.methods.sed.beats_model import BEATs
from .RNN import BidirectionalGRU
from .CNN import CNN

# select different self-supervised pretraining models
with open('./confs/ssl_crnn.yaml', "r") as f: 
    configs = yaml.safe_load(f)
encoder_name = configs["comparison"]["model"]
print('Encoder of the selected self-supervised pre-trained model:', encoder_name)

class CRNN(nn.Module):
    def __init__(
        self,
        unfreeze_encoder_layer=0,
        n_in_channel=1,
        nclass=10,
        activation="glu",
        dropout=0.5,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        embedding_size=768,
        encoder_init=None,
        encoder_dropout=0.0,
        mode=None,
        **kwargs,
    ):
        super(CRNN, self).__init__()

        self.encoder_name = encoder_name
        self.n_in_channel = n_in_channel
        self.encoder_dropout = encoder_dropout
        n_in_cnn = n_in_channel
        self.cnn = CNN(
            n_in_channel=n_in_cnn, activation=activation, conv_dropout=dropout, **kwargs
        )

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1]
            nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()

        self.dense_softmax = nn.Linear(n_RNN_cell * 2, nclass)
        self.softmax = nn.Softmax(dim=-1)

        self.cat_tf = torch.nn.Linear(nb_in+embedding_size, nb_in) 
        # self.cat_tf = torch.nn.Linear(embedding_size, nb_in) # Use only encoder features 


        self.init_encoder(encoder_init)      
        self.unfreeze_encoder_layer = unfreeze_encoder_layer

    def init_encoder(self, path=None):

        if self.encoder_name == 'SSAST':
            path = configs["ultra"]["SSAST_Path"]
        elif self.encoder_name == 'AudioMAE':
            path = configs["ultra"]["AudioMAE_Path"]
        elif self.encoder_name == 'MAE_AST':
            path = configs["ultra"]["MAE_AST_Path"]
        elif self.encoder_name == 'ATST':
            path = configs["ultra"]["ATST_Path"]
        elif self.encoder_name == 'BEATs':
            path = configs["ultra"]["BEATs_Path"]
        else:
            raise Exception('Encoder path unrecognized.')

        if path is None:
            print("Path is empty")
        else:
            if self.encoder_name == 'SSAST':
                self.encoder = SSAST(path)
            elif self.encoder_name == 'AudioMAE':
                self.encoder = AudioMAE(path)
            elif self.encoder_name == 'MAE_AST':
                self.encoder = MAE_AST(path)
            elif self.encoder_name == 'ATST':
                self.encoder = ATST(path)
            elif self.encoder_name == 'BEATs':
                self.encoder = BEATs(path)
            else:
                raise Exception('Encoder unrecognized.')
            
            print("Loading encoder from:", path)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.detach_()

    def forward(self, x, pretrain_x, embeddings=None):
        
        x = x.transpose(1, 2).unsqueeze(1) # input x size : (batch_size, n_channels, n_frames, n_freq)
    
        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        x = x.squeeze(-1) 
        x = x.permute(0, 2, 1)  # x size : [batch_size, n_frames, n_channels]
        torch.use_deterministic_algorithms(False)

        # encoder features
        embeddings = self.asit(pretrain_x) # embeddings size : [batch_size, n_frames, embeddings_dim]
        embeddings = embeddings.transpose(1, 2)

        # nearest neighbor interpolation operation
        target_length = 156
        original_length = embeddings.shape[2]
        scale_factor = target_length / original_length
        embeddings = F.interpolate(embeddings, scale_factor=scale_factor, mode='linear', align_corners=False).transpose(1, 2)

        # CNN and Encoder feature fusion
        x = self.cat_tf(torch.cat((x, embeddings), -1))
       
        # use only encoder features
        # x = self.cat_tf(embeddings)

        # rnn features
        x = self.rnn(x)

        # T-SNE图
        # y = x.reshape(x.shape[0], x.shape[1] * x.shape[2]).cpu().numpy()
        # print('data_embed_npy:',y.shape)
        # np.save("./data_embed_npy_SSAST.npy",y)

        x = self.dropout(x)
        strong = self.dense(x)  # [batch_size, n_frames, n_class]
        strong = self.sigmoid(strong)
        sof = self.dense_softmax(x)  # [batch_size, n_frames, n_class]
        sof = self.softmax(sof)
        sof = torch.clamp(sof, min=1e-7, max=1)
        weak = (strong * sof).sum(1) / sof.sum(1)  # [batch_size, n_class]
        return strong.transpose(1, 2), weak

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CRNN, self).train(mode)
