import torch.nn as nn
import torch
import torch.nn.functional as F
from .asit.asit_model import ASiT
from .RNN import BidirectionalGRU
from .CNN import CNN

class CRNN(nn.Module):
    def __init__(
        self,
        unfreeze_asit_layer=0,
        n_in_channel=1,
        nclass=10,
        activation="glu",
        dropout=0.5,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        embedding_size=768,
        model_init=None,
        asit_init=None,
        asit_dropout=0.0,
        mode=None,
        **kwargs,
    ):
        super(CRNN, self).__init__()

        self.n_in_channel = n_in_channel
        self.asit_dropout = asit_dropout
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
        # self.cat_tf = torch.nn.Linear(embedding_size, nb_in) # use only ASiT features


        self.init_asit(asit_init)
        self.init_model(model_init, mode=mode)
        
        self.unfreeze_asit_layer = unfreeze_asit_layer

    def init_asit(self, path=None):
        if path is None:
            self.asit = ASiT(None)
        else:
            self.asit = ASiT(path)
            
            print("Loading asit from:", path)
        self.asit.eval()
        for param in self.asit.parameters():
            param.detach_()
    
    def init_model(self, path, mode=None):
        if path is None:
            pass
        else:
            if mode == "teacher":
                print("Loading teacher from:", path)
                state_dict = torch.load(path, map_location="cpu")["sed_teacher"]
            else:
                print("Loading student from:", path)
                state_dict = torch.load(path, map_location="cpu")["sed_student"]
            self.load_state_dict(state_dict, strict=True)
            print("Model loaded")

    def forward(self, x, pretrain_x, embeddings=None):

        x = x.transpose(1, 2).unsqueeze(1) # input x size : (batch_size, n_channels, n_frames, n_freq)
    
        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        x = x.squeeze(-1) 
        x = x.permute(0, 2, 1)  # x size : [batch_size, n_frames, n_channels]
        torch.use_deterministic_algorithms(False)

        # asit features
        embeddings = self.asit(pretrain_x) # embeddings size : [batch_size, n_frames, embeddings_dim]
        embeddings = embeddings.transpose(1, 2)

        # nearest neighbor interpolation operation
        target_length = 156
        original_length = embeddings.shape[2]
        scale_factor = target_length / original_length
        embeddings = F.interpolate(embeddings, scale_factor=scale_factor, mode='linear', align_corners=False).transpose(1, 2)

        # CNN and ASiT feature fusion
        x = self.cat_tf(torch.cat((x, embeddings), -1))
       
        # use only ASiT features
        # x = self.cat_tf(embeddings)

        # rnn features
        x = self.rnn(x)

        # T-SNE图
        # y = x.reshape(x.shape[0], x.shape[1] * x.shape[2]).cpu().numpy()
        # print('data_embed_npy:',y.shape)
        # np.save("./data_embed_npy.npy",y)

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

