import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
os.environ['TORCH_HOME'] = '../../pretrained_models'
import timm
from timm.models.layers import to_2tuple,trunc_normal_

import vision_transformer as vits


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    def __init__(self, finetune_from, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, patch_size=16, model_name='vit_base', verbose=True, keyword='teacher'):

        super(ASTModel, self).__init__()

        if verbose == True:
            print('---------------AST Model Summary---------------')
            
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        
        # load pretrained weights
        pretrained = finetune_from
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dic = checkpoint[keyword].copy()            
        state_dic = {k.replace("module.", ""): v for k, v in state_dic.items()}
        state_dic = {k.replace("backbone.", ""): v for k, v in state_dic.items()}  
        
        # create model
        _, self.original_num_patches, self.original_embedding_dim = state_dic['pos_embed'].shape
        self.original_num_patches -= 1
        self.oringal_hw = [224, 224] if self.original_num_patches == 196 else [self.original_num_patches*patch_size*patch_size//128, 128]            
        _, in_chans, _, _ = state_dic['patch_embed.proj.weight'].shape
        self.v = vits.__dict__[model_name](audio_size=self.oringal_hw, in_chans=in_chans)

        # update weights
        msg = self.v.load_state_dict(state_dic, strict=False)
        print(msg)
        
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
        self.oringal_hw = [self.oringal_hw[0] // patch_size, self.oringal_hw[1] // patch_size]

        # automatcially get the intermediate shape
        t_dim, f_dim = self.get_shape(fstride, tstride, patch_size, input_tdim, input_fdim)
        num_patches = f_dim * t_dim
        
        self.v.patch_embed.num_patches = num_patches
        if verbose == True:
            print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
            print('number of patches={:d}'.format(num_patches))


        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(patch_size, patch_size), stride=(fstride, tstride))
        if in_chans == 3:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
        else:
            new_proj.weight = self.v.patch_embed.proj.weight
        new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj
        

        # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
        new_pos_embed = self.v.pos_embed[:, 1:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw[0], self.oringal_hw[1])
        # cut (from middle) or interpolate the second dimension of the positional embedding
        if f_dim <= self.oringal_hw[1]:
            new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw[1] / 2) - int(f_dim / 2): int(self.oringal_hw[1] / 2) - int(f_dim / 2) + f_dim]
            
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw[0], f_dim), mode='bilinear')
        # cut (from middle) or interpolate the first dimension of the positional embedding
        if t_dim <= self.oringal_hw[0]:
            new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw[0] / 2) - int(t_dim / 2): int(self.oringal_hw[0] / 2) - int(t_dim / 2) + t_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(t_dim, f_dim), mode='bilinear')
        # flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
        # concatenate the above positional embedding with the cls token and distillation token of the deit model.
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :1, :].detach(), new_pos_embed], dim=1))


    def get_shape(self, fstride, tstride, patch_size, input_tdim=1024, input_fdim=128):
        test_input = torch.randn(1, 1, input_tdim, input_fdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(patch_size, patch_size), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        t_dim = test_out.shape[2]
        f_dim = test_out.shape[3]
        return t_dim, f_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        #x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        x = self.mlp_head(x[:, 0] )
        return x

if __name__ == '__main__':
    input_tdim = 100
    ast_mdl = ASTModel(input_tdim=input_tdim)
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)

    input_tdim = 256
    ast_mdl = ASTModel(input_tdim=input_tdim,label_dim=50, audioset_pretrain=True)
    # input a batch of 10 spectrogram, each with 512 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = ast_mdl(test_input)
    # output should be in shape [10, 50], i.e., 10 samples, each with prediction of 50 classes.
    print(test_output.shape)
