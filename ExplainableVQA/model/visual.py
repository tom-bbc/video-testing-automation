import torch
import torch.nn as nn

class EnhancedVisualEncoder(nn.Module):
    def __init__(self, clip_model, fast_vqa_encoder, aes_clip_len, tech_clip_len, tech_num_clips):
        super().__init__()
        self.clip_visual = clip_model.visual
        self.fast_vqa_encoder = fast_vqa_encoder.technical_backbone

        self.output_len = aes_clip_len
        self.tech_clip_len = tech_clip_len
        self.tech_num_clips = tech_num_clips

    def forward(self, x_tech, x_aes=None, aes_on=False):
        """
        clip_feats and fast_feats vectors must be of the same shape (diff lengths)
        as they are combined into a single vector of spatial + temporal features
        which is shaped `[1, self.output_len, 49, 1792]`
        where 1792 is the no. spatial CLIP features (1024) + no. temporal FAST features (768) combined.
        """

        # frame-wise
        if aes_on and x_aes is not None:
            x_aes = x_aes.transpose(1, 2).reshape(-1, 3, 224, 224)
            clip_feats = self.clip_visual(x_aes)
            clip_feats = clip_feats[1:].reshape(7, 7, -1, 1024).permute(3, 2, 0, 1)
            clip_feats = clip_feats.reshape(1024, -1, self.output_len, 49).permute(1, 2, 3, 0)
            # print("3a) CLIP features:", clip_feats.shape)
        else:
            clip_feats = torch.zeros([1, self.output_len, 49, 1024])
            # print("3a) Zeroed CLIP features:", clip_feats.shape)

        # chunk-wise
        x_tech = x_tech.reshape(-1, 3, self.tech_num_clips, self.tech_clip_len, 224, 224).permute(0,2,1,3,4,5).reshape(-1, 3, self.tech_clip_len, 224, 224)
        fast_feats = self.fast_vqa_encoder(x_tech).reshape(-1,4,768,16,7,7).permute(0,1,3,4,5,2)
        fast_feats = fast_feats.reshape(-1, self.output_len, 49, 768)

        return torch.cat((clip_feats, fast_feats), -1)
