from layer import *
import logging

class Model(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, dropout=0.1):
        super(Model, self).__init__()
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)

        emb_inter_t = nn.Embedding(u_dim, 1, padding_idx=0,_weight=torch.ones(u_dim,1))
        emb_inter_s = nn.Embedding(u_dim, 1, padding_idx=0,_weight=torch.ones(u_dim,1))
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_tu, emb_sl, emb_tl, emb_inter_t, emb_inter_s
        self.embed_dim = embed_dim
        self.emb = emb_u
        self.Agg = Agg(emb_u, embed_dim,embed_layers)
        self.AggLocation = AggLocation(emb_l, embed_dim,embed_layers)
        self.SoftmaxEmbed= SoftmaxEmbed(embed_dim, embed_layers)
        self.EmbedLocation = EmbedLocation(embed_dim, l_dim-1, embed_layers)
        

    def forward(self, traj, mat1, mat2, vec, traj_len, neibour, neibour_relation, neibour_location,location_data,ex,location_candidate):
        traj[:, :, 2] = (traj[:, :, 2]-1) % (24*7) + 1 
        user_neibour = self.Agg(traj[...,0], neibour, neibour_relation, traj_len,ex)
        user_record_embedding = self.SoftmaxEmbed(traj, user_neibour, mat1, traj_len, ex)
        location = self.AggLocation(neibour_location, location_data, traj_len, ex, traj[...,0], location_candidate)
        location_embedding = self.EmbedLocation(traj, mat2, vec,traj_len, location, ex, location_candidate) 
        out = torch.matmul(location_embedding, user_record_embedding).squeeze(-1) 
        return out
