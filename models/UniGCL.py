import torch
from .BaseCF import BaseCF
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

class UniGCL(BaseCF):
    def __init__(self, data_config, device):
        super().__init__(data_config)
        self.dataset = data_config['dataset_name']
        self.emb_size = data_config['latent_dim']
        self.n_layers = data_config['gcn_layer']
        self.temperature = data_config['temperature']
        self.ssl_reg = data_config['ssl_reg']
        self.eps = data_config['eps']
        self.r_mat = self.create_r_mat()
        self.adj_mat = self.create_adj_mat()
        self.sparse_norm_adj = self.convert_csr_to_spare_tensor(self.UniGCNNormAdjMat(self.adj_mat)).to(device)
        self.device = device
        self.user_ips, self.item_ips = self.calculate_ips_from_sparse_adjacency(self.convert_csr_to_spare_tensor(self.r_mat).to(device), self.num_user, self.num_item)

    def forward(self):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        emb_lists = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            emb_lists.append(ego_embeddings)
        rec_embeddings = 0.2 * emb_lists[0] + 0.8 * emb_lists[1]
        user_rec_embeddings, item_rec_embeddings = torch.split(rec_embeddings, [self.num_user, self.num_item])
        return user_rec_embeddings, item_rec_embeddings

    def compute_batch_loss(self, user_emb, item_emb, u_idx, i_idx, j_idx):
        u_emb, i_emb, j_emb = self.get_embedding(user_emb, item_emb, u_idx, i_idx, j_idx)
        bpr_loss, auc = self.compute_bpr_loss(u_emb, i_emb, j_emb)
        cl_loss = self.compute_ips_cl_loss(user_emb, item_emb, u_idx, i_idx)
        reg_loss = self.compute_reg_loss_LGN(u_idx, i_idx, j_idx)
        total_loss = bpr_loss + cl_loss + reg_loss
        return auc, bpr_loss, cl_loss, reg_loss, total_loss

    @staticmethod
    # ************* convolution: D^-1/2 * A * 1/(ln(D)+2I) *************
    def UniGCNNormAdjMat(adj_mat):
        row_sum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv = np.power(row_sum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        log_d = np.zeros_like(row_sum, dtype=float)
        nonzero_mask = row_sum > 0
        log_d[nonzero_mask] = np.log(row_sum[nonzero_mask])
        log_d_plus_I = log_d + 2.0
        log_d_plus_I = 1 / log_d_plus_I
        log_d_plus_I_mat = sp.diags(log_d_plus_I)
        norm_adj = d_mat_inv.dot(adj_mat).dot(log_d_plus_I_mat)
        return norm_adj

    def adding_random_noise(self, user_emb, item_emb):
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        random_noise = (torch.rand_like(all_emb)).to(self.device)
        all_emb += self.eps * F.normalize(random_noise, dim=-1) * torch.sign(all_emb)
        use_noised_embeddings, item_noised_embeddings = torch.split(all_emb, [self.num_user, self.num_item])
        return use_noised_embeddings, item_noised_embeddings

    @staticmethod
    def calculate_ips_from_sparse_adjacency(r_sparse, num_users, num_items, clip_val=20):
        user_degrees = torch.sparse.sum(r_sparse, dim=1).to_dense()  # [num_users]
        item_degrees = torch.sparse.sum(r_sparse, dim=0).to_dense()  # [num_items]
        smoothing = 1.0
        user_probs = (user_degrees + smoothing) / (num_items + smoothing)
        item_probs = (item_degrees + smoothing) / (num_users + smoothing)
        user_ips = torch.sqrt(1.0 / (user_probs + 1e-8))
        item_ips = torch.sqrt(1.0 / (item_probs + 1e-8))
        user_ips = torch.clamp(user_ips, max=clip_val)
        item_ips = torch.clamp(item_ips, max=clip_val)
        user_ips = user_ips / user_ips.mean()
        item_ips = item_ips / item_ips.mean()
        return user_ips, item_ips

    def compute_ips_cl_loss(self, user_emb, item_emb, u_idx, i_idx):
        user_view_1, item_view_1 = self.adding_random_noise(user_emb, item_emb)
        user_view_2, item_view_2 = self.adding_random_noise(user_emb, item_emb)
        u_idx_unique, i_idx_unique = torch.unique(u_idx), torch.unique(i_idx)
        user_view_1_batch, item_view_1_batch = user_view_1[u_idx_unique], item_view_1[i_idx_unique]
        user_view_2_batch, item_view_2_batch = user_view_2[u_idx_unique], item_view_2[i_idx_unique]
        loss1 = self.infonce_with_ips(user_view_1_batch, user_view_2_batch, u_idx_unique, i_idx_unique, self.user_ips, self.item_ips, self.temperature, weight_mode='user_only')
        loss2 = self.infonce_with_ips(item_view_1_batch, item_view_2_batch, u_idx_unique, i_idx_unique, self.user_ips, self.item_ips, self.temperature, weight_mode='item_only')
        loss3 = self.infonce_with_ips(user_emb[u_idx], item_emb[i_idx], u_idx, i_idx, self.user_ips, self.item_ips, self.temperature, weight_mode='both')
        return (loss1 + loss2 + loss3  ) * self.ssl_reg

    @staticmethod
    def infonce_with_ips(view1, view2, user_ids, item_ids, user_ips, item_ips, temp=0.2, weight_mode='both'):
        view1_norm = F.normalize(view1, dim=1)
        view2_norm = F.normalize(view2, dim=1)
        sim_matrix = torch.matmul(view1_norm, view2_norm.T) / temp  # [batch_size, batch_size]
        batch_user_ips = user_ips[user_ids]  # [batch_size]
        batch_item_ips = item_ips[item_ids]  # [batch_size]
        if weight_mode == 'user_only':
            weights = batch_user_ips.view(-1, 1) * batch_user_ips.view(1, -1)  # [batch_size, batch_size]
        elif weight_mode == 'item_only':
            weights = batch_item_ips.view(-1, 1) * batch_item_ips.view(1, -1)
        else:
            weights = batch_user_ips.view(-1, 1) * batch_item_ips.view(1, -1)
        row_max, _ = torch.max(weights, dim=1, keepdim=True)  # [batch_size, 1]
        weights = weights / (row_max + 1e-8)
        pos_score = torch.diag(sim_matrix)  # [batch_size]
        exp_sim = torch.exp(sim_matrix)  # [batch_size, batch_size]
        exp_sim = exp_sim * weights
        diag_mask = ~torch.eye(view1.size(0), dtype=torch.bool, device=view1.device)
        neg_score = torch.sum(exp_sim * diag_mask, dim=1)  # [batch_size]
        pos_exp = torch.exp(pos_score)  # [batch_size]
        losses = -torch.log(pos_exp / (pos_exp + neg_score))  # [batch_size]
        return torch.mean(losses)
