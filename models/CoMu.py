import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import TuckERLayer
from .model import BaseModel

    
class ModalFusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim):
        super(ModalFusionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.multi = multi
        self.img_dim = img_dim
        self.text_dim = txt_dim

        modal1 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(in_dim, out_dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.img_dim, out_dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.text_dim, out_dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)

    def forward(self, modal1_emb, modal2_emb, modal3_emb):
        batch_size = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)
            attention_scores = self.ent_attn(x_stack).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)
            x_mm.append(context_vectors)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        return x_mm, attention_weights
    


class ModalFusion_CF(nn.Module):
    """
    Counterfactual-aware Multimodal Fusion for MMKGC (STRONG version):
      - Prediction-level counterfactual effects (JS divergence, bounded & stable)
      - Warmup blending for causal weights (alpha schedule passed from training loop)
      - Dual temperature:
          * cl_temperature: InfoNCE temperature
          * causal_temperature: causal softmax temperature (should be larger, e.g., 0.8~2.0)
      - Causal weights modulate contrastive loss at sample-level (reweight CE), NOT logits scaling.
      - Causal weights are detached for CL to avoid destabilizing feedback.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        img_dim,
        txt_dim,
        cl_temperature=0.1,
        causal_temperature=1.0,
        lambda_cf=1.0,
        weight_clip=(0.2, 5.0),
        eps=1e-9,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.img_dim = img_dim
        self.txt_dim = txt_dim

        self.cl_temperature = cl_temperature
        self.causal_temperature = causal_temperature
        self.lambda_cf = lambda_cf

        self.weight_clip = weight_clip
        self.eps = eps

        # ===== Modal projection =====
        self.modal1 = nn.Linear(in_dim, out_dim)   # structure
        self.modal2 = nn.Linear(img_dim, out_dim)  # image
        self.modal3 = nn.Linear(txt_dim, out_dim)  # text

        # (Optional) keep for interface parity; not used in this strong pred-driven variant
        self.ent_attn = nn.Linear(out_dim, 1, bias=False)
        self.ent_attn.requires_grad_(True)

    # ------------------------------------------------------------------
    # JS divergence: bounded & stable alternative to KL
    # ------------------------------------------------------------------
    def js_div(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        p, q: [B, C] distributions (sum to 1)
        return: [B] JS divergence
        """
        eps = self.eps
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        m = 0.5 * (p + q)

        # KL(p||m)
        kl_pm = (p * (p.log() - m.log())).sum(dim=1)
        # KL(q||m)
        kl_qm = (q * (q.log() - m.log())).sum(dim=1)

        return 0.5 * (kl_pm + kl_qm)

    # ----------------------------------------------------------------
    # Stable, KG-aware InfoNCE with sample-level reweighting
    # ----------------------------------------------------------------
    def contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        heads: torch.Tensor,
        temperature: float,
        sample_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        anchor, positive: [B, D]
        heads: [B]
        sample_weight: [B] (detached outside), reweights per-sample CE
        """
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)

        logits = torch.matmul(anchor, positive.T) / temperature  # [B, B]
        B = logits.size(0)
        device = logits.device

        # KG-aware masking (same head => invalidate except diagonal)
        heads_i = heads.view(-1, 1).expand(B, B)
        heads_j = heads.view(1, -1).expand(B, B)
        eye = torch.eye(B, dtype=torch.bool, device=device)

        valid_mask = (heads_i != heads_j) | eye
        logits = logits.masked_fill(~valid_mask, -1e9)

        labels = torch.arange(B, device=device)

        # per-sample CE
        loss_vec = F.cross_entropy(logits, labels, reduction="none")  # [B]

        if sample_weight is not None:
            w = sample_weight.detach()

            # normalize to keep mean loss scale stable
            w = w / (w.mean().clamp_min(self.eps))

            # clip to avoid extreme gradients
            w = w.clamp(self.weight_clip[0], self.weight_clip[1])

            loss_vec = loss_vec * w

        return loss_vec.mean()

    # ------------------------------------------------------------------
    # Counterfactual invariance loss (cosine)
    # ------------------------------------------------------------------
    # def counterfactual_invariance_loss(self, fused: torch.Tensor, fused_cf: torch.Tensor) -> torch.Tensor:
    #     return 1.0 - F.cosine_similarity(fused, fused_cf, dim=1).mean()
    
    # def counterfactual_invariance_loss(self, fused: torch.Tensor, fused_cf: torch.Tensor, margin_low: float = 0.6, margin_high: float = 0.9): 
    #     cos_sim = F.cosine_similarity(fused, fused_cf, dim=1)

    #     loss_low = F.relu(margin_low - cos_sim)
    #     loss_high = F.relu(cos_sim - margin_high)

    #     return (loss_low + loss_high).mean()
    
    def counterfactual_invariance_loss(self, fused: torch.Tensor, fused_cf: torch.Tensor, margin: float = 0.95): 
        cos_sim = F.cosine_similarity(fused, fused_cf, dim=1)
        loss_low = F.relu(margin - cos_sim)
        return loss_low.mean()

    
    
    # ------------------------------------------------------------------
    # Utility: fuse distributions with weights
    # ------------------------------------------------------------------
    def fuse_probs(self, probs_stack: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        probs_stack: [B, 3, C]
        weights:     [B, 3]
        return:      [B, C]
        """
        return torch.sum(weights.unsqueeze(-1) * probs_stack, dim=1)

    # ------------------------------------------------------------------
    # Prediction-driven causal weights via counterfactual JS effects
    # ------------------------------------------------------------------
    def compute_pred_driven_weights(
        self,
        logits_s: torch.Tensor,
        logits_i: torch.Tensor,
        logits_t: torch.Tensor,
        alpha: float = 1.0,
    ):
        """
        Returns:
          conf_weights:  [B, 3]  (factual confidence weights)
          causal_weights:[B, 3]  (effect-based weights, warmup blended)
          fused_pred:    [B, C]  (factual fused prediction distribution)
        """
        eps = self.eps
        T_causal = self.causal_temperature

        # per-modality distributions
        prob_s = F.softmax(logits_s, dim=1)
        prob_i = F.softmax(logits_i, dim=1)
        prob_t = F.softmax(logits_t, dim=1)
        probs_stack = torch.stack((prob_s, prob_i, prob_t), dim=1)  # [B, 3, C]

        # ---- confidence weights: negative entropy (can be swapped if desired) ----
        ent_s = -(prob_s * prob_s.clamp_min(eps).log()).sum(dim=1)
        ent_i = -(prob_i * prob_i.clamp_min(eps).log()).sum(dim=1)
        ent_t = -(prob_t * prob_t.clamp_min(eps).log()).sum(dim=1)

        conf_scores = torch.stack((-ent_s, -ent_i, -ent_t), dim=1)  # higher => more confident
        conf_weights = torch.softmax(conf_scores / T_causal, dim=1)  # [B, 3]

        # factual fused prediction
        fused_pred = self.fuse_probs(probs_stack, conf_weights)  # [B, C]

        # ---- counterfactual effects: remove modality m, renormalize weights, compare with fused_pred ----
        effects = []
        for m in range(3):
            mask = torch.ones_like(conf_weights)
            mask[:, m] = 0.0

            denom = (conf_weights * mask).sum(dim=1, keepdim=True).clamp_min(eps)
            masked_w = (conf_weights * mask) / denom

            fused_without = self.fuse_probs(probs_stack, masked_w)  # [B, C]

            # bounded effect: JS(fused_without, fused_pred)
            eff = self.js_div(fused_without, fused_pred.detach())   # detach target to avoid feedback loops
            effects.append(eff)

        causal_effects = torch.stack(effects, dim=1)  # [B, 3]
        effects_sum = causal_effects.sum(dim=1, keepdim=True).clamp_min(eps)
        causal_weights_raw = causal_effects / effects_sum

        # ---- warmup blend ----
        # alpha in [0,1]: 0 => uniform, 1 => full causal weights
        alpha = float(alpha)
        uniform = torch.full_like(causal_weights_raw, 1.0 / 3.0)
        causal_weights = (1.0 - alpha) * uniform + alpha * causal_weights_raw

        return conf_weights, causal_weights, fused_pred

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        modal1_emb,
        modal2_emb,
        modal3_emb,
        rel_emb,     # kept for interface parity
        head,
        logits_s,
        logits_i,
        logits_t,
        alpha: float = 1.0,   # warmup blending factor, set from training loop
    ):
        """
        Inputs:
          modal*_emb: [B, *]
          logits_*:   [B, C] per-modality logits (before softmax)
          alpha: warmup factor in [0,1]
        Returns:
          fused_cf:        [B, D]
          causal_weights:  [B, 3]
          contrastive_loss scalar
          cf_inv_loss      scalar
        """

        # ===== 1) project modality embeddings =====
        z_s = torch.tanh(self.modal1(modal1_emb))  # [B, D]
        z_i = torch.tanh(self.modal2(modal2_emb))  # [B, D]
        z_t = torch.tanh(self.modal3(modal3_emb))  # [B, D]
        z_stack = torch.stack((z_s, z_i, z_t), dim=1)  # [B, 3, D]

        # ===== 2) prediction-driven conf & causal weights (JS + warmup + dual temp) =====
        conf_weights, causal_weights, fused_pred = self.compute_pred_driven_weights(
            logits_s=logits_s,
            logits_i=logits_i,
            logits_t=logits_t,
            alpha=alpha,
        )

        # ===== 3) fuse embeddings using both weights =====
        fused_factual = torch.sum(conf_weights.unsqueeze(-1) * z_stack, dim=1)   # [B, D]
        fused_cf = torch.sum(causal_weights.unsqueeze(-1) * z_stack, dim=1)      # [B, D]

        # ===== 4) causal-weighted contrastive learning (sample-level loss reweight) =====
        w_s = causal_weights[:, 0].detach()
        w_i = causal_weights[:, 1].detach()
        w_t = causal_weights[:, 2].detach()

        cl_s = self.contrastive_loss(fused_cf, z_s, head, self.cl_temperature, sample_weight=w_s)
        cl_i = self.contrastive_loss(fused_cf, z_i, head, self.cl_temperature, sample_weight=w_i)
        cl_t = self.contrastive_loss(fused_cf, z_t, head, self.cl_temperature, sample_weight=w_t)

        contrastive_loss = (cl_s + cl_i + cl_t) / 3.0

        # ===== 5) counterfactual invariance regularization =====
        # cf_inv_loss = self.counterfactual_invariance_loss(fused_factual, fused_cf)
        # cf_inv_loss = self.counterfactual_invariance_loss(fused_factual, fused_cf, margin_low=0.6, margin_high=0.9)
        cf_inv_loss = self.counterfactual_invariance_loss(fused_factual, fused_cf, margin=0.95)

        return fused_cf, fused_factual, causal_weights, conf_weights, contrastive_loss, cf_inv_loss



class CoMu(BaseModel):
    def __init__(self, args):
        super(CoMu, self).__init__(args)
        self.entity_embeddings = nn.Embedding(
            len(args.entity2id),
            args.dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.entity_embeddings.weight)

        self.relation_embeddings = nn.Embedding(
            2 * len(args.relation2id), 
            args.r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        if args.pre_trained:
            self.entity_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)
            self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)

        self.rel_gate = nn.Embedding(2 * len(args.relation2id), 1, padding_idx=None)

        img = args.img.to(self.device)
        txt = args.desp.to(self.device)

        

        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=True)
        self.img_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.img_relation_embeddings.weight)
        self.att_relation_fc = nn.Linear(args.r_dim, 1, bias=False)
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=True)
        self.txt_relation_embeddings = nn.Embedding(
            2 * len(args.relation2id),
            args.r_dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.txt_relation_embeddings.weight)
        
        
        self.alpha = None
        
        self.dim = args.dim
        self.img_dim = self.img_entity_embeddings.weight.data.shape[1]
        self.txt_dim = self.txt_entity_embeddings.weight.data.shape[1]     
        self.fuse_out_dim = 200
        self.img_dim_ib = self.img_dim 
        self.txt_dim_ib = self.txt_dim  


        self.TuckER_S = TuckERLayer(self.dim, args.r_dim)
        self.TuckER_I = TuckERLayer(self.dim, args.r_dim)
        self.TuckER_D = TuckERLayer(self.dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(self.fuse_out_dim, self.fuse_out_dim)
        self.fuse_e = ModalFusionLayer(
            in_dim=self.dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=self.dim,
            txt_dim=self.dim
        )
        
        self.fuse_r = ModalFusionLayer(
            in_dim=args.r_dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=args.r_dim,
            txt_dim=args.r_dim
        )
        

        self.bceloss = nn.BCELoss()
       
        self.cl_loss = None
        self.cf_inv_loss = None

        self.Linear_s = nn.Sequential(
            nn.Linear(self.dim, (self.dim+self.dim)//2),
            nn.ReLU(),
            nn.Linear((self.dim+self.dim)//2, self.dim)
        )

        self.Linear_i = nn.Sequential(    
            nn.Linear(self.img_dim, (self.img_dim+self.dim)//2),
            nn.ReLU(),
            nn.Linear((self.img_dim+self.dim)//2, self.dim)
        )

        self.Linear_t = nn.Sequential(    
            nn.Linear(self.txt_dim, (self.txt_dim+self.dim)//2),
            nn.ReLU(),
            nn.Linear((self.txt_dim+self.dim)//2, self.dim)
        )


        self.counterfactualfusion = ModalFusion_CF(
            in_dim=self.dim,
            out_dim=self.fuse_out_dim,
            img_dim=self.dim,
            txt_dim=self.dim,
        )
        
    
        
        
    def forward(self, batch_inputs, return_weights=False, return_extra=False):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        rel_gate = self.rel_gate(relation)
        training = self.training
    
        e_embed = self.Linear_s(self.entity_embeddings(head))
        r_embed = self.relation_embeddings(relation)

        e_img_embed = self.Linear_i(self.img_entity_embeddings(head))
        r_img_embed = self.img_relation_embeddings(relation)

        e_txt_embed = self.Linear_t(self.txt_entity_embeddings(head))    
        r_txt_embed = self.txt_relation_embeddings(relation)
        
        e_img_aligned, e_txt_aligned = e_img_embed, e_txt_embed
        



        p_s = self.TuckER_S(e_embed, r_embed)
        p_i = self.TuckER_I(e_img_aligned, r_img_embed)
        p_d = self.TuckER_D(e_txt_aligned, r_txt_embed)
        
        
        all_s = self.Linear_s(self.entity_embeddings.weight)
        all_v = self.Linear_i(self.img_entity_embeddings.weight)
        all_t = self.Linear_t(self.txt_entity_embeddings.weight)
        
        all_v_aligned, all_t_aligned = all_v, all_t

        
        logits_s = torch.mm(p_s, all_s.transpose(1, 0))
        logits_i = torch.mm(p_i, all_v_aligned.transpose(1, 0))
        logits_t = torch.mm(p_d, all_t_aligned.transpose(1, 0))
        
        e_mm_embed, fused_factual, attn_cf, conf_weights, cl_loss, cf_inv_loss  = self.counterfactualfusion(
            e_embed,
            e_img_aligned,
            e_txt_aligned,
            rel_gate,
            head,
            logits_s,
            logits_i,
            logits_t,
            alpha=self.alpha
        )
        
        self.cl_loss = cl_loss
        self.cf_inv_loss = cf_inv_loss
        
        r_mm_embed, _ = self.fuse_r(r_embed, r_img_embed, r_txt_embed)

        p_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)
        p_mm_conf = self.TuckER_MM(fused_factual, r_mm_embed)
        
        all_f, _ = self.fuse_e(all_s, all_v_aligned, all_t_aligned)

        logits_mm = torch.mm(p_mm, all_f.transpose(1, 0))
        logits_mm_conf = torch.mm(p_mm_conf, all_f.transpose(1, 0))

        pred_s = F.softmax(logits_s, dim=1)
        pred_i = F.softmax(logits_i, dim=1)
        pred_d = F.softmax(logits_t, dim=1)
        pred_mm = F.softmax(logits_mm, dim=1)
        pred_mm_conf = F.softmax(logits_mm_conf, dim=1)

        if return_extra:
            return {
                "pred_s": pred_s,
                "pred_i": pred_i,
                "pred_t": pred_d,
                "pred_mm_causal": pred_mm,
                "pred_mm_conf": pred_mm_conf,
                "pred_joint": pred_s + pred_i + pred_d,
                "causal_weights": attn_cf,
                "conf_weights": conf_weights
            }
        if return_weights:
            return [pred_s, pred_i, pred_d, pred_mm], attn_cf, conf_weights, pred_mm_conf
        if not self.training:
            return [pred_s, pred_i, pred_d, pred_mm], attn_cf
        else:
            return [pred_s, pred_i, pred_d, pred_mm], attn_cf
           


    def loss_func(self, output, target):
        loss_s = self.bceloss(output[0], target)
        loss_i = self.bceloss(output[1], target)
        loss_d = self.bceloss(output[2], target)
        loss_mm = self.bceloss(output[3], target)  

        return loss_s, loss_i, loss_d, loss_mm, self.cl_loss, self.cf_inv_loss
    
    

 
 
