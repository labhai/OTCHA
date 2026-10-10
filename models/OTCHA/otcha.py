import math
from typing import Any, Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class OTCHA(nn.Module):
    def __init__(self, opt, model_name: str, num_classes: int, n: int, pretrained_weights: bool = True, sinkhorn_iters: int = 5, sinkhorn_epsilon: float = 0.1, bin_score_init: float = 0.0,
                 edges: Optional[List[Tuple[int, int]]] = None, normalize_qk: bool = True, refine_dropout: float = 0.0, detach_transport: bool = False, ot_pos_weight: float = 0.2, ot_pos_sigma: float = 1.0, return_otcha_info: bool = True,
                 return_transport_matrix: bool = False, num_otcha_stages: int = 0):
        super().__init__()
        self.n = int(n)
        self.num_classes = int(num_classes)
        self.pretrained_weights = bool(pretrained_weights)

        # === backbone ===
        drop_rate = 0.0 if "tiny" in model_name else 0.1
        self.model = timm.create_model(model_name, pretrained=self.pretrained_weights, num_classes=self.num_classes, img_size=opt.image_size, drop_rate=drop_rate)

        for block in self.model.blocks:
            if hasattr(block, "attn") and hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False
            if hasattr(block, "attn") and hasattr(block.attn, "proj_drop"):
                block.attn.proj_drop = nn.Dropout(p=0.0)

        self.embed_dim = self.model.embed_dim

        self.img_embed_matrix = nn.Parameter(torch.zeros(1, self.n, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.img_embed_matrix)
        nn.init.zeros_(self.model.head.weight)
        nn.init.zeros_(self.model.head.bias)

        # === OT hyperparams ===
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        if self.sinkhorn_epsilon <= 0:
            raise ValueError("sinkhorn_epsilon must be > 0")

        self.normalize_qk = bool(normalize_qk)
        self.detach_transport = bool(detach_transport)

        # Fused OT
        self.ot_pos_weight = ot_pos_weight
        self.ot_pos_sigma = ot_pos_sigma
        if self.ot_pos_weight <= 0:
            raise ValueError("This OTCHA variant requires ot_pos_weight > 0 (geometry-enabled fused OT).")
        if self.ot_pos_sigma <= 0:
            raise ValueError("ot_pos_sigma must be > 0")

        # Learnable latent hub (M < S)
        self.num_latents = int(getattr(opt, "num_latents", 64))
        if self.num_latents <= 0:
            raise ValueError("num_latents must be > 0")
        self.latent_refine_iters = int(getattr(opt, "latent_refine_iters", 1))
        if self.latent_refine_iters < 1:
            raise ValueError("latent_refine_iters must be >= 1")

        # pre-block
        self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latents, self.embed_dim))
        nn.init.xavier_uniform_(self.latent_tokens)

        self.num_otcha_stages = int(num_otcha_stages)
        # mid-layer stage(num_otcha_stages > 0)
        if self.num_otcha_stages > 0:
            self.stage_latent_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, self.num_latents, self.embed_dim)) for _ in range(self.num_otcha_stages)])
            for p in self.stage_latent_tokens:
                nn.init.xavier_uniform_(p)
        
        # Learnable canonical latent positions in [-1,1]^2 (tanh-constrained at runtime)
        init_pos = self._init_latent_pos_grid(self.num_latents)
        self.latent_pos = nn.Parameter(init_pos)  # (1,M,2)

        # Token-conditional dustbin
        self.bin_score = nn.Parameter(torch.tensor(float(bin_score_init)))  # base dustbin logit
        hidden = max(32, self.embed_dim // 4)
        self.token_bin_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.token_bin_mlp[-1].weight)
        nn.init.zeros_(self.token_bin_mlp[-1].bias)

        # projections
        self.corr_norm = nn.LayerNorm(self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.msg_dropout = nn.Dropout(p=float(refine_dropout))

        # gate MLP: input = [target_token(D), conf_target(1), beta(1)]
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.embed_dim + 2, self.embed_dim // 4),
            nn.GELU(),
            nn.Linear(self.embed_dim // 4, 1),
            nn.Sigmoid(),
        )

        # output controls
        self.return_otcha_info = bool(return_otcha_info)
        self.return_transport_matrix = bool(return_transport_matrix)

        # === Multi-stage injection ===
        num_blocks = len(self.model.blocks)
        if self.num_otcha_stages > 0:
            indices = [int((i + 1) * num_blocks / (self.num_otcha_stages + 1)) - 1 for i in range(self.num_otcha_stages)]
            indices = [max(0, min(num_blocks - 1, idx)) for idx in indices]
            self._otcha_block_indices = set(indices)

            self.inject_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
            nn.init.zeros_(self.inject_proj.weight)
            self.inject_scales = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_otcha_stages)])

            print(
                f"[OTCHA-LATENT] before blocks + {self.num_otcha_stages} mid-layer stage(s) "
                f"at block indices {sorted(self._otcha_block_indices)} "
                f"(total {num_blocks} blocks)"
            )
        else:
            self._otcha_block_indices = set()
            print("[OTCHA-LATENT] Single-stage: latent-OTCHA before blocks only")

        self._pos_cache: Dict[int, torch.Tensor] = {}

        if edges is not None:
            print("[WARN] edges provided but ignored in latent-barycenter OTCHA.")

    # Utilities: multi-view token formatting
    def format_multi_image_tokens(self, x: torch.Tensor, batch_size: int, tokens_per_image: int) -> torch.Tensor:
        x = einops.rearrange(x, "(b n) s c -> b (n s) c", b=batch_size, n=self.n)

        first_img_token_idx = 0
        if self.model.cls_token is not None:
            for i in range(1, self.n):
                excess_cls_index = i * tokens_per_image + 1
                x = torch.cat((x[:, :excess_cls_index], x[:, excess_cls_index + 1 :]), dim=1)
            first_img_token_idx = 1

        image_embeddings = F.normalize(self.img_embed_matrix, dim=-1)
        x[:, first_img_token_idx:] += torch.repeat_interleave(image_embeddings, tokens_per_image, dim=1)
        return x

    # Patch grid positions
    def _get_patch_grid_size(self, tokens_per_image: int) -> Tuple[int, int]:
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "grid_size"):
            gs = self.model.patch_embed.grid_size
            if isinstance(gs, tuple) and len(gs) == 2:
                gh, gw = int(gs[0]), int(gs[1])
                if gh * gw == tokens_per_image:
                    return gh, gw
        gh = int(math.sqrt(tokens_per_image))
        gw = int(tokens_per_image / max(gh, 1))
        if gh * gw != tokens_per_image:
            return 1, tokens_per_image
        return gh, gw

    def _get_patch_pos(self, tokens_per_image: int, device: torch.device) -> torch.Tensor:
        # returns (1,S,2) in [-1,1]
        S = int(tokens_per_image)
        if S not in self._pos_cache:
            gh, gw = self._get_patch_grid_size(S)
            ys = torch.linspace(-1.0, 1.0, steps=gh)
            xs = torch.linspace(-1.0, 1.0, steps=gw)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            pos = torch.stack([grid_x, grid_y], dim=-1).view(1, gh * gw, 2).contiguous()  # (1,S,2)
            self._pos_cache[S] = pos  # CPU float32
        return self._pos_cache[S].to(device=device, dtype=torch.float32)

    @staticmethod
    def _init_latent_pos_grid(M: int) -> torch.Tensor:
        # grid init in [-1,1]^2, shape (1,M,2)
        gh = int(math.sqrt(M))
        gw = int(math.ceil(M / max(gh, 1)))
        ys = torch.linspace(-1.0, 1.0, steps=gh)
        xs = torch.linspace(-1.0, 1.0, steps=gw)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([grid_x, grid_y], dim=-1).view(1, gh * gw, 2).contiguous()
        return pos[:, :M, :].clone()

    # Sinkhorn (rectangular) with token-wise dustbin
    @staticmethod
    def _log_sinkhorn_iterations(log_alpha: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(log_alpha + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(log_alpha + u.unsqueeze(2), dim=1)
        return log_alpha + u.unsqueeze(2) + v.unsqueeze(1)

    def _compute_token_bin_logits(self, xu_n: torch.Tensor, y_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # base bin_score + token-wise delta
        base = self.bin_score.to(dtype=xu_n.dtype)
        bin_u = (base + self.token_bin_mlp(xu_n).squeeze(-1)).clamp(-10.0, 10.0)  # (B,S)
        bin_y = (base + self.token_bin_mlp(y_n).squeeze(-1)).clamp(-10.0, 10.0)   # (B,M)
        return bin_u, bin_y

    def _sinkhorn_rect_with_token_dustbin(self, scores: torch.Tensor, bin_u: torch.Tensor, bin_y: torch.Tensor) -> torch.Tensor:
        B, S, M = scores.shape
        dtype = scores.dtype
        base = self.bin_score.to(dtype=dtype).item()

        scores_aug = scores.new_full((B, S + 1, M + 1), base)
        scores_aug[:, :S, :M] = scores
        scores_aug[:, :S, -1] = bin_u.to(dtype=dtype)   # u -> dustbin column
        scores_aug[:, -1, :M] = bin_y.to(dtype=dtype)   # dustbin row -> y
        scores_aug[:, -1, -1] = base

        log_alpha = scores_aug / self.sinkhorn_epsilon

        mu = scores.new_ones(S + 1)
        nu = scores.new_ones(M + 1)
        mu[-1] = float(M)
        nu[-1] = float(S)

        log_mu = mu.log().unsqueeze(0).expand(B, -1)
        log_nu = nu.log().unsqueeze(0).expand(B, -1)

        log_P = self._log_sinkhorn_iterations(log_alpha, log_mu, log_nu, self.sinkhorn_iters)
        return log_P.exp()

    # View ↔ Latent transport (Proposal 1+3+4 core)
    def _view_to_latent_plan(self, x_view: torch.Tensor, y_lat: torch.Tensor, pos_view: torch.Tensor, pos_lat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        with torch.amp.autocast(device_type=x_view.device.type, enabled=False):
            x_f = x_view.float()
            y_f = y_lat.float()

            x_n = self.corr_norm(x_f)
            y_n = self.corr_norm(y_f)

            q = self.q_proj(x_n)
            k = self.k_proj(y_n)

            if self.normalize_qk:
                q = F.normalize(q, dim=-1)
                k = F.normalize(k, dim=-1)

            scores_feat = torch.matmul(q, k.transpose(1, 2))  # (B,S,M)

            # fused geometry score
            diff = pos_view.unsqueeze(2) - pos_lat.unsqueeze(1)         # (B,S,M,2)
            dist2 = (diff * diff).sum(dim=-1)                           # (B,S,M)
            score_geom = -dist2 / (2.0 * (self.ot_pos_sigma ** 2))      # (B,S,M)

            scores = scores_feat + (self.ot_pos_weight * score_geom)
    
            bin_u, bin_y = self._compute_token_bin_logits(x_n, y_n)
            P = self._sinkhorn_rect_with_token_dustbin(scores, bin_u=bin_u, bin_y=bin_y)

            if self.detach_transport:
                P = P.detach()

            match = P[:, : x_view.shape[1], : y_lat.shape[1]]  # (B,S,M)
            dustbin_col = P[:, : x_view.shape[1], -1]          # (B,S)

            # masses/confidences
            conf_view = (1.0 - dustbin_col).clamp(0.0, 1.0)     # (B,S)
            conf_lat = match.sum(dim=1).clamp(0.0, 1.0)         # (B,M)
            beta = conf_view.mean(dim=1).clamp(0.0, 1.0)        # (B,)

        P_out = P.to(dtype=x_view.dtype) if self.return_transport_matrix else None
        return match.to(dtype=x_view.dtype), conf_lat.to(dtype=x_view.dtype), conf_view.to(dtype=x_view.dtype), beta.to(dtype=x_view.dtype), P_out

    # Latent-barycenter refinement (pre-block)
    def _latent_refine_tokens(self, tokens: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        n = self.n
        Bn, T, D = tokens.shape
        if Bn != batch_size * n:
            raise ValueError(f"Bn mismatch: got {Bn}, expected {batch_size*n}")

        tokens_bn = einops.rearrange(tokens, "(b n) t d -> b n t d", b=batch_size, n=n)

        has_cls = self.model.cls_token is not None
        if has_cls:
            cls_tok = tokens_bn[:, :, :1, :]          # (B,N,1,D)
            patch_tok = tokens_bn[:, :, 1:, :]        # (B,N,S,D)
        else:
            cls_tok = None
            patch_tok = tokens_bn                     # (B,N,S,D)

        B, N, S, D = patch_tok.shape
        device = patch_tok.device

        pos_view = self._get_patch_pos(S, device=device).expand(B, -1, -1)  # (B,S,2)
        pos_lat = torch.tanh(self.latent_pos).to(device=device, dtype=torch.float32).expand(B, -1, -1)  # (B,M,2)

        lat = self.latent_tokens.to(device=device, dtype=patch_tok.dtype).expand(B, -1, -1)  # (B,M,D)

        # ---- 1) refine latents by aggregating from views ----
        for _ in range(self.latent_refine_iters):
            agg_lat = patch_tok.new_zeros((B, self.num_latents, D))

            for v in range(N):
                match, conf_lat, conf_view, beta, _ = self._view_to_latent_plan(x_view=patch_tok[:, v], y_lat=lat, pos_view=pos_view, pos_lat=pos_lat)

                # message to latents: (B,M,D)
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    x_values = self.v_proj(patch_tok[:, v].float())
                    msg_lat = torch.matmul(match.transpose(1, 2), x_values)                 # (B,M,D)
                    msg_lat = msg_lat / (conf_lat.float().unsqueeze(-1) + 1e-8)             # avg
                msg_lat = msg_lat.to(dtype=patch_tok.dtype)

                beta_expand = beta.view(B, 1, 1).expand(-1, self.num_latents, -1)
                gate_in = torch.cat([lat, conf_lat.unsqueeze(-1), beta_expand], dim=-1)
                gate = self.gate_mlp(gate_in)  # (B,M,1)

                weighted = msg_lat * gate * conf_lat.unsqueeze(-1)
                agg_lat = agg_lat + weighted

            agg_lat = agg_lat / float(N)
            agg_lat = self.msg_dropout(agg_lat)
            lat = lat + self.out_proj(agg_lat)

        # ---- 2) project refined latents back to each view ----
        refined_patch = patch_tok.new_zeros((B, N, S, D))

        # otcha_info (latent->view edges only): E=N
        edges = [(N, v) for v in range(N)]
        E = len(edges)
        edges_tensor = torch.tensor(edges, device=device, dtype=torch.long)

        gates = patch_tok.new_zeros((B, E, S))
        messages = patch_tok.new_zeros((B, E, S, D))
        confs_v = patch_tok.new_zeros((B, E, S))
        confs_u = patch_tok.new_zeros((B, E, S))
        betas = patch_tok.new_zeros((B, E))

        if self.return_transport_matrix:
            transports = patch_tok.new_zeros((B, E, S + 1, self.num_latents + 1))
        else:
            transports = None

        for v in range(N):
            e_idx = v
            match, conf_lat, conf_view, beta, P = self._view_to_latent_plan(x_view=patch_tok[:, v], y_lat=lat, pos_view=pos_view, pos_lat=pos_lat)

            with torch.amp.autocast(device_type=device.type, enabled=False):
                y_values = self.v_proj(lat.float())  # (B,M,D)
                msg_view = torch.matmul(match, y_values)                                # (B,S,D)
                msg_view = msg_view / (conf_view.float().unsqueeze(-1) + 1e-8)          # avg
            msg_view = msg_view.to(dtype=patch_tok.dtype)

            with torch.amp.autocast(device_type=device.type, enabled=False):
                conf_u_view = torch.matmul(match, conf_lat.float().unsqueeze(-1)).squeeze(-1)  # (B,S)
                conf_u_view = conf_u_view / (conf_view.float() + 1e-8)
            conf_u_view = conf_u_view.to(dtype=patch_tok.dtype).clamp(0.0, 1.0)

            beta_expand = beta.view(B, 1, 1).expand(-1, S, -1)
            gate_in = torch.cat([patch_tok[:, v], conf_view.unsqueeze(-1), beta_expand], dim=-1)
            gate = self.gate_mlp(gate_in)  # (B,S,1)

            weighted = msg_view * gate * conf_view.unsqueeze(-1)
            delta = self.out_proj(self.msg_dropout(weighted))
            refined_patch[:, v] = patch_tok[:, v] + delta

            gates[:, e_idx] = gate.squeeze(-1)
            messages[:, e_idx] = msg_view
            confs_v[:, e_idx] = conf_view
            confs_u[:, e_idx] = conf_u_view
            betas[:, e_idx] = beta
            if transports is not None and P is not None:
                transports[:, e_idx] = P

        if has_cls:
            tokens_refined_bn = torch.cat([cls_tok, refined_patch], dim=2)
        else:
            tokens_refined_bn = refined_patch

        refined_tokens = einops.rearrange(tokens_refined_bn, "b n t d -> (b n) t d")

        otcha_info: Dict[str, Any] = {
            "edges": edges_tensor,
            "tokens": patch_tok,
            "tokens_refined": refined_patch,
            "latent_tokens": lat,  # (B,M,D)
            "gate": gates,
            "messages": messages,
            "conf_v": confs_v,
            "conf_u": confs_u,
            "beta": betas,
        }
        if transports is not None:
            otcha_info["P"] = transports

        return refined_tokens, otcha_info

    # Latent-barycenter refinement (mid-layer, teacher-guided)
    def _latent_refine_fused_tokens_teacher_guided(self, fused_tokens: torch.Tensor, teacher_tokens: torch.Tensor, batch_size: int, tokens_per_image: int, stage_idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        has_cls = self.model.cls_token is not None
        B = batch_size
        N = self.n
        D = fused_tokens.shape[-1]

        if has_cls:
            cls_tok = fused_tokens[:, :1, :]
            patch_flat = fused_tokens[:, 1:, :]
        else:
            cls_tok = None
            patch_flat = fused_tokens

        S = int(tokens_per_image)
        expected = N * S
        if patch_flat.shape[1] != expected:
            raise ValueError(f"[LATENT-OTCHA] fused token length mismatch: got {patch_flat.shape[1]}, expected {expected}")

        student_patch = patch_flat.view(B, N, S, D)

        teacher_bn = einops.rearrange(teacher_tokens, "(b n) t d -> b n t d", b=B, n=N)
        if has_cls:
            teacher_patch = teacher_bn[:, :, 1:, :]  # (B,N,S,D)
        else:
            teacher_patch = teacher_bn

        device = fused_tokens.device
        dtype = fused_tokens.dtype

        # positions
        pos_view = self._get_patch_pos(S, device=device).expand(B, -1, -1)  # (B,S,2)
        pos_lat = torch.tanh(self.latent_pos).to(device=device, dtype=torch.float32).expand(B, -1, -1)

        base_lat = self.stage_latent_tokens[stage_idx].to(device=device, dtype=dtype).expand(B, -1, -1)

        # 1) teacher plans (fixed)
        match_list: List[torch.Tensor] = []
        conf_lat_list: List[torch.Tensor] = []
        conf_view_list: List[torch.Tensor] = []
        beta_list: List[torch.Tensor] = []
        P_list: List[Optional[torch.Tensor]] = []

        for v in range(N):
            match, conf_lat, conf_view, beta, P = self._view_to_latent_plan(x_view=teacher_patch[:, v], y_lat=base_lat, pos_view=pos_view, pos_lat=pos_lat)
            match_list.append(match)
            conf_lat_list.append(conf_lat)
            conf_view_list.append(conf_view)
            beta_list.append(beta)
            P_list.append(P)

        # 2) aggregate student values into latents using teacher plans
        lat = base_lat
        for _ in range(self.latent_refine_iters):
            agg_lat = fused_tokens.new_zeros((B, self.num_latents, D))
            for v in range(N):
                match = match_list[v]
                conf_lat = conf_lat_list[v]
                beta = beta_list[v]

                with torch.amp.autocast(device_type=device.type, enabled=False):
                    x_values = self.v_proj(student_patch[:, v].float())
                    msg_lat = torch.matmul(match.transpose(1, 2), x_values)
                    msg_lat = msg_lat / (conf_lat.float().unsqueeze(-1) + 1e-8)
                msg_lat = msg_lat.to(dtype=dtype)

                beta_expand = beta.view(B, 1, 1).expand(-1, self.num_latents, -1)
                gate_in = torch.cat([lat, conf_lat.unsqueeze(-1), beta_expand], dim=-1)
                gate = self.gate_mlp(gate_in)

                agg_lat = agg_lat + (msg_lat * gate * conf_lat.unsqueeze(-1))

            agg_lat = agg_lat / float(N)
            agg_lat = self.msg_dropout(agg_lat)
            lat = lat + self.out_proj(agg_lat)

        # 3) project back to student patches and inject
        agg_patch = fused_tokens.new_zeros((B, N, S, D))

        edges = [(N, v) for v in range(N)]
        E = len(edges)
        edges_tensor = torch.tensor(edges, device=device, dtype=torch.long)

        gates = fused_tokens.new_zeros((B, E, S))
        messages = fused_tokens.new_zeros((B, E, S, D))
        confs_v = fused_tokens.new_zeros((B, E, S))
        confs_u = fused_tokens.new_zeros((B, E, S))
        betas = fused_tokens.new_zeros((B, E))

        if self.return_transport_matrix:
            transports = fused_tokens.new_zeros((B, E, S + 1, self.num_latents + 1))
        else:
            transports = None

        with torch.amp.autocast(device_type=device.type, enabled=False):
            y_values = self.v_proj(lat.float())  # (B,M,D)
        y_values = y_values.to(dtype=dtype)

        for v in range(N):
            e_idx = v
            match = match_list[v]
            conf_lat = conf_lat_list[v]
            conf_view = conf_view_list[v]
            beta = beta_list[v]
            P = P_list[v]

            with torch.amp.autocast(device_type=device.type, enabled=False):
                msg_view = torch.matmul(match, y_values.float())
                msg_view = msg_view / (conf_view.float().unsqueeze(-1) + 1e-8)
            msg_view = msg_view.to(dtype=dtype)

            with torch.amp.autocast(device_type=device.type, enabled=False):
                conf_u_view = torch.matmul(match, conf_lat.float().unsqueeze(-1)).squeeze(-1)
                conf_u_view = conf_u_view / (conf_view.float() + 1e-8)
            conf_u_view = conf_u_view.to(dtype=dtype).clamp(0.0, 1.0)

            beta_expand = beta.view(B, 1, 1).expand(-1, S, -1)
            gate_in = torch.cat([student_patch[:, v], conf_view.unsqueeze(-1), beta_expand], dim=-1)
            gate = self.gate_mlp(gate_in)

            weighted = msg_view * gate * conf_view.unsqueeze(-1)
            agg_patch[:, v] = weighted

            gates[:, e_idx] = gate.squeeze(-1)
            messages[:, e_idx] = msg_view
            confs_v[:, e_idx] = conf_view
            confs_u[:, e_idx] = conf_u_view
            betas[:, e_idx] = beta
            if transports is not None and P is not None:
                transports[:, e_idx] = P

        agg_patch = self.msg_dropout(agg_patch)

        alpha = self.inject_scales[stage_idx].sigmoid()
        delta = self.inject_proj(agg_patch)
        refined_patch = student_patch + alpha * delta

        patch_out = refined_patch.reshape(B, N * S, D)
        if has_cls:
            fused_out = torch.cat([cls_tok, patch_out], dim=1)
        else:
            fused_out = patch_out

        otcha_info: Dict[str, Any] = {
            "stage_idx": int(stage_idx),
            "edges": edges_tensor,
            "tokens": student_patch,
            "tokens_refined": refined_patch,
            "latent_tokens": lat,
            "gate": gates,
            "messages": messages,
            "conf_v": confs_v,
            "conf_u": confs_u,
            "beta": betas,
        }
        if transports is not None:
            otcha_info["P"] = transports

        return fused_out, otcha_info

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        batch_size = x.shape[0]
        if x.shape[1] != self.n:
            raise ValueError(f"Input has n={x.shape[1]} views, but model initialized with n={self.n}")

        output_dict: Dict[str, Any] = {"single": {}}
        if self.n > 1:
            output_dict["mv_collection"] = {}

        x = einops.rearrange(x, "b n c h w -> (b n) c h w")
        x = self.model.patch_embed(x)
        tokens_per_image = x.shape[1]  # number of patch tokens (no cls)
        x = self.model._pos_embed(x)   # adds cls if used

        otcha_infos: List[Dict[str, Any]] = []

        # ---- single-view only ----
        if self.n == 1:
            tokens = self.model.blocks(x)
            tokens = self.model.norm(tokens)
            output_dict["single"]["logits"] = self.model.forward_head(tokens)

        # ---- multi-view, no mid-layer stages ----
        elif self.num_otcha_stages == 0:
            single_tokens = self.model.blocks(x.clone())
            single_tokens = self.model.norm(single_tokens)
            output_dict["single"]["logits"] = self.model.forward_head(single_tokens)

            mv_tokens, si = self._latent_refine_tokens(x.clone(), batch_size)
            otcha_infos.append(si)

            mv_tokens = self.format_multi_image_tokens(mv_tokens, batch_size, tokens_per_image)
            mv_tokens = self.model.blocks(mv_tokens)
            mv_tokens = self.model.norm(mv_tokens)
            output_dict["mv_collection"]["logits"] = self.model.forward_head(mv_tokens)

        # ---- multi-view, with mid-layer stages ----
        else:
            single_tokens = x.clone()
            mv_tokens = x.clone()

            mv_tokens, si = self._latent_refine_tokens(mv_tokens, batch_size)
            otcha_infos.append(si)

            mv_tokens = self.format_multi_image_tokens(mv_tokens, batch_size, tokens_per_image)

            stage_counter = 0
            for b_idx, block in enumerate(self.model.blocks):
                single_tokens = block(single_tokens)
                mv_tokens = block(mv_tokens)

                if b_idx in self._otcha_block_indices:
                    mv_tokens, si = self._latent_refine_fused_tokens_teacher_guided(fused_tokens=mv_tokens, teacher_tokens=single_tokens.detach(), batch_size=batch_size, tokens_per_image=tokens_per_image, stage_idx=stage_counter)
                    otcha_infos.append(si)
                    stage_counter += 1

            if stage_counter != self.num_otcha_stages:
                print(
                    f"[WARN] stage_counter={stage_counter}, expected={self.num_otcha_stages}. "
                    f"indices={sorted(self._otcha_block_indices)}"
                )

            single_tokens = self.model.norm(single_tokens)
            output_dict["single"]["logits"] = self.model.forward_head(single_tokens)

            mv_tokens = self.model.norm(mv_tokens)
            output_dict["mv_collection"]["logits"] = self.model.forward_head(mv_tokens)

        if self.return_otcha_info and len(otcha_infos) > 0:
            output_dict["otcha"] = otcha_infos[0]
            output_dict["otcha_stages"] = otcha_infos

        return output_dict
