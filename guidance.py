# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
from typing import Any, Dict, Tuple
from gdf import BaseSchedule, DDIMSampler, DDPMSampler, GDF, SimpleSampler
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnf
import torchvision.transforms as T
from train import WurstCoreC
from utils import Style_Storage, setup_csd


transform = T.ToPILImage()


def setup_dino(device: str = "cpu") -> torch.nn.Module:
  """Sets up the DINO ViT model for subject extraction (rho)."""
  hub_dir = torch.hub.get_dir()
  dino_dir = os.path.join(hub_dir, "facebookresearch_dino_main")
  if not os.path.exists(dino_dir):
    dino_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dino_main")
  sys_path_bak = list(sys.path)
  try:
    if dino_dir in sys.path:
      sys.path.remove(dino_dir)
    sys.path.insert(0, dino_dir)
    import vision_transformer as vits
    model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
    url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
  finally:
    sys.path = sys_path_bak
  return model


def dino_preprocess(x: torch.Tensor) -> torch.Tensor:
  """Preprocesses image tensor for DINO ViT."""
  x_res = nnf.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
  mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
  return (x_res - mean) / std


def compute_ssm(features: torch.Tensor) -> torch.Tensor:
  """Computes pairwise cosine self-similarity matrix for feature tokens."""
  norm_features = nnf.normalize(features, p=2, dim=-1)
  ssm = torch.bmm(norm_features, norm_features.transpose(1, 2))
  return ssm


def get_dino_keys(dino_model: torch.nn.Module, img: torch.Tensor, layer: int = 11) -> torch.Tensor:
  """Extracts Key (K) projection matrix from a specific Self-Attention block of DINO ViT."""
  x = dino_model.prepare_tokens(dino_preprocess(img))
  for idx, blk in enumerate(dino_model.blocks):
    if idx == layer:
      b, n, c = x.shape
      qkv = blk.attn.qkv(blk.norm1(x)).reshape(b, n, 3, blk.attn.num_heads, c // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
      k = qkv[1]  # shape: [B, num_heads, N, head_dim]
      k = k.permute(0, 2, 1, 3).reshape(b, n, c)  # shape: [B, N, C]
      return k
    x = blk(x)
  return x


class OrthoGuidance(GDF):
  """Implementation of OrthoStyle Score-Orthogonal Guidance and sampling loop.
  """

  def diffuse_forward(
      self,
      x0: torch.Tensor,
      logSNR: torch.Tensor,
      epsilon: torch.Tensor = None,
      device: str = None,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward noising function for reference latents."""
    if device is None:
      device = x0.device
    if epsilon is None:
      epsilon = torch.randn_like(x0).to(device)
    a, b = self.input_scaler(logSNR)
    if len(a.shape) == 1:
      a, b = (
          a.view(-1, *[1] * (len(x0.shape) - 1)),
          b.view(-1, *[1] * (len(x0.shape) - 1)),
      )
    xt = x0 * a + epsilon * b
    return xt, epsilon, a, b

  def sample(
      self,
      model: torch.nn.Module,
      model_inputs: Dict[str, Any],
      shape: Tuple,
      unconditional_inputs: Dict[str, Any] = None,
      sampler: SimpleSampler = None,
      schedule: BaseSchedule = None,
      t_start: float = 1.0,
      t_end: float = 0.0,
      timesteps: float = 20,
      x_init: torch.Tensor = None,
      cfg: float = 3.0,
      cfg_t_stop: int = None,
      cfg_t_start: int = None,
      cfg_rho: float = 0.7,
      sampler_params: Dict[str, Any] = None,
      shift: int = 1,
      device: str = "cpu",
      x0_forward: torch.Tensor = None,
      x0_style_forward: torch.Tensor = None,
      num_iter: int = 3,
      eta: float = 1e-1,
      tau: int = 20,
      eval_csd: bool = False,
      eval_sub_csd: bool = False,
      apply_pushforward: bool = False,
      tau_pushforward: int = 0,
      tau_pushforward_csd: int = 0,
      lam_content: float = 1.0,
      lam_style: float = 1.0,
      gamma_nc: float = 1.0,
      gamma_ns: float = 1.0,
      lam_txt_alignment: float = 0.0,
      use_attn_mask: bool = False,
      save_attn_mask: bool = False,
      models: WurstCoreC.Models = None,
      extras: WurstCoreC.Extras = None,
      sam_mask: float = 1.0,
      sam_prompt: str = None,
      use_sam_mask: bool = False,
      use_ddim_sampler: bool = False,
      guidance_mode: str = "none",
      zeta: float = 0.1,
      mu_s_0: float = 0.0,
      **kwargs: Any,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sampling with SubZero Disentangled Controller and Terminal Cost Guidance."""
    sampler_params = {} if sampler_params is None else sampler_params
    if sampler is None:
      sampler = DDPMSampler(self)
    if use_ddim_sampler:
      sampler = DDIMSampler(self)
    r_range = torch.linspace(t_start, t_end, timesteps + 1)
    schedule = self.schedule if schedule is None else schedule
    logSNR_range = (
        schedule(r_range, shift=shift)[:, None]
        .expand(-1, shape[0] if x_init is None else x_init.size(0))
        .to(device)
    )

    x = sampler.init_x(shape).to(device) if x_init is None else x_init.clone()
    if cfg is not None:
      if unconditional_inputs is None:
        unconditional_inputs = {
            k: torch.zeros_like(v) for k, v in model_inputs.items()
        }
      model_inputs = {
          k: (
              torch.cat([v, v_u], dim=0)
              if isinstance(v, torch.Tensor)
              else (
                  [
                      (
                          torch.cat([vi, vi_u], dim=0)
                          if isinstance(vi, torch.Tensor)
                          and isinstance(vi_u, torch.Tensor)
                          else None
                      )
                      for vi, vi_u in zip(v, v_u)
                  ]
                  if isinstance(v, list)
                  else (
                      {
                          vk: torch.cat(
                              [v[vk], v_u.get(vk, torch.zeros_like(v[vk]))],
                              dim=0,
                          )
                          for vk in v
                      }
                      if isinstance(v, dict)
                      else None
                  )
              )
          )
          for (k, v), (k_u, v_u) in zip(
              model_inputs.items(), unconditional_inputs.items()
          )
      }

    # Initialize guidance extractors: DINO (rho) and CSD (psi) based on guidance_mode
    csd_model = setup_csd(device=device) if (guidance_mode == "csd" and (eval_csd or eval_sub_csd) and lam_style > 0) else None
    dino_model = kwargs.get("dino_model")
    if dino_model is None and guidance_mode == "dino" and (eval_csd or eval_sub_csd) and lam_style > 0:
      dino_model = setup_dino(device=device)

    sam_model = None
    if use_attn_mask:
      try:
        from lang_sam import LangSAM
        sam_model = LangSAM()
      except ImportError:
        sam_model = None

    # Reference feature extraction: ssm_sub, ssm_sty, psi_sty, psi_sub, feat_sty_dino
    ssm_sub = None
    ssm_sty = None
    psi_sty = None
    psi_sub = None
    feat_sty_dino = None

    enable_guidance = (os.environ.get("ENABLE_GUIDANCE", "1") == "1")

    if enable_guidance and (eval_sub_csd or eval_csd) and x0_forward is not None and x0_style_forward is not None:
      with torch.no_grad():
        org_image = models.previewer(x0_forward)  # shape: [B, 3, 1024, 1024]
        org_style = models.previewer(x0_style_forward)  # shape: [B, 3, 1024, 1024]

        # Extract Style reference features via DINO
        if dino_model is not None:
          feat_sty_dino = dino_model(dino_preprocess(org_style))  # shape: [B, 384]

        # Style features via CSD (psi)
        if csd_model is not None:
          psi_sty = csd_model(extras.clip_preprocess(org_style))[2]  # shape: [B, 1024]
          psi_sub = csd_model(extras.clip_preprocess(org_image))[2]  # shape: [B, 1024]

    # Initialize dynamic style weight
    mu_s = mu_s_0
    Style_Storage.mu_s = mu_s

    for i in range(0, timesteps):
      Style_Storage.current_step = i
      Style_Storage.num_steps = timesteps

      noise_cond = self.noise_cond(logSNR_range[i])
      if (
          cfg is not None
          and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop)
          and (cfg_t_start is None or r_range[i].item() <= cfg_t_start)
      ):
        cfg_val = cfg
        if isinstance(cfg_val, (list, tuple)):
          assert (
              len(cfg_val) == 2
          ), "cfg must be a float or a list/tuple of length 2"
          cfg_val = cfg_val[0] * r_range[i].item() + cfg_val[1] * (
              1 - r_range[i].item()
          )
        with torch.no_grad():
          with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred, pred_unconditional = model(
                torch.cat([x, x], dim=0),
                noise_cond.repeat(2),
                **model_inputs,
            ).chunk(2)
        pred_cfg = torch.lerp(pred_unconditional, pred, cfg_val)
        if cfg_rho > 0:
          std_pos, std_cfg = pred.std(), pred_cfg.std()
          pred = cfg_rho * (
              pred_cfg * std_pos / (std_cfg + 1e-9)
          ) + pred_cfg * (1 - cfg_rho)
        else:
          pred = pred_cfg
      else:
        pred = model(x, noise_cond, **model_inputs)

      x0, epsilon = self.undiffuse(x, logSNR_range[i], pred)

      # Style Guidance Block: DINO or CSD
      if enable_guidance and i < tau and (eval_sub_csd or eval_csd) and guidance_mode in ("dino", "csd") and lam_style > 0:
        z0 = x0.clone().detach().requires_grad_(True)  # shape: [B, 16, H, W]

        for _ in range(num_iter):
          pred_image = models.previewer(z0)  # shape: [B, 3, 1024, 1024]

          if guidance_mode == "dino" and dino_model is not None and feat_sty_dino is not None:
            pred_dino_input = dino_preprocess(pred_image)  # shape: [B, 3, 224, 224]
            feat_pred_dino = dino_model(pred_dino_input)  # shape: [B, 384]
            loss_style = nnf.mse_loss(feat_pred_dino, feat_sty_dino)  # shape: scalar
          elif guidance_mode == "csd" and csd_model is not None and psi_sty is not None:
            psi_pred = csd_model(extras.clip_preprocess(pred_image))[2]  # shape: [B, 1024]
            loss_style = nnf.mse_loss(psi_pred, psi_sty)  # shape: scalar
          else:
            loss_style = None

          if loss_style is not None:
            loss = lam_style * loss_style  # shape: scalar

            # Score-Orthogonal Gradient Projection (Manifold Preservation)
            g = torch.autograd.grad(loss, z0, retain_graph=False)[0]  # shape: [B, 16, H, W]
            if os.environ.get("USE_ORTHO_GUIDANCE", "1") == "1":
              dim_proj = tuple(range(1, g.ndim))
              dot_product = torch.sum(g * epsilon, dim=dim_proj, keepdim=True)  # shape: [B, 1, 1, 1]
              norm_sq = torch.sum(epsilon * epsilon, dim=dim_proj, keepdim=True) + 1e-6  # shape: [B, 1, 1, 1]
              proj_g_on_eps = (dot_product / norm_sq) * epsilon  # shape: [B, 16, H, W]
              g_ortho = g - proj_g_on_eps  # shape: [B, 16, H, W]
            else:
              g_ortho = g

            # Safe latent manifold update via orthogonal guidance
            eta_dynamic = eta * (1.0 - i / timesteps)
            z0 = (z0 - eta_dynamic * g_ortho).detach().requires_grad_(True)  # shape: [B, 16, H, W]
            del loss, g, g_ortho

        if loss_style is not None:
          x0 = z0.detach()  # shape: [B, 16, H, W]
          print(f"[{guidance_mode.upper()} Style Guidance] Step i={i}/{timesteps}: loss_style={loss_style.item():.4f}, eta_eff={eta_dynamic:.4f}")
          del pred_image, loss_style

      # AdaIN Clean Latent Pushforward Block (Aligns style color/tone into x0 during early timesteps)
      if i < tau_pushforward and x0_style_forward is not None and tau_pushforward > 0:
        gamma_pushforward = 0.6

        style_ref = x0_style_forward.to(x0.device)
        mean_style = style_ref.mean(dim=[-2, -1], keepdim=True)
        std_style = style_ref.std(dim=[-2, -1], keepdim=True)

        mean_x0 = x0.mean(dim=[-2, -1], keepdim=True)
        std_x0 = x0.std(dim=[-2, -1], keepdim=True)

        x0_norm = (x0 - mean_x0) / (std_x0 + 1e-6)
        x0_adain = x0_norm * std_style + mean_style
        x0 = (1.0 - gamma_pushforward) * x0 + gamma_pushforward * x0_adain
        print(f"[AdaIN on x0] Step i={i}/{tau_pushforward}: gamma={gamma_pushforward:.2f}, mean_x0={mean_x0.mean().item():.4f} -> mean_style={mean_style.mean().item():.4f}, std_style={std_style.mean().item():.4f}")

      x = sampler(
          x, x0, epsilon, logSNR_range[i], logSNR_range[i + 1], **sampler_params
      )
      altered_vars = yield (x0, x, pred)

      if altered_vars is not None:
        cfg = altered_vars.get("cfg", cfg)
        cfg_rho = altered_vars.get("cfg_rho", cfg_rho)
        sampler = altered_vars.get("sampler", sampler)
        model_inputs = altered_vars.get("model_inputs", model_inputs)
        x = altered_vars.get("x", x)
        x_init = altered_vars.get("x_init", x_init)


# Backward compatibility alias
RBM = OrthoGuidance

