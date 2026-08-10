
import math
import random
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationSensitiveAttention(nn.Module):
    """
    Lightweight location-sensitive attention suitable for controlled experiments.
    """
    def __init__(self, enc_dim=64, dec_dim=64, attn_dim=64, conv_channels=16, kernel_size=7):
        super().__init__()
        self.query_proj = nn.Linear(dec_dim, attn_dim, bias=False)
        self.key_proj = nn.Linear(enc_dim, attn_dim, bias=False)
        self.loc_conv = nn.Conv1d(1, conv_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.loc_proj = nn.Linear(conv_channels, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys, prev_attn):
        # query: [B, Dd], keys: [B, N, De], prev_attn: [B, N]
        q = self.query_proj(query).unsqueeze(1)
        k = self.key_proj(keys)
        loc = self.loc_conv(prev_attn.unsqueeze(1)).transpose(1, 2)
        loc = self.loc_proj(loc)
        energy = self.v(torch.tanh(q + k + loc)).squeeze(-1)
        return F.softmax(energy, dim=-1)


class MonotonicPrior(nn.Module):
    """
    Differentiable Gaussian monotonic prior centered near the previous attention position.
    """
    def __init__(self, sigma=1.5, forward_step=1.0):
        super().__init__()
        self.sigma = sigma
        self.forward_step = forward_step

    def forward(self, prev_attn):
        B, N = prev_attn.shape
        pos = torch.arange(N, device=prev_attn.device, dtype=prev_attn.dtype).unsqueeze(0)
        prev_center = (prev_attn * pos).sum(dim=-1, keepdim=True)
        center = prev_center + self.forward_step
        logits = -0.5 * ((pos - center) / self.sigma) ** 2
        return F.softmax(logits, dim=-1)


class AdaptiveHybridAttention(nn.Module):
    """
    α_t = λ_t α_loc + (1-λ_t) α_mono
    λ_t = sigmoid(W [decoder_state ; context_summary] + b)
    """
    def __init__(self, enc_dim=64, dec_dim=64, attn_dim=64):
        super().__init__()
        self.location = LocationSensitiveAttention(enc_dim, dec_dim, attn_dim)
        self.monotonic = MonotonicPrior()
        self.gate = nn.Sequential(
            nn.Linear(dec_dim + enc_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, query, keys, prev_attn):
        alpha_loc = self.location(query, keys, prev_attn)
        alpha_mono = self.monotonic(prev_attn)

        context_summary = torch.bmm(prev_attn.unsqueeze(1), keys).squeeze(1)
        lam = torch.sigmoid(self.gate(torch.cat([query, context_summary], dim=-1)))
        alpha = lam * alpha_loc + (1.0 - lam) * alpha_mono
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)

        return alpha, lam.squeeze(-1), alpha_loc, alpha_mono


@dataclass
class Metrics:
    backward_rate: float
    jump_rate: float
    mean_abs_error: float
    entropy: float


def attention_metrics(path, target, max_forward_jump=3):
    # path: [T, N], target: [T]
    eps = 1e-9
    peaks = path.argmax(dim=-1)
    diffs = peaks[1:] - peaks[:-1]
    backward_rate = (diffs < 0).float().mean().item() if len(diffs) else 0.0
    jump_rate = (diffs > max_forward_jump).float().mean().item() if len(diffs) else 0.0
    mae = (peaks.float() - target.float()).abs().mean().item()
    entropy = (-(path * (path + eps).log()).sum(dim=-1)).mean().item()
    return Metrics(backward_rate, jump_rate, mae, entropy)


def generate_trial(T=80, N=40, noise=1.0, seed=0):
    """
    Creates a controlled text/acoustic alignment problem.
    Ground truth advances monotonically from token 0 to token N-1.
    Local attention scores are corrupted by noise and occasional distractors.
    """
    g = torch.Generator().manual_seed(seed)
    target = torch.clamp(torch.floor(torch.linspace(0, N - 1, T)), 0, N - 1).long()

    loc_path = []
    hybrid_path = []

    prev_loc = F.one_hot(torch.tensor(0), N).float().unsqueeze(0)
    prev_hyb = prev_loc.clone()

    pos = torch.arange(N).float()

    # In this synthetic benchmark the learned gate is represented by a
    # deterministic confidence rule so no training is required.
    for t in range(T):
        center = target[t].float()

        # "Location-sensitive" candidate: correct Gaussian + stochastic corruption
        logits = -0.5 * ((pos - center) / 1.7) ** 2
        logits += noise * torch.randn(N, generator=g)

        # occasional strong distractor produces skips/repeats/backward movement
        if torch.rand(1, generator=g).item() < 0.18:
            distractor = int(torch.randint(0, N, (1,), generator=g).item())
            logits[distractor] += 4.0 + 1.5 * noise

        alpha_loc = F.softmax(logits, dim=-1)

        # monotonic prior based on previous hybrid center
        prev_center = (prev_hyb.squeeze(0) * pos).sum()
        mono_center = torch.clamp(prev_center + 0.5, 0, N - 1)
        mono_logits = -0.5 * ((pos - mono_center) / 1.5) ** 2
        alpha_mono = F.softmax(mono_logits, dim=-1)

        # adaptive confidence gate:
        # sharp/high-confidence local attention => trust local branch more.
        entropy = -(alpha_loc * (alpha_loc + 1e-9).log()).sum()
        confidence = torch.exp(-entropy)
        lam = torch.clamp(0.20 + 2.4 * confidence, 0.15, 0.85)

        alpha_h = lam * alpha_loc + (1 - lam) * alpha_mono
        alpha_h /= alpha_h.sum()

        loc_path.append(alpha_loc)
        hybrid_path.append(alpha_h)
        prev_loc = alpha_loc.unsqueeze(0)
        prev_hyb = alpha_h.unsqueeze(0)

    return torch.stack(loc_path), torch.stack(hybrid_path), target


def run_benchmark(trials=100, noise=1.25):
    vals_loc, vals_hyb = [], []
    for seed in range(trials):
        loc, hyb, target = generate_trial(noise=noise, seed=seed)
        vals_loc.append(attention_metrics(loc, target))
        vals_hyb.append(attention_metrics(hyb, target))

    def avg(xs, attr):
        return sum(getattr(x, attr) for x in xs) / len(xs)

    result = {
        "trials": trials,
        "noise": noise,
        "location_sensitive": {
            "backward_rate": avg(vals_loc, "backward_rate"),
            "jump_rate": avg(vals_loc, "jump_rate"),
            "mean_abs_error": avg(vals_loc, "mean_abs_error"),
            "entropy": avg(vals_loc, "entropy"),
        },
        "adaptive_hybrid": {
            "backward_rate": avg(vals_hyb, "backward_rate"),
            "jump_rate": avg(vals_hyb, "jump_rate"),
            "mean_abs_error": avg(vals_hyb, "mean_abs_error"),
            "entropy": avg(vals_hyb, "entropy"),
        }
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--noise", type=float, default=1.25)
    args = parser.parse_args()

    result = run_benchmark(args.trials, args.noise)
    print("Synthetic alignment benchmark")
    print(f"Trials: {result['trials']}  Noise: {result['noise']}")
    print()
    print(f"{'Metric':<22}{'Location':>14}{'Adaptive Hybrid':>18}")
    print("-" * 54)
    for key, label in [
        ("backward_rate", "Backward-step rate"),
        ("jump_rate", "Large-jump rate"),
        ("mean_abs_error", "Mean abs. error"),
        ("entropy", "Attention entropy"),
    ]:
        a = result["location_sensitive"][key]
        b = result["adaptive_hybrid"][key]
        print(f"{label:<22}{a:>14.4f}{b:>18.4f}")
