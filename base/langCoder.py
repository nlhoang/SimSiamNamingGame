from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import GenericMLP


def straight_through_onehot(probs: torch.Tensor):
    """Return (onehot_with_STE, argmax_ids) from probs in [B, T, V]."""
    ids = probs.argmax(dim=-1)  # [B, T]
    hard = F.one_hot(ids, num_classes=probs.size(-1)).float()  # [B, T, V]
    onehot = hard.detach() + probs - probs.detach()            # STE
    return onehot, ids


class LangCoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        word_length: int,
        dictionary_size: int,
        enc_hidden_dim: int = 256,
        dec_hidden_dim: int = 256,
        enc_layers: int = 2,
        dec_layers: int = 2,
        last_relu: bool = False,
        last_bn: bool = False,
        use_soft_decode: bool = True,   # train decoder on soft tokens
        use_gumbel: bool = True,        # turn Gumbel on/off
        use_STE: bool = True,           # turn STE on/off
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.word_length = word_length          # message length (T_msg)
        self.V = dictionary_size
        self.use_soft_decode = use_soft_decode
        self.use_gumbel = use_gumbel
        self.use_STE = use_STE

        # total message dimension (flattened [T_msg, V])
        msg_dim = word_length * dictionary_size

        # ----- Encoder: latent -> logits -----
        self.mlp_enc = GenericMLP(
            input_dim=latent_dim,
            hidden_dim=enc_hidden_dim,
            output_dim=msg_dim,
            num_layers=enc_layers,
            last_relu=last_relu,
            last_bn=last_bn,
        )

        # ----- Decoder: tokens -> latent -----
        self.mlp_dec = GenericMLP(
            input_dim=msg_dim,
            hidden_dim=dec_hidden_dim,
            output_dim=latent_dim,
            num_layers=dec_layers,
            last_relu=False,
            last_bn=False,
        )

    def encoder(self, x: torch.Tensor, sampling: bool = True, tau=0.5):
        B = x.size(0)

        # ----- logits -----
        logits = self.mlp_enc(x)                     # [B, T*V]
        logits = logits.view(B, self.word_length, self.V)  # [B, T_msg, V]

        # --- (1) choose distribution: Gumbel vs softmax ---
        if sampling and self.training and self.use_gumbel:
            if isinstance(tau, torch.Tensor):
                tau = float(tau.item())
            probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)  # [B, T, V]
        else:
            probs = F.softmax(logits, dim=-1)        # [B, T, V]

        # --- (2) choose how to make one-hots: STE or no-grad argmax ---
        if self.use_STE:
            onehots, ids = straight_through_onehot(probs)  # [B, T, V], [B, T]
        else:
            ids = probs.argmax(dim=-1)                     # [B, T]
            onehots = F.one_hot(ids, num_classes=self.V).float()  # [B, T, V]

        return probs, onehots, ids

    def decoder(self, tokens: torch.Tensor):
        B, T, V = tokens.shape
        assert T == self.word_length and V == self.V

        msg_flat = tokens.view(B, T * V)             # flatten message sequence [B, T*V]
        recon = self.mlp_dec(msg_flat)               # [B, latent_dim]
        return recon

    def forward(self, x: torch.Tensor):
        probs, onehots, ids = self.encoder(x, sampling=True)

        if self.training:
            tokens_for_decode = probs if self.use_soft_decode else onehots
        else:
            tokens_for_decode = onehots

        recons = self.decoder(tokens_for_decode)
        return probs, onehots, ids, recons

