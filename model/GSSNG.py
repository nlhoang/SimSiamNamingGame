"""
Generalized N-agent version of SSNG
"""
import math
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from base.base_model import Encoder, GenericMLP
from base.utils import negative_cosine_similarity, ce_loss
from base.langCoder import LangCoder


INIT_TAU: float = 2.0
MIN_TAU: float = 1.0
ANNEAL_RATE: float = 1e-2


def _broadcast(value, n: int, name: str) -> list:
    """Allow either one value shared by every agent, or a per-agent list/tuple of length n."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"'{name}' has length {len(value)}, expected {n} (one entry per agent)")
        return list(value)
    return [value] * n


class MultiAgentSSNG(nn.Module):
    """
    Each of the fixed-N files hardcodes encoder1/2/3..., projector1/2/3..., langCoder1/2/3...
    Here every agent lives in an nn.ModuleList, and forward()
    loss_EmCom() loop over N and over all (i, j) pairs.
    """

    def __init__(
        self,
        num_agents: int,
        feature_dim: int,
        latent_dim: int,
        word_length: int,
        dictionary_size: int,
        backbones: Union[str, Sequence[str]],
        proj_layers: Union[int, Sequence[int]] = 2,
        lc_enc_layers: Union[int, Sequence[int]] = 1,
        lc_dec_layers: Union[int, Sequence[int]] = 1,
        lc_enc_hidden: Union[int, Sequence[int]] = 128,
        lc_dec_hidden: Union[int, Sequence[int]] = 128,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        use_soft_decode: bool = True,
        backbone_stop_grad: bool = False,
    ):
        super().__init__()
        assert num_agents >= 2, "MultiAgentSSNG needs at least 2 agents for cross/sign losses"
        self.num_agents = num_agents
        self.use_soft_decode = use_soft_decode
        self.backbone_stop_grad = backbone_stop_grad
        self.word_length = word_length
        self.dictionary_size = dictionary_size

        self.register_buffer('tau', torch.tensor(INIT_TAU))
        self.register_buffer('global_step', torch.tensor(0.0))

        backbone_list: list = _broadcast(backbones, num_agents, 'backbones')
        proj_layers_list: list = _broadcast(proj_layers, num_agents, 'proj_layers')
        enc_layers_list: list = _broadcast(lc_enc_layers, num_agents, 'lc_enc_layers')
        dec_layers_list: list = _broadcast(lc_dec_layers, num_agents, 'lc_dec_layers')
        enc_hidden_list: list = _broadcast(lc_enc_hidden, num_agents, 'lc_enc_hidden')
        dec_hidden_list: list = _broadcast(lc_dec_hidden, num_agents, 'lc_dec_hidden')

        self.encoders = nn.ModuleList([
            Encoder(backbone=backbone_list[i], freeze_backbone=freeze_backbone, pretrained=pretrained_backbone)
            for i in range(num_agents)
        ])
        self.projectors = nn.ModuleList([
            GenericMLP(input_dim=feature_dim, hidden_dim=feature_dim, output_dim=latent_dim,
                       num_layers=proj_layers_list[i], last_relu=False, last_bn=True)
            for i in range(num_agents)
        ])
        self.langcoders = nn.ModuleList([
            LangCoder(latent_dim=latent_dim, word_length=word_length, dictionary_size=dictionary_size,
                      enc_hidden_dim=enc_hidden_list[i], dec_hidden_dim=dec_hidden_list[i],
                      enc_layers=enc_layers_list[i], dec_layers=dec_layers_list[i])
            for i in range(num_agents)
        ])

    def encode(self, xs):
        """Run every agent's encoder -> projector -> langCoder.encoder. Returns zs, tokens (each list[N])."""
        N = self.num_agents
        assert len(xs) == N, f"expected {N} inputs (one per agent), got {len(xs)}"
        zs, tokens = [], []
        for i in range(N):
            y = self.encoders[i](xs[i])
            z = self.projectors[i](y)
            probs, onehot, _message = self.langcoders[i].encoder(z, tau=self.tau)
            tok = probs if (self.training and self.use_soft_decode) else onehot
            zs.append(z)
            tokens.append(tok)
        return zs, tokens

    def forward(self, *xs):
        """
        xs: N input tensors, one per agent.
        Returns:
          zs:     list[N]    zs[i] = agent i's own projected latent
          tokens: list[N]    tokens[i] = agent i's message (soft probs while training, else onehot)
          p:      list[N][N] p[i][j] = agent i's decoder applied to agent j's message
                              (tokens are detached whenever i != j, matching the two-agent SSNG)
        """
        zs, tokens = self.encode(xs)
        N = self.num_agents
        detached_tokens = [t.detach() for t in tokens]

        p = [[None] * N for _ in range(N)]
        for i in range(N):
            decoder = self.langcoders[i].decoder
            for j in range(N):
                p[i][j] = decoder(tokens[j] if j == i else detached_tokens[j])
        return zs, tokens, p

    def forward_NoCom(self, *xs):
        """Each agent only decodes its own message, no cross-agent communication."""
        zs, tokens = self.encode(xs)
        ps = [self.langcoders[i].decoder(tokens[i]) for i in range(self.num_agents)]
        return zs, tokens, ps

    def loss_EmCom(self, zs, tokens, p,
                   alpha=0.4, beta=0.4, gamma=0.2, sign_ce=True):
        """
        zs, tokens: list[N] as returned by forward()
        p:          list[N][N] as returned by forward()
        Normalization recon is averaged over the N self terms,
        cross/sign are averaged over the N*(N-1) ordered cross-agent pairs.
        """
        N = len(zs)

        recon_loss = sum(
            negative_cosine_similarity(p[i][i], zs[i], detach=False) for i in range(N)
        )

        cross_loss = sum(
            negative_cosine_similarity(p[i][j], zs[i], detach=self.backbone_stop_grad)
            for i in range(N) for j in range(N) if i != j
        )

        if sign_ce:
            sign_loss = sum(
                ce_loss(tokens[i], tokens[j].detach())
                for i in range(N) for j in range(N) if i != j
            )
        else:
            sign_loss = sum(
                F.mse_loss(tokens[i], tokens[j].detach())
                for i in range(N) for j in range(N) if i != j
            )

        num_pairs = N * (N - 1)
        total = alpha * recon_loss / N + beta * cross_loss / num_pairs + gamma * sign_loss / num_pairs
        return total, recon_loss, cross_loss, sign_loss

    def loss_NoCom(self, zs, ps):
        return sum(negative_cosine_similarity(ps[i], zs[i], detach=False) for i in range(len(zs)))

    def set_gumbel_ste(self, use_gumbel: bool, use_ste: bool):
        for lc in self.langcoders:
            lc.use_gumbel = use_gumbel
            lc.use_STE = use_ste

    def anneal_temperature(self):  # Exponential anneal: tau = max(min_tau, init_tau * exp(-anneal_rate * step))
        init_tau, min_tau, rate = INIT_TAU, MIN_TAU, ANNEAL_RATE
        step = int(self.global_step.item())
        value = max(min_tau, init_tau * math.exp(-rate * step))
        self.tau.fill_(value)


#-------------------------------------

def train(model, dataloader,
          learning_rate, device, epochs=100, save_interval=1,
          alpha=0.4, beta=0.4, gamma=0.2, ep_warmup=20, save_prefix='SSNG_'):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device='cuda')
    model.train()
    loss_history = []

    save_path = f'{save_prefix}_0.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    for epoch in range(epochs):
        if epoch < ep_warmup:
            model.set_gumbel_ste(use_gumbel=False, use_ste=False)
        else:
            model.set_gumbel_ste(use_gumbel=True, use_ste=True)
            model.global_step.fill_(float(epoch - ep_warmup))
            model.anneal_temperature()

        train_loss, recon_loss, cross_loss, sign_loss = 0, 0, 0, 0
        for batch_idx, xs in enumerate(dataloader):
            xs = [x.to(device) for x in xs]
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=True):
                zs, tokens, p = model(*xs)
                loss, recon, cross, sign = model.loss_EmCom(zs, tokens, p, alpha, beta, gamma)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            recon_loss += recon.item()
            cross_loss += cross.item()
            sign_loss += sign.item()

        avg_loss = train_loss / len(dataloader)
        avg_recon_loss = recon_loss / len(dataloader)
        avg_cross_loss = cross_loss / len(dataloader)
        avg_sign_loss = sign_loss / len(dataloader)
        loss_history.append(avg_loss)
        temp_tau = model.tau.item()
        print(f'====> Epoch [{epoch + 1}/{epochs}], Temperature: {temp_tau:.4f}, Total Loss: {avg_loss:.4f}, '
              f'Recon Loss: {avg_recon_loss:.4f}, Cross Loss: {avg_cross_loss:.4f}, Sign Loss: {avg_sign_loss:.4f}')
        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            save_path = f'{save_prefix}_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')

    return loss_history

#-------------------------------------

def train_NoCom(model, dataloader,
                learning_rate, device, epochs=100, save_interval=1, save_prefix='SSNG_NC_'):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device='cuda')
    model.train()
    loss_history = []

    save_path = f'{save_prefix}_0.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    for epoch in range(epochs):
        model.global_step.fill_(float(epoch))
        model.anneal_temperature()
        train_loss = 0
        for batch_idx, xs in enumerate(dataloader):
            xs = [x.to(device) for x in xs]
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=True):
                zs, tokens, ps = model.forward_NoCom(*xs)
                loss = model.loss_NoCom(zs, ps)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_loss = train_loss / len(dataloader)
        loss_history.append(avg_loss)
        temp_tau = model.tau.item()
        print(f'====> Epoch [{epoch + 1}/{epochs}], Temperature: {temp_tau:.4f}, Total Loss: {avg_loss:.4f}')
        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            save_path = f'{save_prefix}_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')

    return loss_history

#-------------------------------------
