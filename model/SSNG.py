"""
SimSiam Naming Game
"""
import math

import torch
import torch.nn as nn
import torch.optim as optim
from base.base_model import Encoder, GenericMLP
from base.utils import negative_cosine_similarity, ce_loss, vicreg_loss
from base.langCoder import LangCoder


INIT_TAU: float = 2.0
MIN_TAU: float = 1.0
ANNEAL_RATE: float = 0.02


class SSNG(nn.Module):
    """
    Two agents each observe their own view of the same object and align their representations through message exchange.
    Each agent: perception encoder f_i maps the observed object x_i to the interpretant z_i;
    the language coder's generator h_i produces the sign/message w_i from z_i, and its interpreter l_i maps a received
    message back into agent i's latent space.
    """
    def __init__(
            self,
            feature_dim: int,
            latent_dim: int,
            word_length: int,
            dictionary_size: int,
            backbone1: str,
            backbone2: str,
            pretrained_backbone: bool = True,
            freeze_backbone: bool = False,
            use_soft_decode: bool = True,
            backbone_stop_grad: bool = False,
        ):
        super(SSNG, self).__init__()
        self.use_soft_decode = use_soft_decode
        self.backbone_stop_grad = backbone_stop_grad

        self.word_length = word_length
        self.dictionary_size = dictionary_size

        self.register_buffer('tau', torch.tensor(INIT_TAU))
        self.register_buffer('global_step', torch.tensor(0.0))

        # Perception encoder (backbone + projector)
        self.encoder1 = Encoder(backbone=backbone1, freeze_backbone=freeze_backbone, pretrained=pretrained_backbone)
        self.encoder2 = Encoder(backbone=backbone2, freeze_backbone=freeze_backbone, pretrained=pretrained_backbone)

        self.projector1 = GenericMLP(input_dim=feature_dim, hidden_dim=feature_dim, output_dim=latent_dim,
                                     num_layers=2, last_relu=False, last_bn=True)
        self.projector2 = GenericMLP(input_dim=feature_dim, hidden_dim=feature_dim, output_dim=latent_dim,
                                     num_layers=2, last_relu=False, last_bn=True)

        # Language coder per agent
        self.langCoder1 = LangCoder(latent_dim=latent_dim, word_length=word_length, dictionary_size=dictionary_size,
                                    enc_hidden_dim=128, dec_hidden_dim=128, enc_layers=1, dec_layers=1)
        self.langCoder2 = LangCoder(latent_dim=latent_dim, word_length=word_length, dictionary_size=dictionary_size,
                                    enc_hidden_dim=128, dec_hidden_dim=128, enc_layers=1, dec_layers=1)

    def forward(self, x1, x2):
        # Perception: z_i^(xi) = f_i(x_i)
        y1 = self.encoder1(x1)
        z1 = self.projector1(y1)
        # Naming: w_i = h_i(z_i^(xi))
        probs1, onehot1, message1 = self.langCoder1.encoder(z1, tau=self.tau)

        y2 = self.encoder2(x2)
        z2 = self.projector2(y2)
        probs2, onehot2, message2 = self.langCoder2.encoder(z2, tau=self.tau)

        tokens1 = probs1 if (self.training and self.use_soft_decode) else onehot1
        tokens2 = probs2 if (self.training and self.use_soft_decode) else onehot2
        # Interpretation of own message: z_i^(wi) = l_i(w_i) -- Individual Prediction Error
        p1 = self.langCoder1.decoder(tokens1)
        p2 = self.langCoder2.decoder(tokens2)

        # Interpretation of partner's message: z_i^(wj) = l_i(w_j) -- Individual Regularization
        # The received message is treated as a fixed input, so no gradient flows into the sender.
        p12 = self.langCoder1.decoder(tokens2.detach())
        p21 = self.langCoder2.decoder(tokens1.detach())

        return (z1, tokens1, p1, p12,
                z2, tokens2, p2, p21)

    def loss_EmCom(self, z1, w1, p1, p12, z2, w2, p2, p21, alpha=0.4, beta=0.4, gamma=0.2,
                   sign_loss_fn='ce'): # ce mse vicreg
        """
        Per-agent objective J_i:
        alpha * z_i^(wj).z_i^(xi) [Individual Regularization]
        beta * z_i^(wi).z_i^(xi) [Individual Prediction Error]
        gamma * w_j.w_i [Collective Regularization]
        Summed over both agents this gives J_SSNG = J_A + J_B
        """
        # Individual Prediction Error: agreement between z_i^(xi) and z_i^(wi) (own message).
        recon_loss = (negative_cosine_similarity(p1, z1, detach=False)
                      + negative_cosine_similarity(p2, z2, detach=False))
        # Individual Regularization: agreement between z_i^(xi) and z_i^(wj) (partner's message).
        cross_loss = (negative_cosine_similarity(p12, z1, detach=self.backbone_stop_grad)
                      + negative_cosine_similarity(p21, z2, detach=self.backbone_stop_grad))
        # Collective Regularization: agreement between w_i and w_j. Implemented as an in-batch cross-entropy surrogate.
        if sign_loss_fn == 'ce':
            sign_loss = ce_loss(w1, w2.detach()) + ce_loss(w2, w1.detach())
        elif sign_loss_fn == 'vicreg':
            sign_loss = vicreg_loss(w1, w2)
        else:
            sign_loss = (negative_cosine_similarity(w1, w2.detach())
                         + negative_cosine_similarity(w2, w1.detach()))

        total = alpha * 1/2 * recon_loss + beta * 1/2 * cross_loss + gamma * 1/2 * sign_loss
        return total, recon_loss, cross_loss, sign_loss

    def forward_NoCom(self, x1, x2):
        """
        No-communication ablation: each agent only self-reconstructs from its own message
        (Individual Prediction Error only), with no partner message exchange.
        """
        y1 = self.encoder1(x1)
        z1 = self.projector1(y1)
        probs1, onehot1, message1 = self.langCoder1.encoder(z1, tau=self.tau)

        y2 = self.encoder2(x2)
        z2 = self.projector2(y2)
        probs2, onehot2, message2 = self.langCoder2.encoder(z2, tau=self.tau)

        tokens1 = probs1 if (self.training and self.use_soft_decode) else onehot1
        tokens2 = probs2 if (self.training and self.use_soft_decode) else onehot2
        p1 = self.langCoder1.decoder(tokens1)
        p2 = self.langCoder2.decoder(tokens2)

        return (z1, tokens1, p1,
                z2, tokens2, p2)

    def anneal_temperature(self):
        # Gumbel-Softmax temperature schedule for message discretization (Discretization: Gumbel-Softmax + STE).
        # Exponential anneal: tau = max(min_tau, init_tau * exp(-anneal_rate * step))
        init_tau, min_tau, rate = INIT_TAU, MIN_TAU, ANNEAL_RATE
        step = int(self.global_step.item())
        value = max(min_tau, init_tau * math.exp(-rate * step))
        self.tau.fill_(value)


def train(model, dataloader, learning_rate, device, epochs=100, save_interval=1,
          alpha=1.0, beta=1.0, gamma=1.0, save_prefix='SSNG_lang'):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device='cuda')
    model.train()
    loss_history = []

    save_path = f'{save_prefix}_0.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    # Discretization is disabled during the warm-up epochs and enabled with temperature annealing afterward.
    for epoch in range(epochs):
        ep_warmup = 20
        if epoch < ep_warmup:
            model.langCoder1.use_gumbel = False
            model.langCoder1.use_STE =False
            model.langCoder2.use_gumbel = False
            model.langCoder2.use_STE =False
        else:
            model.langCoder1.use_gumbel = True
            model.langCoder1.use_STE =True
            model.langCoder2.use_gumbel = True
            model.langCoder2.use_STE =True
            model.global_step.fill_(float(epoch - ep_warmup))
            model.anneal_temperature()

        train_loss, recon_loss, cross_loss, sign_loss = 0, 0, 0, 0
        for batch_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=True):
                z1, tokens1, p1, p12, z2, tokens2, p2, p21 = model(x1, x2)
                loss, recon, cross, sign = model.loss_EmCom(z1, tokens1, p1, p12,
                                                            z2, tokens2, p2, p21,
                                                            alpha, beta, gamma)

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

def loss_NoCom(z1, z2, p1, p2):
    # Individual Prediction Error only (no partner message term) -- the "No Communication" baseline.
    loss = 0.5 * (negative_cosine_similarity(p1, z1, detach=False)
                  + negative_cosine_similarity(p2, z2, detach=False))
    return loss


def train_NoCom(model, dataloader, learning_rate, device, epochs=100, save_interval=1,
                alpha=0.4, beta=0.4, gamma=0.2, save_prefix='SSNG_NoCom'):
    """Training loop for the No-Communication ablation (forward_NoCom/loss_NoCom)."""
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device='cuda')
    model.train()
    loss_history = []

    save_path = f'{save_prefix}_0.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    for epoch in range(epochs):
        ep_warmup = 20
        if epoch < ep_warmup:
            model.langCoder1.use_gumbel = False
            model.langCoder1.use_STE = False
            model.langCoder2.use_gumbel = False
            model.langCoder2.use_STE = False
        else:
            model.langCoder1.use_gumbel = True
            model.langCoder1.use_STE = True
            model.langCoder2.use_gumbel = True
            model.langCoder2.use_STE = True
            model.global_step.fill_(float(epoch - ep_warmup))
            model.anneal_temperature()

        train_loss = 0
        for batch_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=True):
                z1, w1, p1, z2, w2, p2 = model.forward_NoCom(x1, x2)
                loss = loss_NoCom(z1, z2, p1, p2)

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
