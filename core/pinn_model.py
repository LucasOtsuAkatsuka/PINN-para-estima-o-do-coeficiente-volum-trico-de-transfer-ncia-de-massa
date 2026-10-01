import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import (
    Z_T,
    G_B,
    CP_B,
    CP_W_GAS,
    CP_W_LIQ,
    DH_VAP,
    LE,
    SCALE_YW,
    SCALE_L,
    SCALE_T,
)


def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"[GPU] Usando CUDA: {torch.cuda.get_device_name(0)}")
        return dev
    else:
        print("[CPU] Nenhuma GPU detectada, usando CPU")
        return torch.device('cpu')


class ProfileNet(nn.Module):

    def __init__(self, hidden_layers=4, hidden_dim=32):
        super().__init__()


        layers = [nn.Linear(6, hidden_dim), nn.SiLU()]

        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]

        layers.append(nn.Linear(hidden_dim, 4))

        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z_norm, TL_in_norm, TG_in_norm, Yw_in_norm, L_in_norm, kyam_norm):
        x = torch.cat([z_norm, TL_in_norm, TG_in_norm, Yw_in_norm, L_in_norm, kyam_norm], dim=1)
        raw = self.net(x)

        Yw = 0.005 + 0.055 * torch.sigmoid(raw[:, 0:1])
        L_in = 6.0 + 2.0 * L_in_norm
        L = L_in * torch.exp(-0.01 * (1.0 - z_norm) * F.softplus(raw[:, 1:2]))
        TG = 290.0 + 25.0 * raw[:, 2:3]
        TL = 300.0 + 25.0 * raw[:, 3:4]

        return torch.cat([Yw, L, TG, TL], dim=1)


class KyamNet(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, sensors_norm):
        raw = self.net(sensors_norm)
        return 0.5 + F.softplus(raw)


def normalize_sensors(TL_in, TG_out, TG_in, TL_out, Yw_in, L_in):
    vals = torch.tensor([
        (float(TL_in)  - 300.0) / 15.0,
        (float(TG_out) - 290.0) / 10.0,
        (float(TG_in)  - 285.0) / 10.0,
        (float(TL_out) - 295.0) / 10.0,
        (float(Yw_in)  - 0.012) / 0.008,
        (float(L_in)   - 6.0)   / 2.0,
    ], dtype=torch.float32)
    return vals.unsqueeze(0)


def normalize_inputs(TL_in, TG_in, Yw_in, L_in):
    tl = torch.tensor(float(TL_in), dtype=torch.float32).reshape(1, 1)
    tg = torch.tensor(float(TG_in), dtype=torch.float32).reshape(1, 1)
    yw = torch.tensor(float(Yw_in), dtype=torch.float32).reshape(1, 1)
    li = torch.tensor(float(L_in),  dtype=torch.float32).reshape(1, 1)
    return (
        (tl - 300.0) / 15.0,
        (tg - 285.0) / 10.0,
        (yw - 0.012) / 0.008,
        (li - 6.0)   / 2.0,
    )


def normalize_sensors_batch(TL_in, TG_out, TG_in, TL_out, Yw_in, L_in):
    return torch.stack([
        (TL_in  - 300.0) / 15.0,
        (TG_out - 290.0) / 10.0,
        (TG_in  - 285.0) / 10.0,
        (TL_out - 295.0) / 10.0,
        (Yw_in  - 0.012) / 0.008,
        (L_in   - 6.0)   / 2.0,
    ], dim=1)


def normalize_inputs_batch(TL_in, TG_in, Yw_in, L_in):
    return (
        ((TL_in - 300.0) / 15.0).unsqueeze(1),
        ((TG_in - 285.0) / 10.0).unsqueeze(1),
        ((Yw_in - 0.012) / 0.008).unsqueeze(1),
        ((L_in  - 6.0)   / 2.0).unsqueeze(1),
    )


def make_collocation_batch(B, n_colloc, device=None):
    n_f = n_colloc // 2
    n_r = n_colloc - n_f

    z_f = torch.linspace(0, Z_T, n_f, device=device)
    z_f = z_f.unsqueeze(0).expand(B, -1)

    z_r = torch.rand(B, n_r, device=device) * Z_T

    z_all = torch.cat([z_f, z_r], dim=1).unsqueeze(2)

    return z_all.requires_grad_(True)


def physics_loss_batch(profile_net, z_c, TL_in_n, TG_in_n, Yw_in_n, L_in_n, kyam_batch):
    B, Nc, _ = z_c.shape
    BNc = B * Nc

    z_flat = z_c.reshape(BNc, 1) / Z_T

    tl_flat = TL_in_n.unsqueeze(1).expand(B, Nc, 1).reshape(BNc, 1)
    tg_flat = TG_in_n.unsqueeze(1).expand(B, Nc, 1).reshape(BNc, 1)
    yw_flat = Yw_in_n.unsqueeze(1).expand(B, Nc, 1).reshape(BNc, 1)
    l_flat  = L_in_n.unsqueeze(1).expand(B, Nc, 1).reshape(BNc, 1)


    kyam_norm_flat = (kyam_batch / 5.0).unsqueeze(1).expand(B, Nc, 1).reshape(BNc, 1)


    state = profile_net(z_flat, tl_flat, tg_flat, yw_flat, l_flat, kyam_norm_flat)

    Yw = state[:, 0]
    L  = state[:, 1]
    TG = state[:, 2]
    TL = state[:, 3]

    dYw_dz = torch.autograd.grad(Yw.sum(), z_c, create_graph=True)[0].reshape(BNc)
    dL_dz  = torch.autograd.grad(L.sum(),  z_c, create_graph=True)[0].reshape(BNc)
    dTG_dz = torch.autograd.grad(TG.sum(), z_c, create_graph=True)[0].reshape(BNc)
    dTL_dz = torch.autograd.grad(TL.sum(), z_c, create_graph=True)[0].reshape(BNc)


    kyam_v = kyam_batch.squeeze(1).unsqueeze(1).expand(B, Nc).reshape(BNc)

    L_s  = torch.clamp(L,  min=0.1)
    Yw_s = torch.clamp(Yw, min=1e-4)
    TL_c = torch.clamp(TL, 260.0, 380.0)

    psat_val = torch.exp(73.649 - 7258.2 / TL_c - 7.3037 * torch.log(TL_c) + 4.1653e-6 * TL_c**2)
    psat_val = torch.clamp(psat_val, max=101325.0 * 0.99)

    Yw_I = 0.622 * psat_val / (101325.0 - psat_val)
    dYw_diff = Yw_I - Yw_s
    Cp_h = CP_B + Yw_s * CP_W_GAS
    hG_aH = LE * kyam_v * Cp_h

    rhs_Yw = kyam_v * dYw_diff / G_B
    rhs_L  = kyam_v * dYw_diff
    rhs_TG = (hG_aH * (TL - TG) + kyam_v * dYw_diff * CP_W_GAS * (TL - TG)) / (G_B * Cp_h)
    rhs_TL = (hG_aH * (TL - TG) + kyam_v * dYw_diff * DH_VAP) / (L_s * CP_W_LIQ)

    residual = (
        ((dYw_dz - rhs_Yw) / SCALE_YW)**2 +
        ((dL_dz  - rhs_L)  / SCALE_L)**2 +
        ((dTG_dz - rhs_TG) / SCALE_T)**2 +
        ((dTL_dz - rhs_TL) / SCALE_T)**2
    )

    return residual.mean()


def data_loss_split_batch(profile_net, TL_in, TG_out, TG_in, TL_out, Yw_in, L_in,
                          kyam_batch, device):
    B = TL_in.shape[0]

    TL_in_n, TG_in_n, Yw_in_n, L_in_n = normalize_inputs_batch(TL_in, TG_in, Yw_in, L_in)


    kyam_norm = kyam_batch / 5.0

    z0 = torch.zeros(B, 1, device=device)
    zT = torch.ones(B, 1, device=device)


    s0 = profile_net(z0, TL_in_n, TG_in_n, Yw_in_n, L_in_n, kyam_norm)
    sT = profile_net(zT, TL_in_n, TG_in_n, Yw_in_n, L_in_n, kyam_norm)


    loss_in = (
        ((s0[:, 0] - Yw_in)  / SCALE_YW)**2 +
        ((sT[:, 1] - L_in)   / SCALE_L)**2 +
        ((s0[:, 2] - TG_in)  / SCALE_T)**2 +
        ((sT[:, 3] - TL_in)  / SCALE_T)**2
    ).mean()


    loss_out = (
        ((sT[:, 2] - TG_out) / SCALE_T)**2 +
        ((s0[:, 3] - TL_out) / SCALE_T)**2
    ).mean()

    return loss_in, loss_out
