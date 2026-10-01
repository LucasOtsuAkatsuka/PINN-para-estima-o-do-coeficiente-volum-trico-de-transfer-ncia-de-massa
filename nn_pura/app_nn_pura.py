import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.config import (
    Z_T, YW_IN, L_IN,
    PINN_TRAIN_SPLIT, PINN_GRAD_CLIP,
    SCALE_YW, SCALE_L, SCALE_T,
)


class ProfileNet_NN(nn.Module):
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


class KyamNet_NN(nn.Module):
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


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


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


def normalize_sensors_single(TL_in, TG_out, TG_in, TL_out, Yw_in, L_in):
    vals = torch.tensor([
        (float(TL_in)  - 300.0) / 15.0,
        (float(TG_out) - 290.0) / 10.0,
        (float(TG_in)  - 285.0) / 10.0,
        (float(TL_out) - 295.0) / 10.0,
        (float(Yw_in)  - 0.012) / 0.008,
        (float(L_in)   - 6.0)   / 2.0,
    ], dtype=torch.float32)
    return vals.unsqueeze(0)


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


WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', 'nn_pura_weights.pth')

_state = {
    'kyam_net': None,
}


def treinar(csv_file, epochs, progress=gr.Progress()):
    if csv_file is None:
        return "Erro: faca upload de um CSV.", None

    csv_path = csv_file.name if hasattr(csv_file, 'name') else str(csv_file)

    try:
        device = get_device()
        epochs = int(epochs)
        batch_size = 32
        lr = 1e-4
        w_bc_in = 150.0
        w_bc_out = 500.0

        df = pd.read_csv(csv_path)
        has_true = 'kyam_true' in df.columns
        has_yw_l = 'Yw_in' in df.columns and 'L_in' in df.columns

        n = len(df)
        n_train = int(n * PINN_TRAIN_SPLIT)
        indices = np.random.RandomState(42).permutation(n)
        df_train = df.iloc[indices[:n_train]].reset_index(drop=True)
        df_test  = df.iloc[indices[n_train:]].reset_index(drop=True)

        TL_in_t  = torch.tensor(df_train['TL_in'].values,  dtype=torch.float32, device=device)
        TG_out_t = torch.tensor(df_train['TG_out'].values, dtype=torch.float32, device=device)
        TG_in_t  = torch.tensor(df_train['TG_in'].values,  dtype=torch.float32, device=device)
        TL_out_t = torch.tensor(df_train['TL_out'].values, dtype=torch.float32, device=device)

        if has_yw_l:
            Yw_in_t = torch.tensor(df_train['Yw_in'].values, dtype=torch.float32, device=device)
            L_in_t  = torch.tensor(df_train['L_in'].values,  dtype=torch.float32, device=device)
        else:
            Yw_in_t = torch.full((len(df_train),), YW_IN, dtype=torch.float32, device=device)
            L_in_t  = torch.full((len(df_train),), L_IN,  dtype=torch.float32, device=device)

        profile_net = ProfileNet_NN().to(device)
        kyam_net    = KyamNet_NN().to(device)

        all_params = list(profile_net.parameters()) + list(kyam_net.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {'loss': [], 'bc_in': [], 'bc_out': []}
        n_tr = len(df_train)

        for epoch in range(epochs):
            profile_net.train()
            kyam_net.train()

            prog = epoch / max(epochs - 1, 1)
            w_bc_in_eff  = w_bc_in * (1.0 + 2.0 * (1.0 - prog))
            w_bc_out_eff = w_bc_out

            perm = torch.randperm(n_tr, device=device)
            epoch_loss = 0.0
            epoch_bc_in = 0.0
            epoch_bc_out = 0.0
            n_batches = 0

            for start in range(0, n_tr, batch_size):
                end = min(start + batch_size, n_tr)
                idx = perm[start:end]

                optimizer.zero_grad()

                tl_in_b  = TL_in_t[idx]
                tg_out_b = TG_out_t[idx]
                tg_in_b  = TG_in_t[idx]
                tl_out_b = TL_out_t[idx]
                yw_in_b  = Yw_in_t[idx]
                l_in_b   = L_in_t[idx]

                sensors_b = normalize_sensors_batch(
                    tl_in_b, tg_out_b, tg_in_b, tl_out_b, yw_in_b, l_in_b
                )
                kyam_b = kyam_net(sensors_b)

                bc_in_loss, bc_out_loss = data_loss_split_batch(
                    profile_net,
                    tl_in_b, tg_out_b, tg_in_b, tl_out_b,
                    yw_in_b, l_in_b,
                    kyam_b, device,
                )

                loss = w_bc_in_eff * bc_in_loss + w_bc_out_eff * bc_out_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=PINN_GRAD_CLIP)
                optimizer.step()

                epoch_loss   += loss.item()
                epoch_bc_in  += bc_in_loss.item()
                epoch_bc_out += bc_out_loss.item()
                n_batches    += 1

            scheduler.step()
            history['loss'].append(epoch_loss / n_batches)
            history['bc_in'].append(epoch_bc_in / n_batches)
            history['bc_out'].append(epoch_bc_out / n_batches)

            if epoch % 50 == 0 or epoch == epochs - 1:
                progress(epoch / epochs, desc=f"Epoca {epoch}/{epochs} — loss={epoch_loss/n_batches:.4f}")

        profile_net = profile_net.cpu()
        kyam_net    = kyam_net.cpu()
        profile_net.eval()
        kyam_net.eval()

        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        torch.save({
            'profile_net': profile_net.state_dict(),
            'kyam_net': kyam_net.state_dict(),
        }, WEIGHTS_PATH)
        _state['kyam_net'] = kyam_net


        def evaluate(df_eval):
            eval_yw_l = 'Yw_in' in df_eval.columns and 'L_in' in df_eval.columns
            results = []
            with torch.no_grad():
                for _, row in df_eval.iterrows():
                    yw = row['Yw_in'] if eval_yw_l else YW_IN
                    li = row['L_in']  if eval_yw_l else L_IN
                    sensors = normalize_sensors_single(
                        row['TL_in'], row['TG_out'], row['TG_in'], row['TL_out'], yw, li
                    )
                    kyam_pred = kyam_net(sensors).item()
                    r = {'kyam_pred': kyam_pred}
                    if has_true:
                        r['kyam_true'] = row['kyam_true']
                        r['error'] = abs(kyam_pred - row['kyam_true'])
                    results.append(r)
            return results

        train_res = evaluate(df_train)
        test_res  = evaluate(df_test)

        resumo = "NN PURA — Treinamento concluido!\n"
        resumo += f"Amostras treino: {len(train_res)} | teste: {len(test_res)}\n"

        if has_true:
            train_errs = [r['error'] for r in train_res]
            test_errs  = [r['error'] for r in test_res]
            train_rel  = [r['error']/r['kyam_true']*100 for r in train_res]
            test_rel   = [r['error']/r['kyam_true']*100 for r in test_res]

            resumo += "TREINO:\n"
            resumo += f"  Erro abs. medio: {np.mean(train_errs):.4f}\n"
            resumo += f"  Erro rel. medio: {np.mean(train_rel):.2f}%\n\n"
            resumo += "TESTE:\n"
            resumo += f"  Erro abs. medio: {np.mean(test_errs):.4f}\n"
            resumo += f"  Erro rel. medio: {np.mean(test_rel):.2f}%\n"
            resumo += f"  Erro maximo:     {np.max(test_errs):.4f}\n"


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.semilogy(history['loss'], color='#333', linewidth=1, label='Total')
        ax1.semilogy(history['bc_in'], color='#378ADD', linewidth=0.8, alpha=0.6, label='BC entrada')
        ax1.semilogy(history['bc_out'], color='#EF9F27', linewidth=1.2, alpha=0.8, label='BC saida')
        ax1.set_xlabel('Epoca')
        ax1.set_ylabel('Loss')
        ax1.set_title('NN Pura — Convergencia (SEM fisica)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        if has_true:
            kt_tr = [r['kyam_true'] for r in train_res]
            kp_tr = [r['kyam_pred'] for r in train_res]
            kt_te = [r['kyam_true'] for r in test_res]
            kp_te = [r['kyam_pred'] for r in test_res]
            ax2.scatter(kt_tr, kp_tr, c='#378ADD', s=30, alpha=0.6,
                        edgecolors='k', linewidths=0.3, label='Treino')
            ax2.scatter(kt_te, kp_te, c='#E24B4A', s=50, alpha=0.8,
                        edgecolors='k', linewidths=0.5, label='Teste')
            all_v = kt_tr + kt_te + kp_tr + kp_te
            lims = [min(all_v) - 0.5, max(all_v) + 0.5]
            ax2.plot(lims, lims, 'k--', alpha=0.3)
            ax2.set_xlim(lims); ax2.set_ylim(lims)
            ax2.set_xlabel('kyam real'); ax2.set_ylabel('kyam predito')
            ax2.set_title(f'Caixa Preta — Real vs Predito')
            ax2.legend()
            ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        return resumo, fig

    except Exception as e:
        return f"ERRO:\n\n{str(e)}\n\n{traceback.format_exc()}", None


def validar(csv_file):
    if csv_file is None:
        return "Erro: faca upload de um CSV.", None

    csv_path = csv_file.name if hasattr(csv_file, 'name') else str(csv_file)

    kyam_net = _state.get('kyam_net')
    if kyam_net is None:
        if os.path.exists(WEIGHTS_PATH):
            checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True)
            kyam_net = KyamNet_NN()
            kyam_net.load_state_dict(checkpoint['kyam_net'])
            kyam_net.eval()
            _state['kyam_net'] = kyam_net
        else:
            return "Erro: treine na Aba 1 primeiro.", None

    try:
        df = pd.read_csv(csv_path)
        if 'kyam_true' not in df.columns:
            return "Erro: CSV nao tem coluna 'kyam_true'.", None

        has_yw_l = 'Yw_in' in df.columns and 'L_in' in df.columns

        kyam_real, kyam_pred = [], []
        with torch.no_grad():
            for _, row in df.iterrows():
                yw = row['Yw_in'] if has_yw_l else YW_IN
                li = row['L_in']  if has_yw_l else L_IN
                sensors = normalize_sensors_single(
                    row['TL_in'], row['TG_out'], row['TG_in'], row['TL_out'], yw, li
                )
                kyam_pred.append(kyam_net(sensors).item())
                kyam_real.append(row['kyam_true'])

        kyam_real = np.array(kyam_real)
        kyam_pred = np.array(kyam_pred)
        erros = kyam_pred - kyam_real
        erros_abs = np.abs(erros)

        mae = np.mean(erros_abs)
        mre = np.mean(erros_abs / kyam_real * 100)
        bias = np.mean(erros)

        resumo = f"NN PURA — Generalizacao\n\n"
        resumo += f"CSV: {os.path.basename(csv_path)}\n"
        resumo += f"Amostras: {len(kyam_real)}\n"
        resumo += f"kyam range: [{kyam_real.min():.2f}, {kyam_real.max():.2f}]\n\n"
        resumo += f"MAE:  {mae:.4f}\n"
        resumo += f"MRE:  {mre:.2f}%\n"
        resumo += f"Bias: {bias:+.4f}\n"
        resumo += f"Max erro: {np.max(erros_abs):.4f}\n"


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        scatter = ax1.scatter(kyam_real, kyam_pred, c=erros_abs, cmap='RdYlGn_r', s=50,
                    edgecolors='k', linewidths=0.4, alpha=0.85)
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
        cbar.set_label('|Erro|', fontsize=9)
        all_vals = np.concatenate([kyam_real, kyam_pred])
        lims = [min(all_vals) - 0.3, max(all_vals) + 0.3]
        ax1.plot(lims, lims, 'k--', alpha=0.3)
        ax1.set_xlim(lims); ax1.set_ylim(lims)
        ax1.set_xlabel('kyam real')
        ax1.set_ylabel('kyam predito')
        ax1.set_title(f'NN Pura — Generalizacao\nMAE={mae:.4f}  MRE={mre:.1f}%')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.2)

        ordem = np.argsort(kyam_real)
        erros_ord = erros[ordem]
        cores = ['#E24B4A' if e > 0 else '#378ADD' for e in erros_ord]
        ax2.bar(range(len(erros_ord)), erros_ord, color=cores, alpha=0.7, width=1.0)
        ax2.axhline(0, color='k', linewidth=0.5)
        ax2.axhline(mae, color='#E24B4A', linestyle='--', alpha=0.5, label=f'+MAE={mae:.3f}')
        ax2.axhline(-mae, color='#378ADD', linestyle='--', alpha=0.5, label=f'-MAE={-mae:.3f}')
        ax2.set_xlabel('Amostras (ordenadas por kyam real)')
        ax2.set_ylabel('Erro (predito - real)')
        ax2.set_title('Erro por amostra')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()

        return resumo, fig

    except Exception as e:
        return f"ERRO:\n\n{str(e)}\n\n{traceback.format_exc()}", None


with gr.Blocks(title="NN Pura — Comparacao", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# NN Pura — Sem fisica (comparacao)")
    gr.Markdown("**Mesma arquitetura da PINN** (KyamNet → ProfileNet em cascata) "
                "mas **sem EDOs, sem pontos de colocacao, sem physics_loss**. Apenas dados.")

    with gr.Tab("1. Treinar NN Pura"):
        gr.Markdown("### Carregue o MESMO CSV usado na PINN para comparacao justa")
        with gr.Row():
            with gr.Column():
                csv_upload = gr.File(label="Upload CSV de treino", file_types=['.csv'])
                epochs_sl = gr.Slider(1000, 20000, value=10000, step=500, label="Epocas")
                btn_treinar = gr.Button("Treinar NN Pura", variant="primary")
            with gr.Column():
                resumo_txt = gr.Textbox(label="Resultado", lines=16, interactive=False)
        plot_train = gr.Plot(label="Convergencia e scatter")
        btn_treinar.click(fn=treinar, inputs=[csv_upload, epochs_sl],
                          outputs=[resumo_txt, plot_train])

    with gr.Tab("2. Validar Generalizacao"):
        gr.Markdown("### Carregue um CSV com kyam FORA da faixa de treino")
        with gr.Row():
            with gr.Column():
                csv_gen = gr.File(label="Upload CSV de generalizacao", file_types=['.csv'])
                btn_validar = gr.Button("Validar", variant="primary")
            with gr.Column():
                resumo_gen = gr.Textbox(label="Resultado", lines=12, interactive=False)
        plot_gen = gr.Plot(label="Generalizacao")
        btn_validar.click(fn=validar, inputs=[csv_gen],
                          outputs=[resumo_gen, plot_gen])

if __name__ == '__main__':
    demo.launch(share=False)
