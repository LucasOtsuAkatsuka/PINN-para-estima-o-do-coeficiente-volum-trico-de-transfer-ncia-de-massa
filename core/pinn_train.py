import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .pinn_model import (
    ProfileNet,
    KyamNet,
    get_device,

    normalize_sensors,
    normalize_inputs,

    normalize_sensors_batch,
    normalize_inputs_batch,
    make_collocation_batch,
    physics_loss_batch,
    data_loss_split_batch,
)

from .config import (
    Z_T,
    YW_IN,
    L_IN,
    PINN_EPOCHS,
    PINN_N_COLLOC,
    PINN_LR,
    PINN_W_PHYS,
    PINN_W_BC_IN,
    PINN_W_BC_OUT,
    PINN_GRAD_CLIP,
    PINN_BATCH_SIZE,
    PINN_TRAIN_SPLIT,
)


def train_pinn(csv_path, epochs=None, n_colloc=None, lr=None,
               w_phys=None, w_bc_in=None, w_bc_out=None, batch_size=None,
               progress_callback=None):

    device = get_device()

    epochs     = epochs     or PINN_EPOCHS
    n_colloc   = n_colloc   or PINN_N_COLLOC
    lr         = lr         or PINN_LR
    w_phys     = w_phys     if w_phys   is not None else PINN_W_PHYS
    w_bc_in    = w_bc_in    if w_bc_in  is not None else PINN_W_BC_IN
    w_bc_out   = w_bc_out   if w_bc_out is not None else PINN_W_BC_OUT
    batch_size = batch_size or PINN_BATCH_SIZE

    df = pd.read_csv(csv_path)

    has_true = 'kyam_true' in df.columns
    has_yw_l = 'Yw_in' in df.columns and 'L_in' in df.columns

    n = len(df)
    n_train = int(n * PINN_TRAIN_SPLIT)

    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)


    TL_in_train  = torch.tensor(df_train['TL_in'].values,  dtype=torch.float32, device=device)
    TG_out_train = torch.tensor(df_train['TG_out'].values, dtype=torch.float32, device=device)
    TG_in_train  = torch.tensor(df_train['TG_in'].values,  dtype=torch.float32, device=device)
    TL_out_train = torch.tensor(df_train['TL_out'].values, dtype=torch.float32, device=device)

    if has_yw_l:
        Yw_in_train = torch.tensor(df_train['Yw_in'].values, dtype=torch.float32, device=device)
        L_in_train  = torch.tensor(df_train['L_in'].values,  dtype=torch.float32, device=device)
    else:
        Yw_in_train = torch.full((len(df_train),), YW_IN, dtype=torch.float32, device=device)
        L_in_train  = torch.full((len(df_train),), L_IN,  dtype=torch.float32, device=device)


    profile_net = ProfileNet().to(device)
    kyam_net    = KyamNet().to(device)

    all_params = list(profile_net.parameters()) + list(kyam_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'loss': [], 'phys': [], 'bc_in': [], 'bc_out': []}
    n_tr = len(df_train)

    for epoch in range(epochs):
        profile_net.train()
        kyam_net.train()

        prog = epoch / max(epochs - 1, 1)
        w_phys_eff   = w_phys * (0.1 + 0.9 * prog)
        w_bc_in_eff  = w_bc_in * (1.0 + 2.0 * (1.0 - prog))
        w_bc_out_eff = w_bc_out

        perm = torch.randperm(n_tr, device=device)

        epoch_loss   = 0.0
        epoch_phys   = 0.0
        epoch_bc_in  = 0.0
        epoch_bc_out = 0.0
        n_batches    = 0

        for start in range(0, n_tr, batch_size):
            end = min(start + batch_size, n_tr)
            idx = perm[start:end]
            B = end - start

            optimizer.zero_grad()


            tl_in_b  = TL_in_train[idx]
            tg_out_b = TG_out_train[idx]
            tg_in_b  = TG_in_train[idx]
            tl_out_b = TL_out_train[idx]
            yw_in_b  = Yw_in_train[idx]
            l_in_b   = L_in_train[idx]


            sensors_b = normalize_sensors_batch(
                tl_in_b, tg_out_b, tg_in_b, tl_out_b, yw_in_b, l_in_b
            )
            kyam_b = kyam_net(sensors_b)


            tl_in_n, tg_in_n, yw_in_n, l_in_n = normalize_inputs_batch(
                tl_in_b, tg_in_b, yw_in_b, l_in_b
            )


            z_c = make_collocation_batch(B, n_colloc, device=device)


            p_loss = physics_loss_batch(
                profile_net, z_c,
                tl_in_n, tg_in_n, yw_in_n, l_in_n,
                kyam_b,
            )


            bc_in_loss, bc_out_loss = data_loss_split_batch(
                profile_net,
                tl_in_b, tg_out_b, tg_in_b, tl_out_b,
                yw_in_b, l_in_b,
                kyam_b, device,
            )


            loss = (w_phys_eff * p_loss +
                    w_bc_in_eff * bc_in_loss +
                    w_bc_out_eff * bc_out_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=PINN_GRAD_CLIP)
            optimizer.step()

            epoch_loss   += loss.item()
            epoch_phys   += p_loss.item()
            epoch_bc_in  += bc_in_loss.item()
            epoch_bc_out += bc_out_loss.item()
            n_batches    += 1

        scheduler.step()

        history['loss'].append(epoch_loss / n_batches)
        history['phys'].append(epoch_phys / n_batches)
        history['bc_in'].append(epoch_bc_in / n_batches)
        history['bc_out'].append(epoch_bc_out / n_batches)

        if progress_callback and (epoch % 50 == 0 or epoch == epochs - 1):
            progress_callback(epoch, epochs, epoch_loss / n_batches)


    profile_net = profile_net.cpu()
    kyam_net    = kyam_net.cpu()
    profile_net.eval()
    kyam_net.eval()

    def evaluate(df_eval):
        eval_has_yw_l = 'Yw_in' in df_eval.columns and 'L_in' in df_eval.columns
        results = []
        with torch.no_grad():
            for _, row in df_eval.iterrows():
                yw_in_val = row['Yw_in'] if eval_has_yw_l else YW_IN
                l_in_val  = row['L_in']  if eval_has_yw_l else L_IN

                sensors = normalize_sensors(
                    row['TL_in'], row['TG_out'],
                    row['TG_in'], row['TL_out'],
                    yw_in_val, l_in_val,
                )
                kyam_pred = kyam_net(sensors).item()

                r = {
                    'sample_id': int(row.get('sample_id', 0)),
                    'kyam_pred': kyam_pred,
                    'TL_in':  row['TL_in'],
                    'TG_out': row['TG_out'],
                    'TG_in':  row['TG_in'],
                    'TL_out': row['TL_out'],
                    'Yw_in':  yw_in_val,
                    'L_in':   l_in_val,
                }
                if has_true:
                    r['kyam_true'] = row['kyam_true']
                    r['error'] = abs(kyam_pred - row['kyam_true'])
                results.append(r)
        return results

    train_results = evaluate(df_train)
    test_results  = evaluate(df_test)

    return profile_net, kyam_net, train_results, test_results, history


def predict_single(profile_net, kyam_net, TL_in, TG_out, TG_in, TL_out,
                   Yw_in=None, L_in=None):
    if Yw_in is None:
        Yw_in = YW_IN
    if L_in is None:
        L_in = L_IN

    profile_net.eval()
    kyam_net.eval()

    with torch.no_grad():
        sensors = normalize_sensors(TL_in, TG_out, TG_in, TL_out, Yw_in, L_in)
        kyam = kyam_net(sensors).item()

        z_norm = torch.linspace(0, 1, 200).unsqueeze(1)
        tl_in_n, tg_in_n, yw_in_n, l_in_n = normalize_inputs(TL_in, TG_in, Yw_in, L_in)
        tl_exp = tl_in_n.expand(200, 1)
        tg_exp = tg_in_n.expand(200, 1)
        yw_exp = yw_in_n.expand(200, 1)
        l_exp  = l_in_n.expand(200, 1)


        kyam_norm = torch.tensor([[kyam / 5.0]], dtype=torch.float32)
        kyam_exp = kyam_norm.expand(200, 1)

        state = profile_net(z_norm, tl_exp, tg_exp, yw_exp, l_exp, kyam_exp)

    z_real = z_norm.squeeze().numpy() * Z_T

    profiles = {
        'z':  z_real,
        'Yw': state[:, 0].numpy(),
        'L':  state[:, 1].numpy(),
        'TG': state[:, 2].numpy(),
        'TL': state[:, 3].numpy(),
    }

    return kyam, profiles


def diagnose(kyam):
    if kyam >= 4.0:
        return "SAUDAVEL", "green"
    elif kyam >= 2.5:
        return "INCRUSTACAO", "orange"
    else:
        return "FALHA MECANICA", "red"


def save_models(profile_net, kyam_net, path='pinn_weights.pth'):
    torch.save({
        'profile_net': profile_net.cpu().state_dict(),
        'kyam_net':    kyam_net.cpu().state_dict(),
    }, path)


def load_models(path='pinn_weights.pth'):
    profile_net = ProfileNet()
    kyam_net    = KyamNet()
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    profile_net.load_state_dict(checkpoint['profile_net'])
    kyam_net.load_state_dict(checkpoint['kyam_net'])
    profile_net.eval()
    kyam_net.eval()
    return profile_net, kyam_net
