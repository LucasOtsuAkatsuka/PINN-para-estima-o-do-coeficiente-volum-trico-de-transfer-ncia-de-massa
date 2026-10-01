import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.config import YW_IN, L_IN, PINN_TRAIN_SPLIT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--weights', default='weights/nn_pura_weights.pth')
    parser.add_argument('--output', default='outputs/real_vs_predito_nn_pura.pdf')
    args = parser.parse_args()

    from nn_pura.app_nn_pura import KyamNet_NN

    ck = torch.load(args.weights, map_location='cpu', weights_only=False)
    kyam_net = KyamNet_NN()
    kyam_net.load_state_dict(ck['kyam_net'])
    kyam_net.eval()

    n_inputs = list(kyam_net.parameters())[0].shape[1]

    df = pd.read_csv(args.csv)
    has_true = 'kyam_true' in df.columns
    has_yw_l = 'Yw_in' in df.columns and 'L_in' in df.columns

    if not has_true:
        print("ERRO: CSV sem kyam_true.")
        return

    n = len(df)
    n_train = int(n * PINN_TRAIN_SPLIT)
    indices = np.random.RandomState(42).permutation(n)
    df_train = df.iloc[indices[:n_train]].reset_index(drop=True)
    df_test = df.iloc[indices[n_train:]].reset_index(drop=True)

    def evaluate(df_eval):
        results = []
        with torch.no_grad():
            for _, row in df_eval.iterrows():
                yw = row['Yw_in'] if has_yw_l else YW_IN
                li = row['L_in'] if has_yw_l else L_IN

                if n_inputs == 6:
                    sensors = torch.tensor([
                        (float(row['TL_in'])  - 300.0) / 15.0,
                        (float(row['TG_out']) - 290.0) / 10.0,
                        (float(row['TG_in'])  - 285.0) / 10.0,
                        (float(row['TL_out']) - 295.0) / 10.0,
                        (float(yw)  - 0.012) / 0.008,
                        (float(li)  - 6.0)   / 2.0,
                    ], dtype=torch.float32).unsqueeze(0)
                else:
                    sensors = torch.tensor([
                        (float(row['TL_in'])  - 300.0) / 15.0,
                        (float(row['TG_in'])  - 285.0) / 10.0,
                        (float(row['TL_out']) - 295.0) / 10.0,
                        (float(yw)  - 0.012) / 0.008,
                        (float(li)  - 6.0)   / 2.0,
                    ], dtype=torch.float32).unsqueeze(0)

                kyam_pred = kyam_net(sensors).item()
                results.append({
                    'kyam_true': row['kyam_true'],
                    'kyam_pred': kyam_pred,
                    'error': abs(kyam_pred - row['kyam_true']),
                })
        return results

    train_res = evaluate(df_train)
    test_res = evaluate(df_test)

    train_errs = [r['error'] for r in train_res]
    test_errs = [r['error'] for r in test_res]
    train_rel = [r['error'] / r['kyam_true'] * 100 for r in train_res]
    test_rel = [r['error'] / r['kyam_true'] * 100 for r in test_res]

    print(f"NN Pura — Treino:  MAE={np.mean(train_errs):.4f}  MRE={np.mean(train_rel):.2f}%")
    print(f"NN Pura — Teste:   MAE={np.mean(test_errs):.4f}  MRE={np.mean(test_rel):.2f}%")


    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    kt_tr = [r['kyam_true'] for r in train_res]
    kp_tr = [r['kyam_pred'] for r in train_res]
    kt_te = [r['kyam_true'] for r in test_res]
    kp_te = [r['kyam_pred'] for r in test_res]

    ax.scatter(kt_tr, kp_tr, c='#378ADD', s=20, alpha=0.5,
               edgecolors='k', linewidths=0.2, label='Treino')
    ax.scatter(kt_te, kp_te, c='#E24B4A', s=35, alpha=0.7,
               edgecolors='k', linewidths=0.4, label='Teste')

    all_v = kt_tr + kt_te + kp_tr + kp_te
    lims = [min(all_v) - 0.3, max(all_v) + 0.3]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel('$K_Y a_M$ real', fontsize=13)
    ax.set_ylabel('$K_Y a_M$ predito', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSalvo: {args.output}")


if __name__ == '__main__':
    main()
