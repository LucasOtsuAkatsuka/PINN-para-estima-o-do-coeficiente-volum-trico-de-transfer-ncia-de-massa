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

from core.config import YW_IN, L_IN


def predict_all(kyam_net, df, n_inputs):
    has_yw_l = 'Yw_in' in df.columns and 'L_in' in df.columns
    kyam_real, kyam_pred = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
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
            kyam_pred.append(kyam_net(sensors).item())
            kyam_real.append(row['kyam_true'])
    return np.array(kyam_real), np.array(kyam_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--weights_pinn', default='weights/pinn_weights.pth')
    parser.add_argument('--weights_nn', default='weights/nn_pura_weights.pth')
    parser.add_argument('--output', default='outputs/boxplot_comparacao.pdf')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"CSV: {len(df)} amostras, kyam [{df['kyam_true'].min():.2f}, {df['kyam_true'].max():.2f}]")


    from core.pinn_model import KyamNet as KyamNet_PINN
    ck_pinn = torch.load(args.weights_pinn, map_location='cpu', weights_only=False)
    kyam_net_pinn = KyamNet_PINN()
    kyam_net_pinn.load_state_dict(ck_pinn['kyam_net'])
    kyam_net_pinn.eval()
    n_in_pinn = list(kyam_net_pinn.parameters())[0].shape[1]
    k_real, k_pred_pinn = predict_all(kyam_net_pinn, df, n_in_pinn)
    erros_pinn = np.abs(k_pred_pinn - k_real)


    from nn_pura.app_nn_pura import KyamNet_NN
    ck_nn = torch.load(args.weights_nn, map_location='cpu', weights_only=False)
    kyam_net_nn = KyamNet_NN()
    kyam_net_nn.load_state_dict(ck_nn['kyam_net'])
    kyam_net_nn.eval()
    n_in_nn = list(kyam_net_nn.parameters())[0].shape[1]
    _, k_pred_nn = predict_all(kyam_net_nn, df, n_in_nn)
    erros_nn = np.abs(k_pred_nn - k_real)

    mae_pinn = np.mean(erros_pinn)
    mre_pinn = np.mean(erros_pinn / k_real * 100)
    mae_nn = np.mean(erros_nn)
    mre_nn = np.mean(erros_nn / k_real * 100)

    print(f"PINN:    MAE={mae_pinn:.4f}  MRE={mre_pinn:.1f}%")
    print(f"NN pura: MAE={mae_nn:.4f}  MRE={mre_nn:.1f}%")

    fig, ax = plt.subplots(figsize=(3.5, 4.0))

    bp = ax.boxplot(
        [erros_pinn, erros_nn],
        labels=[f'Caixa cinza', f'Caixa preta'],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color='k', linewidth=1.5),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
        flierprops=dict(marker='o', markersize=4, alpha=0.5),
    )

    bp['boxes'][0].set_facecolor('#5DCAA5')
    bp['boxes'][0].set_edgecolor('#0F6E56')
    bp['boxes'][0].set_alpha(0.7)

    bp['boxes'][1].set_facecolor('#B0B0B0')
    bp['boxes'][1].set_edgecolor('#666666')
    bp['boxes'][1].set_alpha(0.7)

    ax.set_ylabel('Erro absoluto $|K_Y a_M^{predito} - K_Y a_M^{real}|$', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.2, axis='y')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSalvo: {args.output}")


if __name__ == '__main__':
    main()
