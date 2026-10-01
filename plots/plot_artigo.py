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

from core.pinn_train import train_pinn, load_models
from core.pinn_model import normalize_sensors, KyamNet
from core.config import YW_IN, L_IN, PINN_TRAIN_SPLIT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--weights', default='weights/pinn_weights.pth')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--treinar', action='store_true',
                        help='Treina do zero (se nao, usa pesos existentes)')
    args = parser.parse_args()

    if args.treinar:
        print("Treinando...")
        profile_net, kyam_net, train_res, test_res, history = train_pinn(
            csv_path=args.csv, epochs=args.epochs,
            progress_callback=lambda e, t, l: print(f"  {e}/{t} loss={l:.4f}") if e % 500 == 0 else None
        )
        os.makedirs(os.path.dirname(args.weights), exist_ok=True)
        torch.save({
            'profile_net': profile_net.state_dict(),
            'kyam_net': kyam_net.state_dict(),
            'history': history,
            'train_res': train_res,
            'test_res': test_res,
        }, args.weights)
        print(f"Pesos + historico salvos: {args.weights}")
    else:
        print(f"Carregando: {args.weights}")
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)

        if 'history' not in checkpoint:
            print("ERRO: arquivo de pesos nao contem historico.")
            print("Rode com --treinar para gerar historico.")
            return

        history = checkpoint['history']


        kyam_net = KyamNet()
        kyam_net.load_state_dict(checkpoint['kyam_net'])
        kyam_net.eval()

        df = pd.read_csv(args.csv)
        has_true = 'kyam_true' in df.columns
        has_yw_l = 'Yw_in' in df.columns and 'L_in' in df.columns

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
                    sensors = normalize_sensors(
                        row['TL_in'], row['TG_out'], row['TG_in'],
                        row['TL_out'], yw, li
                    )
                    kyam_pred = kyam_net(sensors).item()
                    r = {'kyam_pred': kyam_pred}
                    if has_true:
                        r['kyam_true'] = row['kyam_true']
                        r['error'] = abs(kyam_pred - row['kyam_true'])
                    results.append(r)
            return results

        train_res = evaluate(df_train)
        test_res = evaluate(df_test)

    has_true = 'kyam_true' in train_res[0]


    fig1, ax1 = plt.subplots(figsize=(4.5, 3.5))

    ax1.semilogy(history['loss'], color='#333', linewidth=1, label='Total')
    ax1.semilogy(history['phys'], color='#E24B4A', linewidth=0.8, alpha=0.6, label='Fisica')
    if 'bc_in' in history:
        ax1.semilogy(history['bc_in'], color='#378ADD', linewidth=0.8, alpha=0.6, label='BC entrada')
        ax1.semilogy(history['bc_out'], color='#EF9F27', linewidth=1.2, alpha=0.8, label='BC saida')

    ax1.set_xlabel('Epoca', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.tick_params(axis='both', labelsize=9)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    os.makedirs('outputs', exist_ok=True)
    plt.tight_layout()
    fig1.savefig('outputs/convergencia.pdf', bbox_inches='tight')
    plt.close(fig1)
    print("Salvo: outputs/convergencia.pdf")


    if has_true:
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.0))

        kt_tr = [r['kyam_true'] for r in train_res]
        kp_tr = [r['kyam_pred'] for r in train_res]
        kt_te = [r['kyam_true'] for r in test_res]
        kp_te = [r['kyam_pred'] for r in test_res]

        ax2.scatter(kt_tr, kp_tr, c='#378ADD', s=20, alpha=0.5,
                    edgecolors='k', linewidths=0.2, label='Treino')
        ax2.scatter(kt_te, kp_te, c='#E24B4A', s=35, alpha=0.7,
                    edgecolors='k', linewidths=0.4, label='Teste')

        all_v = kt_tr + kt_te + kp_tr + kp_te
        lims = [min(all_v) - 0.3, max(all_v) + 0.3]
        ax2.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)

        ax2.set_xlabel('$K_Y a_M$ real', fontsize=13)
        ax2.set_ylabel('$K_Y a_M$ predito', fontsize=13)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.legend(fontsize=12)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        test_errs = [r['error'] for r in test_res]
        test_rel = [r['error'] / r['kyam_true'] * 100 for r in test_res]
        print(f"  Teste: MAE={np.mean(test_errs):.4f}  MRE={np.mean(test_rel):.2f}%")

        plt.tight_layout()
        fig2.savefig('outputs/real_vs_predito.pdf', bbox_inches='tight')
        plt.close(fig2)
        print("Salvo: outputs/real_vs_predito.pdf")
    else:
        print("CSV sem kyam_true, scatter nao gerado.")


if __name__ == '__main__':
    main()
