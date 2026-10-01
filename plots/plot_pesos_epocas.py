import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.config import PINN_EPOCHS, PINN_W_PHYS, PINN_W_BC_IN, PINN_W_BC_OUT


def main():
    epochs = PINN_EPOCHS
    w_phys = PINN_W_PHYS
    w_bc_in = PINN_W_BC_IN
    w_bc_out = PINN_W_BC_OUT


    epoch_range = np.arange(epochs)
    prog = epoch_range / max(epochs - 1, 1)

    J_f_hist = w_phys * (0.1 + 0.9 * prog)
    J_in_hist = w_bc_in * (1.0 + 2.0 * (1.0 - prog))
    J_out_hist = np.full(epochs, w_bc_out)


    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.plot(epoch_range, J_out_hist, color='#EF9F27', linewidth=2.5,
            label=r'$\lambda_{out}$ (BC saida)')
    ax.plot(epoch_range, J_in_hist, color='#378ADD', linewidth=2.5,
            label=r'$\lambda_{in}$ (BC entrada)')
    ax.plot(epoch_range, J_f_hist, color='#E24B4A', linewidth=2.5,
            label=r'$\lambda_f$ (fisica)')

    ax.set_xlabel('Epoca', fontsize=11)
    ax.set_ylabel('Peso', fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(0.3, 800)
    ax.set_xlim(0, epochs)
    ax.tick_params(axis='both', labelsize=9)
    ax.legend(loc='center left', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both')

    os.makedirs('outputs', exist_ok=True)
    plt.tight_layout()
    fig.savefig('outputs/pesos_epocas.pdf', bbox_inches='tight')
    plt.close(fig)

    print("Grafico salvo: outputs/pesos_epocas.pdf")
    print(f"\nValores iniciais (epoca 0):")
    print(f"  lambda_f   = {J_f_hist[0]:.2f}")
    print(f"  lambda_in  = {J_in_hist[0]:.2f}")
    print(f"  lambda_out = {J_out_hist[0]:.2f}")
    print(f"\nValores finais (epoca {epochs-1}):")
    print(f"  lambda_f   = {J_f_hist[-1]:.2f}")
    print(f"  lambda_in  = {J_in_hist[-1]:.2f}")
    print(f"  lambda_out = {J_out_hist[-1]:.2f}")


if __name__ == '__main__':
    main()
