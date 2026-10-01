import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from matplotlib import patheffects

from core.pinn_train import predict_single, load_models, diagnose
from core.config import Z_T


def main():
    parser = argparse.ArgumentParser(description='Mapa de calor da torre (Foco na Água)')
    parser.add_argument('--weights', default='weights/pinn_weights.pth')
    parser.add_argument('--tl_in', type=float, default=307.0)
    parser.add_argument('--tg_out', type=float, default=296.0)
    parser.add_argument('--tg_in', type=float, default=285.0)
    parser.add_argument('--tl_out', type=float, default=295.0)
    parser.add_argument('--yw_in', type=float, default=0.012)
    parser.add_argument('--l_in', type=float, default=5.8)
    parser.add_argument('--output', default='outputs/mapa_calor_agua.png')
    args = parser.parse_args()


    print(f"Carregando: {args.weights}")
    profile_net, kyam_net = load_models(args.weights)

    kyam, profiles = predict_single(
        profile_net, kyam_net,
        args.tl_in, args.tg_out, args.tg_in, args.tl_out,
        Yw_in=args.yw_in, L_in=args.l_in,
    )

    z  = profiles['z']
    TL = profiles['TL'] - 273.15

    status, _ = diagnose(kyam)


    T_min = TL.min()
    T_max = TL.max()


    cmap = plt.cm.coolwarm
    norm = mcolors.Normalize(vmin=T_min - 1, vmax=T_max + 1)


    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_aspect('equal')


    tower_left = 1.0
    tower_right = 5.0
    tower_bottom = 1.0
    tower_top = tower_bottom + 8.0
    tower_w = tower_right - tower_left
    tower_h = tower_top - tower_bottom
    mid_x = (tower_left + tower_right) / 2


    n_strips = len(z)
    z_norm = (z - z.min()) / (z.max() - z.min())

    for i in range(n_strips - 1):
        y_bot = tower_bottom + z_norm[i] * tower_h
        y_top_strip = tower_bottom + z_norm[i + 1] * tower_h
        strip_h = y_top_strip - y_bot


        color_water = cmap(norm(TL[i]))
        rect_w = patches.Rectangle(
            (tower_left, y_bot), tower_w, strip_h,
            facecolor=color_water, edgecolor='none', zorder=2,
        )
        ax.add_patch(rect_w)


    tower_outline = patches.FancyBboxPatch(
        (tower_left, tower_bottom), tower_w, tower_h,
        boxstyle="round,pad=0.05",
        facecolor='none', edgecolor='#3a3a3a', linewidth=2.5, zorder=5,
    )
    ax.add_patch(tower_outline)


    stroke = [patheffects.withStroke(linewidth=3, foreground='white', alpha=0.9)]


    sensor_props = dict(fontsize=9, fontweight='bold', zorder=15,
                        path_effects=stroke)


    ax.text(tower_left - 0.2, tower_top,
            f'Entrada da água: {args.tl_in - 273.15:.1f}°C',
            ha='right', va='center', color='#C0392B', **sensor_props)


    ax.text(tower_left - 0.2, tower_bottom,
            f'Saída da água: {args.tl_out - 273.15:.1f}°C',
            ha='right', va='center', color='#993C1D', **sensor_props)


    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20, pad=0.05)
    cbar.set_label('Temperatura da Água (°C)', fontsize=11, fontweight='bold')


    ax.set_title(f'Mapa de Calor da Água na Torre',
                 fontsize=14, fontweight='bold', color='#333', pad=25)


    ax.set_xlim(-2.0, 8.0)
    ax.set_ylim(-0.5, tower_top + 1.2)
    ax.axis('off')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\nGrafico salvo: {args.output}")


if __name__ == '__main__':
    main()
