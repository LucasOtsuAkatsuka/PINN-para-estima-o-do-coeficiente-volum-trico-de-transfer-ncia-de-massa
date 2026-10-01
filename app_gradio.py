import os
import torch
import numpy as np
import traceback
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.pinn_train import (
    train_pinn, predict_single, diagnose,
    save_models, load_models,
)
from core.config import Z_T, YW_IN, L_IN

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'pinn_weights.pth')

_state = {
    'profile_net': None,
    'kyam_net': None,
}


def treinar(csv_file, epochs, n_colloc, progress=gr.Progress()):
    if csv_file is None:
        return "Erro: faça upload de um CSV.", None

    csv_path = csv_file.name if hasattr(csv_file, 'name') else str(csv_file)

    try:
        def prog_cb(epoch, total, loss):
            progress(epoch / total, desc=f"Época {epoch}/{total} — loss={loss:.4f}")

        profile_net, kyam_net, train_res, test_res, history = train_pinn(
            csv_path=csv_path,
            epochs=int(epochs),
            n_colloc=int(n_colloc),
            progress_callback=prog_cb,
        )

        _state['profile_net'] = profile_net
        _state['kyam_net'] = kyam_net
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        torch.save({
            'profile_net': profile_net.state_dict(),
            'kyam_net': kyam_net.state_dict(),
            'history': history,
            'train_res': train_res,
            'test_res': test_res,
        }, WEIGHTS_PATH)

        has_true = 'kyam_true' in train_res[0]
        resumo = f"Treinamento concluído!\n"
        resumo += f"Amostras treino: {len(train_res)} | teste: {len(test_res)}\n"
        resumo += f"Pesos salvos em: {WEIGHTS_PATH}\n\n"

        if has_true:
            train_errs = [r['error'] for r in train_res]
            test_errs  = [r['error'] for r in test_res]
            train_rel  = [r['error']/r['kyam_true']*100 for r in train_res]
            test_rel   = [r['error']/r['kyam_true']*100 for r in test_res]

            resumo += "TREINO:\n"
            resumo += f"  Erro abs. médio: {np.mean(train_errs):.4f}\n"
            resumo += f"  Erro rel. médio: {np.mean(train_rel):.2f}%\n\n"
            resumo += "TESTE (nunca vistos):\n"
            resumo += f"  Erro abs. médio: {np.mean(test_errs):.4f}\n"
            resumo += f"  Erro rel. médio: {np.mean(test_rel):.2f}%\n"
            resumo += f"  Erro máximo:     {np.max(test_errs):.4f}\n"
        else:
            kp = [r['kyam_pred'] for r in train_res + test_res]
            resumo += f"kyam inferido: [{min(kp):.2f}, {max(kp):.2f}]\n"


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


        ax1.semilogy(history['loss'], color='#333', linewidth=1, label='Total')
        ax1.semilogy(history['phys'], color='#E24B4A', linewidth=0.8, alpha=0.6, label='Física')
        if 'bc_in' in history:
            ax1.semilogy(history['bc_in'], color='#378ADD', linewidth=0.8, alpha=0.6, label='BC entrada')
            ax1.semilogy(history['bc_out'], color='#EF9F27', linewidth=1.2, alpha=0.8, label='BC saída (kyam)')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.set_title('Convergência')
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
            lims = [min(all_v)-0.5, max(all_v)+0.5]
            ax2.plot(lims, lims, 'k--', alpha=0.3)
            ax2.set_xlim(lims); ax2.set_ylim(lims)
            ax2.set_xlabel('kyam real'); ax2.set_ylabel('kyam predito')
            ax2.set_title('Real vs predito')
            ax2.legend()
            ax2.set_aspect('equal')
        else:
            ax2.text(0.5, 0.5, 'CSV sem kyam_true',
                     ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        return resumo, fig

    except Exception as e:
        erro = f"ERRO no treinamento:\n\n{str(e)}\n\n{traceback.format_exc()}"
        return erro, None


def _ensure_models():
    if _state['profile_net'] is not None:
        return True
    if os.path.exists(WEIGHTS_PATH):
        try:
            p, k = load_models(WEIGHTS_PATH)
            _state['profile_net'] = p
            _state['kyam_net'] = k
            return True
        except Exception:
            return False
    return False


def prever(TL_in, TG_out, TG_in, TL_out, Yw_in, L_in):
    if not _ensure_models():
        return "Erro: treine na Aba 1 primeiro.", "", None

    if TG_in >= TL_in:
        return "Erro: TG_in deve ser < TL_in.", "", None

    try:
        kyam, profiles = predict_single(
            _state['profile_net'], _state['kyam_net'],
            TL_in, TG_out, TG_in, TL_out,
            Yw_in=Yw_in, L_in=L_in,
        )

        status, _ = diagnose(kyam)

        resultado = f"kyam inferido: {kyam:.3f} kg/m³·s\n"
        resultado += f"Diagnóstico:   {status}\n\n"
        resultado += f"S1 — Água entrada (TL_in):  {TL_in:.1f} K ({TL_in-273.15:.1f} °C)\n"
        resultado += f"S2 — Ar saída    (TG_out):  {TG_out:.1f} K ({TG_out-273.15:.1f} °C)\n"
        resultado += f"S3 — Ar entrada  (TG_in):   {TG_in:.1f} K ({TG_in-273.15:.1f} °C)\n"
        resultado += f"S4 — Água saída  (TL_out):  {TL_out:.1f} K ({TL_out-273.15:.1f} °C)\n\n"

        rng = TL_in - TL_out
        app = TL_out - TG_in
        resultado += f"Range:      {rng:.1f} K\n"
        resultado += f"Approach:   {app:.1f} K\n"
        if rng + app > 0:
            resultado += f"Eficiência: {rng/(rng+app)*100:.1f}%\n"

        diag = f"{status} (kyam={kyam:.3f})"

        fig, ax = plt.subplots(figsize=(8, 6))
        z = profiles['z']
        ax.plot(profiles['TL']-273.15, z, '-', color='#E24B4A', linewidth=2.5, label='Água $T_L(z)$')
        ax.plot(profiles['TG']-273.15, z, '-', color='#378ADD', linewidth=2.5, label='Ar $T_G(z)$')

        ax.plot(TL_in-273.15, Z_T, 'v', color='#E24B4A', ms=12, mec='k', mew=0.8,
                label=f'S1: TL_in={TL_in-273.15:.1f}°C', zorder=5)
        ax.plot(TG_out-273.15, Z_T, '^', color='#378ADD', ms=12, mec='k', mew=0.8,
                label=f'S2: TG_out={TG_out-273.15:.1f}°C', zorder=5)
        ax.plot(TG_in-273.15, 0, '^', color='#85B7EB', ms=12, mec='k', mew=0.8,
                label=f'S3: TG_in={TG_in-273.15:.1f}°C', zorder=5)
        ax.plot(TL_out-273.15, 0, 'v', color='#993C1D', ms=12, mec='k', mew=0.8,
                label=f'S4: TL_out={TL_out-273.15:.1f}°C', zorder=5)

        ax.axhspan(0, 0.3, alpha=0.05, color='#378ADD')
        ax.axhspan(0.3, 1.5, alpha=0.05, color='#5DCAA5')
        ax.axhspan(1.5, Z_T, alpha=0.05, color='#EF9F27')
        ax.set_xlabel('Temperatura (°C)')
        ax.set_ylabel('Altura z (m)')
        ax.set_title(f'Perfis — kyam={kyam:.3f} — {status}')
        ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return resultado, diag, fig

    except Exception as e:
        return f"ERRO: {str(e)}\n\n{traceback.format_exc()}", "", None


with gr.Blocks(title="PINN — Torre de Resfriamento", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# PINN unificada — Torre de resfriamento")
    gr.Markdown("**ProfileNet (6 entradas, inclui kyam) + KyamNet** — loss separada: entrada / saída / física")

    with gr.Tab("1. Treinar PINN"):
        gr.Markdown("### Carregue o CSV, treine as redes N e K, pesos salvos automaticamente")
        with gr.Row():
            with gr.Column():
                csv_upload = gr.File(label="Upload CSV", file_types=['.csv'])
                epochs_sl = gr.Slider(1000, 10000, value=5000, step=500, label="Épocas")
                colloc_sl = gr.Slider(30, 200, value=50, step=10, label="Pontos de colocação")
                btn_treinar = gr.Button("Treinar e salvar pesos", variant="primary")
            with gr.Column():
                resumo_txt = gr.Textbox(label="Resultado", lines=16, interactive=False)
        plot_train = gr.Plot(label="Convergência e validação")
        btn_treinar.click(fn=treinar, inputs=[csv_upload, epochs_sl, colloc_sl],
                          outputs=[resumo_txt, plot_train])

    with gr.Tab("2. Prever / Diagnosticar"):
        gr.Markdown("### Insira as leituras dos 4 sensores → PINN infere kyam instantaneamente")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Topo da torre (z = Z_T):**")
                tl_in_sl = gr.Slider(295, 320, value=307, step=0.1,
                                      label="S1 — TL_in: Água entrando (K)")
                tg_out_sl = gr.Slider(280, 310, value=296, step=0.1,
                                       label="S2 — TG_out: Ar saindo (K)")
                gr.Markdown("**Base da torre (z = 0):**")
                tg_in_sl = gr.Slider(275, 300, value=286, step=0.1,
                                      label="S3 — TG_in: Ar entrando (K)")
                tl_out_sl = gr.Slider(280, 310, value=295, step=0.1,
                                       label="S4 — TL_out: Água saindo (K)")
                gr.Markdown("**Condições operacionais:**")
                yw_in_sl = gr.Slider(0.005, 0.020, value=YW_IN, step=0.001,
                                      label="Yw_in: Umidade do ar na entrada")
                l_in_sl = gr.Slider(4.0, 8.0, value=L_IN, step=0.1,
                                     label="L_in: Vazão de água (kg/m²·s)")
                btn_prever = gr.Button("Prever", variant="primary")
            with gr.Column():
                resultado_txt = gr.Textbox(label="Resultado", lines=14, interactive=False)
                diag_txt = gr.Textbox(label="Diagnóstico", interactive=False)
        plot_perfis = gr.Plot(label="Perfis de temperatura")
        btn_prever.click(fn=prever, inputs=[tl_in_sl, tg_out_sl, tg_in_sl, tl_out_sl,
                                                yw_in_sl, l_in_sl],
                         outputs=[resultado_txt, diag_txt, plot_perfis])

if __name__ == '__main__':
    demo.launch(share=False)
