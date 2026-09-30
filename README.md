# PINN — Torre de Resfriamento

Rede Neural com Informação Física (Physics-Informed Neural Network) para diagnóstico de torres de resfriamento contrafluxo. O sistema infere o coeficiente volumétrico de transferência de massa `kyam` a partir de leituras de 4 sensores e classifica a condição operacional do equipamento.

---

## Estrutura do projeto

```
PINNs-Algoritmo/
├── app_gradio.py                  # Interface principal — PINN (execute este)
├── requirements.txt               # Dependências pip
│
├── core/                          # Módulo central da PINN
│   ├── config.py                  # Constantes físicas e hiperparâmetros
│   ├── physics.py                 # Solver ODE (método do tiro, RK45)
│   ├── pinn_model.py              # Arquiteturas ProfileNet e KyamNet
│   └── pinn_train.py              # Treinamento, inferência e diagnóstico
│
├── nn_pura/                       # Baseline sem restrições físicas
│   └── app_nn_pura.py             # Interface Gradio da NN pura
│
├── plots/                         # Scripts para figuras de publicação
│   ├── plot_artigo.py             # Convergência + real vs predito (PINN)
│   ├── plot_artigo_nn_pura.py     # Real vs predito (NN pura)
│   ├── plot_boxplot_comparacao.py # Boxplot PINN vs NN pura
│   ├── plot_perfis_torre.py       # Mapa de calor de temperatura
│   └── plot_pesos_epocas.py       # Schedule de pesos adaptativos
│
├── weights/                       # Pesos treinados (gerados ao treinar)
│   ├── pinn_weights.pth           # Pesos da PINN
│   └── nn_pura_weights.pth        # Pesos da NN pura
│
├── outputs/                       # Figuras geradas pelos scripts de plots
│   ├── convergencia.pdf
│   ├── real_vs_predito.pdf
│   ├── real_vs_predito_nn_pura.pdf
│   ├── boxplot_comparacao.pdf
│   ├── pesos_epocas.pdf
│   └── mapa_calor_agua.png
│
└── workspace/
    ├── treino_3.0_5.0.csv                  # Dados de treino (kyam 3–5)
    └── dados_generalizacao_1.5_3.0.csv     # Dados de generalização (kyam 1.5–3)
```

As pastas `weights/` e `outputs/` são criadas automaticamente na primeira execução.

---

## Instalação

```bash
pip install -r requirements.txt
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Como usar

### 1. Interface PINN (principal)

```bash
python app_gradio.py
```

Abre no navegador em `http://localhost:7860`.

**Aba 1 — Treinar PINN:**
- Faça upload do CSV de treino (`workspace/treino_3.0_5.0.csv`)
- Ajuste épocas e pontos de colocação
- Clique em **Treinar e salvar pesos**
- Os pesos são salvos em `pinn_weights.pth`

**Aba 2 — Prever / Diagnosticar:**
- Insira as leituras dos 4 sensores (em Kelvin):
  - **S1** — `TL_in`: temperatura da água entrando no topo
  - **S2** — `TG_out`: temperatura do ar saindo pelo topo
  - **S3** — `TG_in`: temperatura do ar entrando pela base
  - **S4** — `TL_out`: temperatura da água saindo pela base
- Ajuste umidade `Yw_in` e vazão `L_in` se necessário
- A PINN infere `kyam` e exibe o diagnóstico:
  - `kyam ≥ 4.0` → **SAUDAVEL** (verde)
  - `2.5 ≤ kyam < 4.0` → **INCRUSTACAO** (laranja)
  - `kyam < 2.5` → **FALHA MECANICA** (vermelho)

---

### 2. Interface NN Pura (baseline de comparação)

```bash
python nn_pura/app_nn_pura.py
```

Mesma arquitetura da PINN, mas **sem física** (sem EDOs, sem pontos de colocação). Útil para demonstrar a superioridade da PINN na generalização fora da faixa de treino.

**Aba 1 — Treinar NN Pura:** idêntico ao fluxo da PINN, sem physics loss.  
**Aba 2 — Validar Generalização:** avalia em CSV com `kyam` fora do intervalo de treino.

---

### 3. Scripts de figuras (para artigos/relatórios)

Todos os scripts devem ser executados **a partir da raiz do projeto**:

```bash
# Convergência + scatter real vs predito da PINN
python plots/plot_artigo.py --csv workspace/treino_3.0_5.0.csv --weights pinn_weights.pth

# Para treinar do zero e gerar os gráficos em seguida:
python plots/plot_artigo.py --csv workspace/treino_3.0_5.0.csv --treinar --epochs 10000

# Scatter real vs predito da NN pura
python plots/plot_artigo_nn_pura.py --csv workspace/treino_3.0_5.0.csv --weights nn_pura_weights.pth

# Boxplot comparativo PINN vs NN pura (dados fora da faixa de treino)
python plots/plot_boxplot_comparacao.py --csv workspace/dados_generalizacao_1.5_3.0.csv

# Mapa de calor dos perfis de temperatura na torre
python plots/plot_perfis_torre.py --weights pinn_weights.pth --tl_in 307 --tg_out 296 --tg_in 285 --tl_out 295

# Schedule de pesos adaptativos (não precisa de CSV nem pesos)
python plots/plot_pesos_epocas.py
```

Todos os PDFs/PNGs são salvos na pasta onde o script é executado (raiz do projeto por padrão).

---

## Formato do CSV

| Coluna      | Descrição                                      | Unidade |
|-------------|------------------------------------------------|---------|
| `sample_id` | Identificador da amostra                       | —       |
| `kyam_true` | Coeficiente volumétrico real                   | kg/m³·s |
| `TL_in`     | Temperatura da água na entrada (topo)          | K       |
| `TG_out`    | Temperatura do ar na saída (topo)              | K       |
| `TG_in`     | Temperatura do ar na entrada (base)            | K       |
| `TL_out`    | Temperatura da água na saída (base)            | K       |
| `Yw_in`     | Razão de umidade do ar na entrada              | kg/kg   |
| `L_in`      | Vazão de água                                  | kg/m²·s |

Se `Yw_in` e `L_in` não estiverem no CSV, os valores nominais de `core/config.py` são usados.

Na PINN, `L_in` é a vazão no topo. A rede impõe `L(Z_T) = L_in` e `L(0) < L_in` pela forma da sua saída. O simulador sintético em `core/physics.py` ainda inicializa `L(0) = L_in`. Por isso, os CSVs e pesos existentes foram produzidos com uma condição de contorno diferente da usada agora no treinamento. Eles precisam ser regenerados e os modelos retreinados antes de uma validação física consistente.


**Função de loss (PINN):**
```
L = λ_f(t) · L_física  +  λ_in(t) · L_BC_entrada  +  λ_out · L_BC_saída
```

- `λ_f` cresce de 0.5 → 5.0 (física ganha importância gradualmente)
- `λ_in` decai de 450 → 150 (BCs de entrada relaxam à medida que a rede estabiliza)
- `λ_out = 500` constante (sinal de kyam sempre forte)

---

## Dependências

| Biblioteca   | Versão mínima | Finalidade                          |
|--------------|---------------|-------------------------------------|
| torch        | 2.0.0         | Redes neurais e autodiferenciação   |
| gradio       | 4.0.0         | Interface web interativa            |
| matplotlib   | 3.7.0         | Gráficos e visualizações            |
| numpy        | 1.24.0        | Computação numérica                 |
| pandas       | 2.0.0         | Leitura e manipulação de CSV        |
| scipy        | 1.10.0        | Integração numérica das EDOs        |
