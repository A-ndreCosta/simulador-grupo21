# INDUSTRIAL INTELLIGENCE — Smart Predictive Maintenance

Aplicação Streamlit do projeto académico **Tema 7 — Automação Inteligente e Deteção de Falhas**: sistema híbrido de dois níveis
(regras industriais + Machine Learning) sobre o dataset AI4I 2020. É uma ferramenta de **demonstração e análise**: não controla
máquinas e não substitui sistemas de segurança certificados.

## O que faz
- **Painel geral**, **Simulador manual** (cenários, explicação das regras, histórico e CSV), **Importar dados** (CSV/XLSX, mapeamento de colunas, relatório de erros),
  **Modelos & Avaliação** (4 modelos, híbrido R/G, τ_F1 e τ_R90 escolhidos no treino), **Simulação de séries** (identificadas como simuladas),
  **Perfis de máquina** (Trabalho / UCI / personalizados), **Fluxo de manutenção** (diagrama + registo preenchido por si), **Dicionário** e **Sobre**.
- Regras: PWF (potência < 3500 W ou > 9000 W), HDF (dT < 8,6 K e Rot < 1380 rpm), OSF (sobrecarga ≥ limite do tipo), TWF opcional (desativada por defeito). RNF não tem regra.
- Fórmulas (um único sítio, `src/features.py`): dT = T_proc − T_ar; Potência = Torque·Rot·2π/60; Sobrecarga = Desgaste·Torque.

## Estrutura
```
app.py            ponto de entrada          src/   features, rules, validation, file_import,
ui/               páginas e tema                   ml_model, hybrid, detection, metrics,
config/           machine_profiles.json            simulation, maintenance, config
data/             ai4i2020.csv (UCI, sintético, opcional), demo_sintetico.csv, modelo_importacao.csv
tests/            suite pytest
```

## Instalar e executar — macOS
Requer Python 3.11 ou superior (testado com 3.13). No Terminal, dentro da pasta do projeto:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest              # opcional: correr os testes (cerca de 1 minuto)
python -m streamlit run app.py
```
O browser abre em `http://localhost:8501`. Para parar: `Ctrl+C` no Terminal. Para sair do ambiente: `deactivate`.
No VS Code: abra a pasta, escolha o interpretador `.venv` (Cmd+Shift+P → *Python: Select Interpreter*) e use o terminal integrado.

## Instalar e executar — Windows (opcional)
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
py -m streamlit run app.py
```

## Problemas comuns
- `command not found: python` → use `python3`. `No module named streamlit` → ative o ambiente (`source .venv/bin/activate`) e reinstale.
- Porta ocupada → `python -m streamlit run app.py --server.port 8502`.
- O treino demora cerca de 20 s no AI4I completo; os modelos são refeitos se mudar dados ou perfil.
- Perfis, históricos e registos ficam só na sessão do browser; exporte em JSON/CSV para guardar.

## Honestidade científica
- AI4I 2020 é **sintético** (UCI) e cada linha é independente: não é série temporal. Séries simuladas na app **não** validam antecipação temporal.
- Alerta ≠ falha. Sem validação em máquina real.
- Hiperparâmetros fixos (escolhidos por validação cruzada no notebook original); a «probabilidade» do SVM é uma sigmóide da função de decisão, não calibrada.
- Resultados de referência (perfil Trabalho, semente 42, teste de 2000 leituras): regras críticas em todo o dataset TP 287 / FP 0 / FN 52;
  RF só sensores P 0,714 / R 0,662 / F1 0,687; RF sensores+derivadas F1 0,882; híbrido R com τ_R90 recall 0,882 / precisão 0,556. Estes valores são verificados por `tests/test_regression_real.py`.
