# gerar_relatorio.py
# ==============================================================================
# Pipeline de DEMONSTRAÇÃO de uma rede neural multi-ômica de 3 ramos
# (longitudinal BiLSTM + 2 ramos estáticos + atenção cruzada) para classificação
# de resistência a BRAFi/MEKi em melanoma, com geração de um relatório em PDF.
#
# ⚠️ IMPORTANTE — DADOS SINTÉTICOS:
#   Os dados são GERADOS POR SIMULAÇÃO (não são dados reais de PRIDE/CPTAC/TCGA).
#   As dimensões são apenas inspiradas nesses repositórios. Os resultados são
#   ILUSTRATIVOS, servindo para demonstrar a arquitetura e o fluxo do pipeline.
#
# Esta versão corrige os problemas técnicos E metodológicos da especificação
# original (docs/forbuildNN.txt):
#   • Runtime/API: fig.line inexistente -> helper Line2D; set_font_size -> set_fontsize;
#     precision_recall_curve chamado 1x; avisos de log2 silenciados.
#   • Metodologia: divisão treino/teste estratificada (antes treino==teste);
#     imputação/padronização ajustadas SOMENTE no treino (antes havia vazamento);
#     rótulos derivados de um sinal latente real (antes: paridade do índice);
#     amostra maior + modelo menor + dropout/weight-decay (antes: overfit ~8000:1).
#   • Integridade: dados rotulados como sintéticos; sumário de execução factual
#     (sem os carimbos fixos "Verificado"/"execução limpa"); comentários corrigidos.
# ==============================================================================
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             f1_score, matthews_corrcoef, silhouette_score)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")  # backend não interativo (não bloqueia/abre janela)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx

# Reprodutibilidade
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Dimensões da coorte sintética (inspiradas em PRIDE/CPTAC/TCGA)
N_AMOSTRAS = 400          # amostras simuladas
PONTOS_TEMPORAIS = [6, 24, 48, 72]
N_FEAT_LONG = 32          # fosfosítios longitudinais (ramo BiLSTM)
N_FEAT_PROT = 24          # features proteômicas estáticas
N_FEAT_GEN = 8            # features genômicas estáticas (inclui BRAF V600E)
DIM_OCULTA = 16


def _linha_figura(fig, p0, p1, **kwargs):
    """Desenha uma linha em coordenadas de figura (0–1).
    matplotlib.Figure não possui método .line(); usamos Line2D + add_artist."""
    fig.add_artist(Line2D([p0[0], p1[0]], [p0[1], p1[1]], transform=fig.transFigure, **kwargs))


# ==============================================================================
# 1. GERAÇÃO DE DADOS SINTÉTICOS (coorte única e coerente)
# ==============================================================================
def gerar_coorte_sintetica():
    """
    Gera UMA coorte sintética coerente em que cada amostra possui as três
    modalidades medidas sobre a MESMA unidade simulada (evitando o vínculo
    biologicamente impossível entre coortes distintas):
      - longitudinal: (N, T, N_FEAT_LONG)  — perfil ao longo de 6/24/48/72h
      - proteômica  : (N, N_FEAT_PROT)      — features estáticas
      - genômica    : (N, N_FEAT_GEN)       — features estáticas (col 0 = BRAF V600E)
    O rótulo de resistência é uma função de um SINAL LATENTE multimodal + ruído,
    de modo que o modelo precisa de fato aprender padrões (e não uma regra trivial).
    """
    print("[INFO] Gerando coorte SINTÉTICA (dados simulados, não reais)...")
    rng = np.random.default_rng(SEED)

    # Genômica: BRAF V600E (binário) + features contínuas
    braf = rng.integers(0, 2, size=N_AMOSTRAS)
    gen_cont = rng.normal(0, 1, size=(N_AMOSTRAS, N_FEAT_GEN - 1))
    X_gen = np.column_stack([braf, gen_cont]).astype(np.float64)

    # Proteômica estática
    X_prot = rng.normal(0, 1, size=(N_AMOSTRAS, N_FEAT_PROT))

    # Longitudinal: linha de base + trajetória temporal com leve deriva
    base = rng.normal(0, 1, size=(N_AMOSTRAS, N_FEAT_LONG))
    X_long = np.empty((N_AMOSTRAS, len(PONTOS_TEMPORAIS), N_FEAT_LONG))
    for t in range(len(PONTOS_TEMPORAIS)):
        X_long[:, t, :] = base + 0.12 * t + rng.normal(0, 0.4, size=(N_AMOSTRAS, N_FEAT_LONG))

    # Sinal latente de resistência (combina as 3 modalidades) + ruído
    sinal_long = X_long[:, 2:, :4].mean(axis=(1, 2))   # tempos tardios, 4 primeiros sítios
    sinal_prot = X_prot[:, 0]
    score = 1.1 * sinal_long + 0.9 * sinal_prot + 1.0 * braf + rng.normal(0, 1.0, size=N_AMOSTRAS)
    y = (score > np.median(score)).astype(np.int64)    # ~50/50, não perfeitamente separável

    # Valores ausentes típicos de MS (DIA), injetados na matriz longitudinal
    mask = rng.random(X_long.shape) < 0.03
    X_long_missing = X_long.copy()
    X_long_missing[mask] = np.nan

    print(f"[INFO] Coorte: {N_AMOSTRAS} amostras | resistentes={int(y.sum())} sensíveis={int((y==0).sum())} "
          f"| ausentes={int(mask.sum())} ({100*mask.mean():.1f}%).")
    return X_long_missing, X_prot, X_gen, y


# ==============================================================================
# 2. PRÉ-PROCESSAMENTO SEM VAZAMENTO (ajuste somente no treino)
# ==============================================================================
def preparar_dados(X_long, X_prot, X_gen, y):
    """
    Divide em treino/teste (estratificado) e ajusta imputação + padronização
    APENAS no conjunto de treino, aplicando as mesmas transformações ao teste.
    Isto evita vazamento de informação do teste para o treino.
    """
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)

    N, T, P = X_long.shape
    long_flat = X_long.reshape(N, T * P)

    # Imputação KNN: ESTIMA os valores ausentes (não "recupera" os originais),
    # com o imputador ajustado somente nas amostras de treino.
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputer.fit(long_flat[idx_tr])
    long_flat = imputer.transform(long_flat)

    # Padronização por modalidade, ajustada somente no treino
    sc_long = StandardScaler().fit(long_flat[idx_tr])
    sc_prot = StandardScaler().fit(X_prot[idx_tr])
    sc_gen = StandardScaler().fit(X_gen[idx_tr])

    X_long_proc = sc_long.transform(long_flat).reshape(N, T, P)
    X_prot_proc = sc_prot.transform(X_prot)
    X_gen_proc = sc_gen.transform(X_gen)

    dataset = TensorDataset(
        torch.tensor(X_long_proc, dtype=torch.float32),
        torch.tensor(X_prot_proc, dtype=torch.float32),
        torch.tensor(X_gen_proc, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
    return Subset(dataset, idx_tr.tolist()), Subset(dataset, idx_te.tolist())


# ==============================================================================
# 3. ARQUITETURA: REDE DE TRÊS RAMOS COM ATENÇÃO CRUZADA (compacta + regularizada)
# ==============================================================================
class RedeMultiRamosMelanoma(nn.Module):
    def __init__(self, dim_long, dim_prot, dim_gen, dim_oculta=DIM_OCULTA, dropout=0.3):
        super().__init__()
        # Ramo 1 (longitudinal): BiLSTM compacta de 1 camada
        self.long_lstm = nn.LSTM(input_size=dim_long, hidden_size=dim_oculta,
                                 num_layers=1, batch_first=True, bidirectional=True)
        self.long_proj = nn.Linear(dim_oculta * 2, dim_oculta)
        self.long_bn = nn.BatchNorm1d(dim_oculta)

        # Ramo 2 (proteômica estática)
        self.prot_fc = nn.Sequential(nn.Linear(dim_prot, dim_oculta), nn.BatchNorm1d(dim_oculta),
                                     nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_oculta, dim_oculta))
        # Ramo 3 (genômica estática)
        self.gen_fc = nn.Sequential(nn.Linear(dim_gen, dim_oculta), nn.BatchNorm1d(dim_oculta),
                                    nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_oculta, dim_oculta))

        # Atenção cruzada por produto escalar escalonado (query = ramo longitudinal)
        self.camada_query = nn.Linear(dim_oculta, dim_oculta)
        self.key_prot = nn.Linear(dim_oculta, dim_oculta)
        self.key_gen = nn.Linear(dim_oculta, dim_oculta)
        self.value_prot = nn.Linear(dim_oculta, dim_oculta)
        self.value_gen = nn.Linear(dim_oculta, dim_oculta)
        self.escala = float(np.sqrt(dim_oculta))

        self.classificador = nn.Sequential(
            nn.Linear(dim_oculta * 3, dim_oculta), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_oculta, 1)
        )

    def forward(self, x_long, x_prot, x_gen, retornar_embeddings=False):
        lstm_out, _ = self.long_lstm(x_long)
        feat_long = F.relu(self.long_bn(self.long_proj(lstm_out[:, -1, :])))
        feat_prot = F.relu(self.prot_fc(x_prot))
        feat_gen = F.relu(self.gen_fc(x_gen))

        Q = self.camada_query(feat_long).unsqueeze(1)
        K = torch.stack([self.key_prot(feat_prot), self.key_gen(feat_gen)], dim=1)
        V = torch.stack([self.value_prot(feat_prot), self.value_gen(feat_gen)], dim=1)

        scores = torch.bmm(Q, K.transpose(1, 2)).squeeze(1) / self.escala
        pesos_atencao = F.softmax(scores, dim=-1).unsqueeze(2)
        feat_estatica_atendida = torch.sum(pesos_atencao * V, dim=1)

        z_fused = torch.cat((feat_long, feat_prot, feat_estatica_atendida), dim=1)
        logits = self.classificador(z_fused)
        if retornar_embeddings:
            return logits, z_fused
        return logits


# ==============================================================================
# 4. COMPILADOR DO RELATÓRIO: GERAÇÃO PROGRAMÁTICA DO PDF VETORIAL
# ==============================================================================
def compilar_pdf_vetorial_master(embeddings, alvos, metricas, perda_epocas, info_exec):
    """Monta um layout multipainel e o grava como PDF vetorial."""
    print("[INFO] Compilando o relatório em PDF...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    fig = plt.figure(figsize=(11, 15))

    # Cabeçalho
    fig.text(0.05, 0.96, "RELATÓRIO DE DEMONSTRAÇÃO — PIPELINE MULTI-ÔMICA (DADOS SINTÉTICOS)", fontsize=13.5, fontweight='bold', color='#1a365d')
    fig.text(0.05, 0.942, "Tema: Classificação de resistência a BRAFi/MEKi em melanoma — rede de 3 ramos com atenção cruzada", fontsize=10.5, style='italic', color='#4a5568')
    fig.text(0.05, 0.927, "Modalidades (sintéticas, dimensionadas conforme PRIDE longitudinal | CPTAC proteômica | TCGA genômica)", fontsize=9, fontweight='bold', color='#2b6cb0')
    fig.text(0.82, 0.96, f"Data: Junho de 2026\nStatus: {info_exec['status']}", fontsize=8.5, color='#718096')
    _linha_figura(fig, (0.05, 0.918), (0.95, 0.918), color='#1a365d', linewidth=2)

    # Disclaimer de dados sintéticos
    fig.text(0.05, 0.903, "⚠ Dados SIMULADOS para fins de demonstração — não são dados reais; métricas são ilustrativas.",
             fontsize=8.5, color='#9b2c2c', fontweight='bold')

    # Seção 1
    fig.text(0.05, 0.885, "1. Arquitetura de múltiplos ramos e protocolo de avaliação", fontsize=12, fontweight='bold', color='#1a365d')
    texto_resumo = (
        "Pipeline de demonstração com três ramos disjuntos integrados por atenção cruzada. "
        "O Ramo 1 modela a dinâmica longitudinal (6h, 24h, 48h, 72h) via LSTM bidirecional compacta; "
        "o Ramo 2 ingere features proteômicas estáticas; o Ramo 3 processa features genômicas basais (ex.: BRAF V600E). "
        "Os três ramos são fundidos por atenção cruzada por produto escalar escalonado para classificar o estado de resistência. "
        "Avaliação com divisão treino/teste estratificada 70/30; imputação (KNN) e padronização são ajustadas SOMENTE no treino "
        "e aplicadas ao teste, de modo a evitar vazamento de informação. As métricas reportadas referem-se ao conjunto de teste retido."
    )
    fig.text(0.05, 0.82, texto_resumo, fontsize=9.5, color='#2d3748', wrap=True,
             bbox=dict(facecolor='#f7fafc', edgecolor='#e2e8f0', boxstyle='round,pad=1'))

    # Seção 2 — perda + métricas (de teste)
    fig.text(0.05, 0.79, "2. Convergência do treino e métricas no conjunto de teste", fontsize=12, fontweight='bold', color='#1a365d')

    ax_perda = fig.add_axes([0.05, 0.65, 0.40, 0.11])
    ax_perda.plot(range(1, len(perda_epocas) + 1), perda_epocas, marker='o', color='#2b6cb0', linewidth=2)
    ax_perda.set_title("Convergência da perda de treino", fontsize=10, fontweight='bold', color='#2d3748')
    ax_perda.set_xlabel("Época", fontsize=8)
    ax_perda.set_ylabel("Perda BCE", fontsize=8)
    ax_perda.grid(True, linestyle=':', alpha=0.6)

    ax_tabela = fig.add_axes([0.52, 0.65, 0.43, 0.11])
    ax_tabela.axis('off')
    dados_metricas = [[m, f"{val:.4f}"] for m, val in metricas.items()]
    tabela = ax_tabela.table(cellText=dados_metricas, colLabels=["Métrica (conjunto de teste)", "Valor"], loc='center', cellLoc='left')
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1.0, 1.4)
    for (linha, col), celula in tabela.get_celld().items():
        if linha == 0:
            celula.set_text_props(weight='bold', color='white')
            celula.set_facecolor('#1a365d')
        else:
            celula.set_facecolor('#f7fafc' if linha % 2 == 0 else 'white')

    # Seção 3 — t-SNE + rede
    fig.text(0.05, 0.61, "3. Espaço latente (t-SNE) e mapa de sinalização de referência", fontsize=12, fontweight='bold', color='#1a365d')

    ax_tsne = fig.add_axes([0.05, 0.35, 0.42, 0.23])
    perplex = max(5, min(30, len(embeddings) - 1))
    coords = TSNE(n_components=2, perplexity=perplex, random_state=SEED, init="pca").fit_transform(embeddings)
    df_tsne = pd.DataFrame({
        "Dimensão 1": coords[:, 0], "Dimensão 2": coords[:, 1],
        "Fenótipo": ["Resistente" if t == 1 else "Sensível" for t in alvos]
    })
    sns.scatterplot(data=df_tsne, x="Dimensão 1", y="Dimensão 2", hue="Fenótipo",
                    palette={"Resistente": '#e53e3e', "Sensível": '#3182ce'},
                    s=70, edgecolors='black', alpha=0.85, ax=ax_tsne)
    ax_tsne.set_title("Espaço latente do conjunto de teste (t-SNE)", fontsize=9.5, fontweight='bold', color='#2d3748')
    ax_tsne.grid(True, linestyle='--', alpha=0.5)
    ax_tsne.legend(loc='best', fontsize=8)

    ax_rede = fig.add_axes([0.53, 0.35, 0.42, 0.23])
    nos = ['BRAF', 'MEK', 'ERK', 'AKT', 'IGF1R', 'MYC', 'PI3K', 'V600E']
    G = nx.Graph()
    G.add_nodes_from(nos)
    arestas = [('BRAF', 'MEK', 0.9), ('MEK', 'ERK', 0.85), ('V600E', 'BRAF', 0.95),
               ('AKT', 'PI3K', 0.75), ('IGF1R', 'PI3K', 0.82), ('AKT', 'MYC', 0.68),
               ('ERK', 'MYC', 0.5), ('PI3K', 'MEK', 0.45)]
    for u, v, w in arestas:
        G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, k=0.6, seed=SEED)
    cores = ['#fc8181' if n in ['BRAF', 'MEK', 'ERK', 'V600E'] else '#63b3ed' for n in nos]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=cores, edgecolors='black', ax=ax_rede)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax_rede)
    nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight'] * 4 for u, v in G.edges()],
                           edge_color='#a0aec0', alpha=0.8, ax=ax_rede)
    ax_rede.set_title("Mapa de sinalização de referência (curado da literatura)", fontsize=9.5, fontweight='bold', color='#2d3748')
    ax_rede.axis('off')

    # Seção 4 — notas factuais de execução (sem carimbos falsos)
    fig.text(0.05, 0.31, "4. Notas de execução e metodologia", fontsize=12, fontweight='bold', color='#1a365d')
    texto_codigo = (
        "Notas de execução (valores reais desta execução):\n"
        f" - Dados: SINTÉTICOS (simulados); {info_exec['n_total']} amostras (treino={info_exec['n_treino']}, teste={info_exec['n_teste']}).\n"
        " - Protocolo: divisão estratificada 70/30; imputação KNN e padronização ajustadas SOMENTE no treino.\n"
        f" - Modelo: rede de 3 ramos (BiLSTM + 2 MLP) com atenção cruzada; {info_exec['n_params']:,} parâmetros; dropout=0.3, weight_decay=1e-4.\n"
        f" - Treino: {info_exec['epocas']} épocas; perda final de treino = {info_exec['perda_final']:.4f}.\n"
        " - Rótulos derivados de um sinal latente multimodal + ruído (não triviais).\n"
        " - Aviso: dados simulados; métricas ILUSTRATIVAS, sem validade biológica/clínica."
    )
    fig.text(0.05, 0.12, texto_codigo, fontfamily='monospace', fontsize=8.5, color='#1a202c',
             bbox=dict(facecolor='#edf2f7', edgecolor='#cbd5e0', boxstyle='square,pad=1'))

    # Rodapé
    _linha_figura(fig, (0.05, 0.08), (0.95, 0.08), color='#718096', linewidth=0.75)
    fig.text(0.05, 0.055, "Pipeline de demonstração de biologia de sistemas multi-ômica para melanoma • Dados sintéticos", fontsize=8, color='#a0aec0')
    fig.text(0.88, 0.055, "Página 1 de 1", fontsize=8, color='#a0aec0')

    plt.savefig("Relatorio_Pipeline_MultiOmica_Melanoma.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SUCESSO] Relatório gerado: 'Relatorio_Pipeline_MultiOmica_Melanoma.pdf'.")


# ==============================================================================
# 5. ORQUESTRADOR DE EXECUÇÃO
# ==============================================================================
def executar_pipeline():
    # 1. Dados sintéticos
    X_long, X_prot, X_gen, y = gerar_coorte_sintetica()

    # 2. Pré-processamento sem vazamento + split treino/teste
    ds_treino, ds_teste = preparar_dados(X_long, X_prot, X_gen, y)
    carregador_treino = DataLoader(ds_treino, batch_size=16, shuffle=True, drop_last=True)
    carregador_teste = DataLoader(ds_teste, batch_size=32, shuffle=False)

    # 3. Modelo
    modelo = RedeMultiRamosMelanoma(dim_long=N_FEAT_LONG, dim_prot=N_FEAT_PROT, dim_gen=N_FEAT_GEN)
    n_params = sum(p.numel() for p in modelo.parameters())
    criterio = nn.BCEWithLogitsLoss()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=0.003, weight_decay=1e-4)

    # 4. Treino (apenas no conjunto de treino)
    print(f"\n[TREINAMENTO] {n_params:,} parâmetros | treino={len(ds_treino)} | teste={len(ds_teste)}")
    EPOCAS = 40
    perda_epocas = []
    for epoca in range(1, EPOCAS + 1):
        modelo.train()
        perda_total = 0.0
        n = 0
        for x_l, x_p, x_g, alvo in carregador_treino:
            otimizador.zero_grad()
            perda = criterio(modelo(x_l, x_p, x_g), alvo)
            perda.backward()
            otimizador.step()
            perda_total += perda.item() * x_l.size(0)
            n += x_l.size(0)
        perda_media = perda_total / max(n, 1)
        perda_epocas.append(perda_media)
        if epoca % 5 == 0 or epoca == 1:
            print(f"  Época {epoca:>2}/{EPOCAS} | perda de treino = {perda_media:.4f}")

    # 5. Avaliação NO CONJUNTO DE TESTE RETIDO
    modelo.eval()
    probs, alvos, embs = [], [], []
    with torch.no_grad():
        for x_l, x_p, x_g, alvo in carregador_teste:
            logits, emb = modelo(x_l, x_p, x_g, retornar_embeddings=True)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            alvos.extend(alvo.cpu().numpy())
            embs.extend(emb.cpu().numpy())

    y_true = np.array(alvos).squeeze()
    y_prob = np.array(probs).squeeze()
    y_pred = (y_prob >= 0.5).astype(int)
    embs = np.array(embs)

    precisao, revocacao, _ = precision_recall_curve(y_true, y_prob)
    metricas = {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "AUC Precisão-Revocação": auc(revocacao, precisao),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Correlação de Matthews (MCC)": matthews_corrcoef(y_true, y_pred),
        "Silhueta (espaço latente)": silhouette_score(embs, y_true),
    }
    print("[AVALIAÇÃO] Métricas no conjunto de teste:")
    for k, v in metricas.items():
        print(f"   {k:32s}: {v:.4f}")

    info_exec = {
        "status": "concluído",
        "n_total": N_AMOSTRAS,
        "n_treino": len(ds_treino),
        "n_teste": len(ds_teste),
        "n_params": n_params,
        "epocas": EPOCAS,
        "perda_final": perda_epocas[-1],
    }

    # 6. Relatório
    compilar_pdf_vetorial_master(embs, y_true, metricas, perda_epocas, info_exec)
    print("[INFO] Execução finalizada.\n")


if __name__ == "__main__":
    executar_pipeline()
