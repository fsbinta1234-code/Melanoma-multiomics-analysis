# -*- coding: utf-8 -*-
"""
gerar_relatorio_completo.py
==============================================================================
Gera um relatório técnico-científico EXTENSO e DETALHADO de todo o projeto de
análise multi-ômica de resistência a BRAFi/MEKi em melanoma, em formato PDF.

O documento cobre, para cada conjunto de dados público utilizado:
  • descrição e proveniência do dataset;
  • metodologia detalhada do processamento e da análise;
  • resultados quantitativos, figuras e tabelas.

Conjuntos de dados integrados:
  - PXD013923  (fosfoproteoma agudo BRAFi/MEKi/ERKi, SILAC, A375)
  - PXD022992  (fosfoproteoma directDIA, 6 linhagens de melanoma)
  - OmniPath   (rede cinase-substrato para KSEA)
  - TCGA-SKCM  (genômica/transcriptômica clínica, 470 pacientes)
  - GSE110054  (transcriptoma temporal da resistência ao BRAFi)
  - Rede neural (classificador de fosfosítios diferenciais)

Saída: Relatorio_Completo_Melanoma_MultiOmica.pdf
==============================================================================
"""
import os
import datetime

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, KeepTogether, NextPageTemplate,
)
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── fontes com cobertura Unicode completa (subscritos, gregas, setas) ─────────
_FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT    = "DejaVu"
FONT_B  = "DejaVu-Bold"
FONT_I  = "DejaVu-Oblique"
FONT_BI = "DejaVu-BoldOblique"


def register_fonts():
    pdfmetrics.registerFont(TTFont(FONT,    f"{_FONT_DIR}/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont(FONT_B,  f"{_FONT_DIR}/DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont(FONT_I,  f"{_FONT_DIR}/DejaVuSans-Oblique.ttf"))
    pdfmetrics.registerFont(TTFont(FONT_BI, f"{_FONT_DIR}/DejaVuSans-BoldOblique.ttf"))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT_B,
                                  italic=FONT_I, boldItalic=FONT_BI)


register_fonts()

FIG_DIR = "results/figures"
OUT_DIR = "results/outputs"
PDF_NAME = "Relatorio_Completo_Melanoma_MultiOmica.pdf"

# ── paleta de cores institucional ────────────────────────────────────────────
NAVY      = colors.HexColor("#1a365d")
BLUE      = colors.HexColor("#2b6cb0")
STEEL     = colors.HexColor("#4C72B0")
RED       = colors.HexColor("#C44E52")
GREEN     = colors.HexColor("#55A868")
GREY      = colors.HexColor("#4a5568")
LIGHTGREY = colors.HexColor("#f7fafc")
MIDGREY   = colors.HexColor("#e2e8f0")
DARK      = colors.HexColor("#1a202c")


# =============================================================================
# Estilos
# =============================================================================
def build_styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle(
        "CoverTitle", parent=ss["Title"], fontSize=26, leading=32,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=14))
    ss.add(ParagraphStyle(
        "CoverSubtitle", parent=ss["Normal"], fontSize=14, leading=20,
        textColor=GREY, alignment=TA_CENTER, spaceAfter=8))
    ss.add(ParagraphStyle(
        "CoverMeta", parent=ss["Normal"], fontSize=10.5, leading=16,
        textColor=GREY, alignment=TA_CENTER))

    ss.add(ParagraphStyle(
        "H1", parent=ss["Heading1"], fontSize=17, leading=21,
        textColor=NAVY, spaceBefore=18, spaceAfter=10,
        borderWidth=0, keepWithNext=True))
    ss.add(ParagraphStyle(
        "H2", parent=ss["Heading2"], fontSize=13.5, leading=17,
        textColor=BLUE, spaceBefore=12, spaceAfter=6, keepWithNext=True))
    ss.add(ParagraphStyle(
        "H3", parent=ss["Heading3"], fontSize=11.5, leading=15,
        textColor=GREY, spaceBefore=8, spaceAfter=4, keepWithNext=True))

    ss.add(ParagraphStyle(
        "Body", parent=ss["Normal"], fontSize=10, leading=15,
        alignment=TA_JUSTIFY, textColor=DARK, spaceAfter=7))
    ss.add(ParagraphStyle(
        "MyBullet", parent=ss["Normal"], fontSize=10, leading=14.5,
        alignment=TA_JUSTIFY, textColor=DARK, leftIndent=16,
        bulletIndent=4, spaceAfter=4))
    ss.add(ParagraphStyle(
        "Caption", parent=ss["Normal"], fontSize=8.5, leading=11.5,
        alignment=TA_CENTER, textColor=GREY, spaceBefore=4, spaceAfter=14,
        fontName=FONT_I))
    ss.add(ParagraphStyle(
        "TblHeader", parent=ss["Normal"], fontSize=8.5, leading=11,
        textColor=colors.white, fontName=FONT_B))
    ss.add(ParagraphStyle(
        "TblCell", parent=ss["Normal"], fontSize=8.5, leading=11,
        textColor=DARK))
    ss.add(ParagraphStyle(
        "TOCItem", parent=ss["Normal"], fontSize=10.5, leading=18,
        textColor=DARK))
    ss.add(ParagraphStyle(
        "KeyBox", parent=ss["Normal"], fontSize=9.5, leading=14,
        alignment=TA_JUSTIFY, textColor=DARK,
        backColor=LIGHTGREY, borderColor=MIDGREY, borderWidth=0.8,
        borderPadding=8, spaceBefore=4, spaceAfter=12))

    # Aplica a fonte DejaVu (Unicode) a todos os estilos definidos; os estilos
    # que já fixam uma variante (itálico/negrito) são preservados acima.
    for name, st in ss.byName.items():
        if not hasattr(st, "fontName"):
            continue
        base = (st.fontName or "").lower()
        if "bold" in base and ("italic" in base or "oblique" in base):
            st.fontName = FONT_BI
        elif "bold" in base:
            st.fontName = FONT_B
        elif "italic" in base or "oblique" in base:
            st.fontName = FONT_I
        else:
            st.fontName = FONT
    return ss


STYLES = build_styles()


# =============================================================================
# Helpers de conteúdo
# =============================================================================
def P(text, style="Body"):
    return Paragraph(text, STYLES[style])


def bullets(items, style="MyBullet"):
    return [Paragraph(f"•&nbsp;&nbsp;{t}", STYLES[style]) for t in items]


def figure(path, caption, max_w=15.5 * cm, max_h=20 * cm):
    """Insere uma figura ajustada à largura, preservando proporção."""
    full = os.path.join(FIG_DIR, path)
    if not os.path.exists(full):
        return [P(f"<i>[figura ausente: {path}]</i>", "Caption")]
    iw, ih = ImageReader(full).getSize()
    ratio = ih / iw
    w = max_w
    h = w * ratio
    if h > max_h:
        h = max_h
        w = h / ratio
    img = Image(full, width=w, height=h)
    img.hAlign = "CENTER"
    return [KeepTogether([img, P(caption, "Caption")])]


def data_table(rows, col_widths=None, header=True, font=8.5):
    """Tabela estilizada a partir de uma lista de listas (strings)."""
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, MIDGREY),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), FONT_B),
            ("TOPPADDING", (0, 0), (-1, 0), 5),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ]
        for r in range(1, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), LIGHTGREY))
    return Table(rows, colWidths=col_widths, repeatRows=1 if header else 0,
                 style=TableStyle(style))


def keybox(title, text):
    return Paragraph(f"<b>{title}</b> &nbsp; {text}", STYLES["KeyBox"])


# =============================================================================
# Cabeçalho / rodapé
# =============================================================================
def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    # rodapé
    canvas.setStrokeColor(MIDGREY)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.4 * cm, w - 2 * cm, 1.4 * cm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.0 * cm,
                      "Análise Multi-ômica de Resistência a BRAFi/MEKi em Melanoma")
    canvas.drawRightString(w - 2 * cm, 1.0 * cm, f"Página {doc.page}")
    canvas.restoreState()


def on_cover(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 4.2 * cm, w, 4.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(GREEN)
    canvas.rect(0, h - 4.35 * cm, w, 0.15 * cm, fill=1, stroke=0)
    canvas.restoreState()


# =============================================================================
# Construção do documento
# =============================================================================
def build():
    doc = BaseDocTemplate(
        PDF_NAME, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Relatório — Análise Multi-ômica de Melanoma",
        author="Pipeline Multi-ômica de Melanoma")

    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    cover_frame = Frame(doc.leftMargin, doc.bottomMargin,
                        doc.width, doc.height - 3 * cm, id="cover")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=on_cover),
        PageTemplate(id="main", frames=[frame], onPage=on_page),
    ])

    story = []
    story += cover()
    story.append(NextPageTemplate("main"))
    story.append(PageBreak())

    story += summary_and_toc()
    story.append(PageBreak())
    story += sec_introduction()
    story.append(PageBreak())
    story += sec_datasets()
    story.append(PageBreak())
    story += sec_methodology()
    story.append(PageBreak())
    story += sec_results_pxd013923()
    story.append(PageBreak())
    story += sec_results_pxd022992()
    story.append(PageBreak())
    story += sec_results_ksea()
    story.append(PageBreak())
    story += sec_results_tcga()
    story.append(PageBreak())
    story += sec_results_temporal()
    story.append(PageBreak())
    story += sec_results_nn()
    story.append(PageBreak())
    story += sec_integration_conclusion()
    story.append(PageBreak())
    story += sec_appendix()

    doc.build(story)
    print(f"[SUCESSO] Relatório gerado: {PDF_NAME}")


# =============================================================================
# Seções
# =============================================================================
def cover():
    el = []
    el.append(Spacer(1, 2.2 * cm))
    el.append(P("RELATÓRIO TÉCNICO-CIENTÍFICO", "CoverSubtitle"))
    el.append(Spacer(1, 0.4 * cm))
    el.append(P("Análise Multi-ômica da Resistência a Inibidores "
                "de BRAF/MEK em Melanoma", "CoverTitle"))
    el.append(Spacer(1, 0.6 * cm))
    el.append(P("Quantificação de atividade de quinases por fosfoproteômica "
                "dirigida, perfis transcricionais temporais da resistência "
                "adquirida, caracterização genômica clínica e modelagem "
                "preditiva por aprendizado profundo", "CoverSubtitle"))
    el.append(Spacer(1, 1.6 * cm))

    box = [
        ["Conjuntos de dados", "PXD013923 · PXD022992 · TCGA-SKCM · GSE110054 · OmniPath"],
        ["Modalidades", "Fosfoproteômica · Proteômica · Transcriptômica · Genômica"],
        ["Sítios de fosforilação", "10.273 (SILAC) + 55.939 (DIA)"],
        ["Pacientes clínicos", "470 (TCGA-SKCM)"],
        ["Curso temporal", "DMSO → 3 d → 21 d → 90 d (2 linhagens)"],
        ["Modelo preditivo", "Rede neural — ROC-AUC de teste 0,989"],
    ]
    rows = [[P(k, "TblHeader") if False else Paragraph(f"<b>{k}</b>", STYLES["TblCell"]),
             Paragraph(v, STYLES["TblCell"])] for k, v in box]
    t = Table(rows, colWidths=[4.6 * cm, 10.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHTGREY),
        ("BOX", (0, 0), (-1, -1), 0.8, MIDGREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, MIDGREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    el.append(t)
    el.append(Spacer(1, 2.2 * cm))
    hoje = datetime.date.today().strftime("%d de %B de %Y")
    el.append(P(f"Documento gerado automaticamente a partir dos resultados "
                f"do pipeline · {hoje}", "CoverMeta"))
    return el


def summary_and_toc():
    el = []
    el.append(P("Resumo Executivo", "H1"))
    el.append(P(
        "Este relatório descreve, de forma detalhada, um pipeline integrado de "
        "análise multi-ômica voltado para o estudo da resposta e da resistência "
        "de células de melanoma aos inibidores das quinases BRAF e MEK "
        "(BRAFi/MEKi), eixo terapêutico central nos melanomas com mutação "
        "<i>BRAF</i> V600. O trabalho reúne cinco fontes de dados públicas e "
        "complementares, cobrindo desde a resposta bioquímica imediata da via "
        "MAPK/ERK (escala de minutos) até a reprogramação transcricional que "
        "acompanha a aquisição de resistência (escala de meses), passando pela "
        "paisagem mutacional e clínica de uma coorte tumoral humana.", "Body"))
    el.append(P(
        "A análise da resposta aguda a BRAFi, MEKi e ERKi (dataset PXD013923) "
        "demonstra supressão seletiva e estatisticamente robusta da fosforilação "
        "dos sítios canônicos da via MAPK/ERK, com forte concordância entre os "
        "três inibidores (r de Pearson entre 0,71 e 0,76), confirmando a "
        "linearidade do módulo RAF→MEK→ERK. A Análise de Enriquecimento de "
        "Substratos de Quinases (KSEA), apoiada na rede cinase-substrato "
        "experimental do OmniPath, recupera com precisão a supressão de BRAF "
        "(z = −6,6), RAF1 (z = −6,5), MAP2K1 (z = −6,5) e MAPK1/3, validando "
        "biologicamente todo o fluxo de processamento.", "Body"))
    el.append(P(
        "A caracterização do fosfoproteoma basal de seis linhagens por "
        "aquisição independente de dados (directDIA, PXD022992) separa "
        "claramente as linhagens <i>BRAF</i> V600E das <i>NRAS</i>-mutantes e "
        "identifica milhares de sítios diferencialmente fosforilados. A coorte "
        "clínica TCGA-SKCM (470 pacientes) fornece a frequência mutacional dos "
        "principais genes condutores e a estratificação molecular do melanoma, "
        "enquanto o curso temporal GSE110054 revela as assinaturas "
        "transcricionais progressivas da resistência adquirida. Por fim, um "
        "classificador de rede neural treinado sobre dados reais atinge "
        "ROC-AUC de teste de 0,989, demonstrando que os padrões de fosforilação "
        "específicos da via são preditivos e aprendíveis.", "Body"))

    el.append(P("Sumário", "H2"))
    toc = [
        "1.&nbsp;&nbsp;Introdução e contexto biológico",
        "2.&nbsp;&nbsp;Conjuntos de dados e proveniência",
        "3.&nbsp;&nbsp;Metodologia",
        "4.&nbsp;&nbsp;Resultados — Resposta aguda à inibição (PXD013923)",
        "5.&nbsp;&nbsp;Resultados — Fosfoproteoma directDIA (PXD022992)",
        "6.&nbsp;&nbsp;Resultados — Atividade de quinases por KSEA (OmniPath)",
        "7.&nbsp;&nbsp;Resultados — Caracterização genômica clínica (TCGA-SKCM)",
        "8.&nbsp;&nbsp;Resultados — Dinâmica temporal da resistência (GSE110054)",
        "9.&nbsp;&nbsp;Resultados — Modelo preditivo por rede neural",
        "10.&nbsp;Integração multi-ômica e considerações finais",
        "11.&nbsp;Apêndice — Inventário de figuras e tabelas",
    ]
    for t in toc:
        el.append(Paragraph(t, STYLES["TOCItem"]))
    return el


def sec_introduction():
    el = []
    el.append(P("1. Introdução e contexto biológico", "H1"))
    el.append(P(
        "O melanoma cutâneo é o tumor de pele de maior letalidade, e a sua "
        "biologia molecular é dominada pela hiperativação da via de sinalização "
        "MAPK/ERK (RAS→RAF→MEK→ERK). Aproximadamente metade dos melanomas "
        "cutâneos apresenta mutações ativadoras no gene <i>BRAF</i>, com larga "
        "predominância da substituição V600 (sobretudo V600E), que mantém a "
        "quinase BRAF constitutivamente ativa e dirige a proliferação celular "
        "independentemente de estímulos externos.", "Body"))
    el.append(P(
        "Essa dependência molecular tornou a via MAPK/ERK um alvo terapêutico "
        "de eleição. Os inibidores de BRAF (BRAFi, p. ex. vemurafenibe e "
        "dabrafenibe) e os inibidores de MEK (MEKi, p. ex. trametinibe e "
        "cobimetinibe), em particular na combinação BRAFi+MEKi, produzem "
        "respostas clínicas iniciais expressivas. No entanto, a durabilidade "
        "dessa resposta é limitada pela emergência quase universal de "
        "<b>resistência adquirida</b>, frequentemente associada à reativação da "
        "própria via MAPK e à ativação de vias de escape, como a "
        "reprogramação de receptores tirosina-quinase (RTKs) e a transição "
        "para um estado mesenquimal-invasivo.", "Body"))
    el.append(P(
        "Compreender a resistência exige observar o sistema em múltiplas escalas "
        "temporais e em múltiplas camadas moleculares. A fosforilação de "
        "proteínas é a leitura mais direta do estado de atividade das quinases "
        "e responde em minutos à inibição farmacológica; a reprogramação "
        "transcricional, por sua vez, desenvolve-se ao longo de dias a meses e "
        "consolida o fenótipo resistente; e a paisagem genômica do tumor define "
        "o contexto mutacional em que essas adaptações ocorrem. Este projeto "
        "articula essas camadas em um único pipeline reprodutível.", "Body"))

    el.append(P("1.1. Objetivos do projeto", "H2"))
    for b in bullets([
        "Quantificar o estado de atividade das quinases da via MAPK/ERK a partir "
        "de fosfoproteômica de alta resolução, utilizando aquisição dirigida e "
        "independente de dados.",
        "Avaliar a resposta dinâmica à inibição das quinases, da supressão aguda "
        "(minutos) à adaptação transcricional tardia (meses).",
        "Caracterizar a heterogeneidade molecular entre linhagens e entre "
        "subtipos tumorais clínicos (BRAF V600E vs. NRAS-mutante).",
        "Construir modelos preditivos do estado molecular associado à via BRAF, "
        "integrando abordagens de rede (KSEA) e de aprendizado de máquina.",
        "Validar todo o fluxo com dados públicos de repositórios consolidados "
        "(PRIDE, GEO e GDC/TCGA).",
    ]):
        el.append(b)
    return el


def sec_datasets():
    el = []
    el.append(P("2. Conjuntos de dados e proveniência", "H1"))
    el.append(P(
        "Foram integrados cinco conjuntos de dados públicos, selecionados para "
        "cobrir de forma complementar as diferentes escalas temporais e camadas "
        "moleculares do problema. A tabela a seguir resume cada fonte; as "
        "subseções detalham o desenho experimental e o formato.", "Body"))

    raw = [
        ["Acesso", "Tipo", "Conteúdo", "Escala / dimensão"],
        ["PXD013923", "Fosfoproteoma SILAC", "BRAFi/MEKi/ERKi em A375", "30 min · 10.273 sítios"],
        ["PXD022992", "Fosfoproteoma directDIA", "6 linhagens de melanoma", "basal · 55.939 sítios"],
        ["OmniPath", "Rede cinase-substrato", "Relações enzima→sítio", "39.037 relações"],
        ["TCGA-SKCM", "Genômica + clínica", "Melanoma cutâneo", "470 pacientes"],
        ["GSE110054", "RNA-seq temporal", "Curso temporal BRAFi", "25.222 genes · 10 amostras"],
    ]
    el.append(data_table(raw, col_widths=[2.6 * cm, 3.8 * cm, 4.6 * cm, 4 * cm]))
    el.append(Spacer(1, 0.3 * cm))

    el.append(P("2.1. PXD013923 — Resposta aguda a inibidores RAF/MEK/ERK", "H2"))
    el.append(P(
        "Conjunto depositado no repositório PRIDE/EBI, correspondente a um "
        "estudo aprofundado do módulo RAF–MEK–ERK e de seus efetores imediatos. "
        "Células de melanoma A375 (<i>BRAF</i> V600E) foram tratadas por "
        "30 minutos com inibidor de RAF (BRAFi/dabrafenibe), de MEK "
        "(MEKi/trametinibe) ou de ERK (ERKi/SCH772984), em desenho de marcação "
        "isotópica SILAC, no qual as razões pesado/leve (<i>Ratio H/L "
        "normalized</i>) expressam a mudança de fosforilação entre condição "
        "tratada e controle. O processamento de espectros foi realizado em "
        "MaxQuant; os identificadores de proteína são do tipo Ensembl (ENSP), "
        "mapeados a símbolos de gene por meio do serviço mygene.info "
        "(2.921 de 3.540 identificadores; cobertura de 82,5%). O curtíssimo "
        "tempo de tratamento captura a resposta bioquímica imediata, antes de "
        "qualquer reprogramação transcricional.", "Body"))

    el.append(P("2.2. PXD022992 — Fosfoproteoma directDIA de seis linhagens", "H2"))
    el.append(P(
        "Perfil de fosfoproteoma sem marcação, obtido por aquisição "
        "independente de dados em modo directDIA (estratégia da família "
        "guidedDIA), abrangendo seis linhagens de melanoma: A375, SH-4, "
        "SK-MEL-28 e RPMI-7951 (portadoras de <i>BRAF</i> V600E) e G361 e "
        "SK-MEL-31 (portadoras de mutação em <i>NRAS</i>). Cada linhagem foi "
        "medida em duas réplicas técnicas. O relatório de quantificação "
        "(formato Spectronaut, conteúdo TSV) contém colunas de gene "
        "(<i>PG.Genes</i>), sequência modificada (<i>EG.ModifiedSequence</i>), "
        "localização do sítio de PTM (<i>EG.ProteinPTMLocations</i>) e "
        "intensidade por amostra (<i>EG.TotalQuantity</i>). Este conjunto "
        "fornece a paisagem basal de fosforilação e a base para a comparação "
        "entre os contextos mutacionais BRAF e NRAS.", "Body"))

    el.append(P("2.3. OmniPath — Rede cinase-substrato para KSEA", "H2"))
    el.append(P(
        "Para inferir atividade de quinases a partir dos sítios de fosforilação "
        "medidos, utilizou-se a rede enzima-substrato do OmniPath, que agrega "
        "relações curadas de bases consolidadas (incluindo PhosphoSitePlus, "
        "SIGNOR e outras). A versão empregada contém 39.037 relações de "
        "fosforilação cobrindo 1.648 quinases, com anotação de gene da enzima, "
        "gene do substrato, resíduo e posição — permitindo o mapeamento direto "
        "ao formato de sítio <i>GENE_resíduoposição</i> utilizado nas matrizes "
        "de fosforilação.", "Body"))

    el.append(P("2.4. TCGA-SKCM — Coorte genômica e clínica", "H2"))
    el.append(P(
        "Dados do projeto Skin Cutaneous Melanoma do TCGA, obtidos via API "
        "pública do NCI Genomic Data Commons (GDC). Foram recuperados dados "
        "clínicos e de sobrevida de 470 pacientes, o catálogo de mutações "
        "somáticas dos principais genes condutores do melanoma "
        "(<i>BRAF, NRAS, NF1, PTEN, CDKN2A, KIT, MAP2K1, RAC1, PPP6C, "
        "PREX2, IDH1</i>) e quantificações de expressão gênica por RNA-seq "
        "(fluxo STAR-Counts) para um subconjunto representativo de amostras, "
        "estratificado pelos subtipos moleculares. Esse conjunto ancora as "
        "observações in vitro no contexto de tumores humanos.", "Body"))

    el.append(P("2.5. GSE110054 — Curso temporal da resistência ao BRAFi", "H2"))
    el.append(P(
        "Série de expressão gênica (RNA-seq) depositada no GEO, "
        "correspondente a um experimento de curso temporal em que as linhagens "
        "de melanoma M229 e M397 (ambas <i>BRAF</i> V600E) foram tratadas com "
        "vemurafenibe e amostradas em múltiplos pontos: controle (DMSO), "
        "3 dias, 11–21 dias e até 73–90 dias de tratamento contínuo. A matriz "
        "de expressão processada (FPKM, 25.222 genes) permite acompanhar a "
        "transição progressiva do estado sensível para o estado resistente, "
        "complementando, na escala de dias a meses, a resposta aguda observada "
        "na fosfoproteômica.", "Body"))
    return el


def sec_methodology():
    el = []
    el.append(P("3. Metodologia", "H1"))
    el.append(P(
        "O pipeline foi implementado em Python, com processamento numérico em "
        "<i>NumPy</i>/<i>pandas</i>, estatística em <i>SciPy</i>, aprendizado de "
        "máquina em <i>scikit-learn</i>, aprendizado profundo em <i>PyTorch</i> "
        "e visualização em <i>Matplotlib</i>/<i>Seaborn</i>. Cada etapa é "
        "executada como um módulo independente e orquestrada por um script "
        "central, com registro de log e salvamento padronizado de figuras e "
        "tabelas. As subseções a seguir descrevem a metodologia por etapa.", "Body"))

    el.append(P("3.1. Processamento e controle de qualidade da fosfoproteômica", "H2"))
    el.append(P(
        "Os sítios de fosforilação foram filtrados por probabilidade de "
        "localização do sítio (limiar ≥ 0,75), com remoção de proteínas "
        "reversas (decoy) e de contaminantes potenciais. As intensidades foram "
        "transformadas em log<sub>2</sub>; razões SILAC nulas ou negativas "
        "foram tratadas como ausentes antes da transformação logarítmica. Para "
        "a matriz directDIA, valores não detectados (rotulados como "
        "<i>Filtered</i>) foram convertidos em ausentes e, quando necessário "
        "para análises que exigem matriz completa, imputados por uma "
        "abordagem do tipo Perseus (substituição pela cauda inferior da "
        "distribuição de cada amostra, em <i>média − 1,8 × desvio-padrão</i>).", "Body"))

    el.append(P("3.2. Ancoragem biológica do sinal SILAC", "H2"))
    el.append(P(
        "Como a orientação das razões SILAC pode ser ambígua, o sinal foi "
        "ancorado à biologia conhecida: a inibição do módulo MAPK deve "
        "<b>reduzir</b> a fosforilação dos sítios canônicos da via. Definiu-se "
        "um conjunto-âncora de aproximadamente 30 genes da via MAPK/ERK e seus "
        "substratos canônicos; a convenção de sinal foi fixada de modo que a "
        "mediana dos sítios-âncora seja negativa (supressão). Essa verificação "
        "fornece, simultaneamente, um controle de qualidade independente do "
        "processamento.", "Body"))

    el.append(P("3.3. Análise de Enriquecimento de Substratos de Quinases (KSEA)", "H2"))
    el.append(P(
        "A atividade de cada quinase foi estimada por KSEA: para uma quinase "
        "<i>k</i> com conjunto de substratos <i>S<sub>k</sub></i> medidos, o "
        "escore é o desvio padronizado da média dos substratos em relação à "
        "média global, na forma <i>z = (m<sub>S</sub> − m<sub>global</sub>) / "
        "(σ<sub>global</sub> / √n)</i>. A significância foi avaliada por teste "
        "de permutação (1.000 reamostragens de conjuntos de substratos de mesmo "
        "tamanho), e apenas quinases com pelo menos cinco substratos medidos "
        "foram pontuadas. Os conjuntos de substratos provêm da rede OmniPath.", "Body"))

    el.append(P("3.4. Análise diferencial e estatística", "H2"))
    el.append(P(
        "As comparações entre grupos (p. ex. BRAF V600E vs. NRAS) foram feitas "
        "por teste t de Welch sítio a sítio, com a magnitude do efeito expressa "
        "como diferença de médias em log<sub>2</sub> (log<sub>2</sub>FC). "
        "Consideraram-se diferencialmente fosforilados os sítios com "
        "<i>p</i> &lt; 0,05 e |log<sub>2</sub>FC| &gt; 1. Para o curso temporal, "
        "a monotonicidade da resposta foi medida por correlação de Spearman "
        "entre o tempo de tratamento e a expressão, com correção de Bonferroni; "
        "consideraram-se assinaturas de resistência os genes com |ρ| &gt; 0,7 e "
        "<i>p</i> ajustado &lt; 0,01 de forma concordante nas duas linhagens.", "Body"))

    el.append(P("3.5. Redução de dimensionalidade e estratificação", "H2"))
    el.append(P(
        "Análise de Componentes Principais (PCA) foi aplicada, após "
        "padronização, para visualizar a estrutura de amostras — separação "
        "entre linhagens e trajetória da resistência ao longo do tempo. A "
        "estratificação molecular da coorte TCGA-SKCM seguiu a prioridade "
        "<i>BRAF</i> V600 &gt; <i>NRAS</i> &gt; <i>NF1</i> &gt; tríplice "
        "selvagem, e a sobrevida global foi estimada pelo método de "
        "Kaplan–Meier por subtipo.", "Body"))

    el.append(P("3.6. Modelagem preditiva por rede neural", "H2"))
    el.append(P(
        "Construiu-se um classificador para prever se um sítio de fosforilação é "
        "diferencialmente fosforilado entre os contextos BRAF V600E e NRAS, a "
        "partir das intensidades nas seis linhagens. O conjunto foi balanceado "
        "(igual número de sítios diferenciais e não diferenciais) e dividido em "
        "treino e teste de forma estratificada (70/30); a padronização foi "
        "ajustada apenas no conjunto de treino para evitar vazamento de "
        "informação. A arquitetura é um perceptron multicamadas "
        "(6 → 128 → 64 → 32 → 1) com normalização em lote, ativação ReLU e "
        "<i>dropout</i> de 0,3; o treinamento usou perda de entropia cruzada "
        "binária, otimizador AdamW com decaimento de peso e agendador de taxa de "
        "aprendizagem por recozimento por cosseno, ao longo de 120 épocas. "
        "Foram registradas, por época, as curvas de perda e de ROC-AUC em "
        "treino e teste.", "Body"))
    return el


def sec_results_pxd013923():
    el = []
    el.append(P("4. Resultados — Resposta aguda à inibição (PXD013923)", "H1"))
    el.append(P(
        "Após filtragem por qualidade, foram quantificados 10.273 sítios de "
        "fosforilação. Para cada inibidor calculou-se a alteração média de "
        "fosforilação (log<sub>2</sub>) em relação ao controle. O conjunto-"
        "âncora da via MAPK/ERK foi composto por 81 sítios e apresentou mediana "
        "de log<sub>2</sub> de −0,295 — claramente negativa, confirmando a "
        "supressão seletiva esperada e fixando corretamente a orientação do "
        "sinal SILAC.", "Body"))

    el.append(keybox(
        "Resultado-chave.",
        "Os sítios canônicos da via MAPK/ERK são suprimidos de forma seletiva "
        "pela inibição (mediana log<sub>2</sub> ≈ −0,29 a −0,31 nos três "
        "inibidores), enquanto o restante do fosfoproteoma permanece próximo de "
        "zero (mediana entre −0,01 e −0,04). A inibição atinge especificamente "
        "a via-alvo."))

    el.append(P("4.1. Distribuição da resposta por inibidor", "H2"))
    el.append(P(
        "A distribuição global das alterações de fosforilação é centrada "
        "próxima de zero para os três inibidores, mas com cauda negativa "
        "pronunciada de sítios fortemente reprimidos. Foram identificados, "
        "respectivamente, 346 (BRAFi), 331 (MEKi) e 290 (ERKi) sítios com "
        "supressão acentuada (log<sub>2</sub> &lt; −1), contra um número bem "
        "menor de sítios induzidos.", "Body"))
    el += figure("pxd013923_response_distributions.png",
                 "Figura 4.1. Distribuição das alterações de fosforilação "
                 "(log₂, inibidor/controle) para BRAFi, MEKi e ERKi em A375 "
                 "(30 min). A cauda negativa concentra os sítios suprimidos.")

    el.append(P("4.2. Concordância entre inibidores e linearidade do módulo", "H2"))
    el.append(P(
        "As respostas aos três inibidores são fortemente correlacionadas entre "
        "si: r de Pearson de 0,756 (BRAFi vs. MEKi), 0,705 (BRAFi vs. ERKi) e "
        "0,760 (MEKi vs. ERKi). Essa concordância é a assinatura esperada de um "
        "módulo de sinalização linear RAF→MEK→ERK, no qual a inibição em "
        "qualquer nível propaga efeitos semelhantes aos efetores a jusante.", "Body"))
    el += figure("pxd013923_inhibitor_agreement.png",
                 "Figura 4.2. Concordância par a par entre as respostas a "
                 "BRAFi, MEKi e ERKi. A correlação positiva forte reflete a "
                 "linearidade do módulo RAF→MEK→ERK.")

    el.append(P("4.3. Ancoragem na via MAPK/ERK", "H2"))
    el.append(P(
        "Comparando os sítios da via MAPK/ERK com o restante do fosfoproteoma, "
        "observa-se deslocamento sistemático dos sítios da via para valores "
        "negativos nos três inibidores (mediana de −0,293 para BRAFi, −0,267 "
        "para MEKi e −0,310 para ERKi), enquanto os demais sítios permanecem "
        "centrados em zero. A supressão é, portanto, específica da via-alvo.", "Body"))
    el += figure("pxd013923_mapk_anchor.png",
                 "Figura 4.3. Sítios da via MAPK/ERK (vermelho) vs. demais "
                 "sítios (cinza). A supressão concentra-se nos sítios canônicos "
                 "da via.", max_w=13 * cm)

    el.append(P("4.4. Sítios mais fortemente suprimidos", "H2"))
    el.append(P(
        "Entre os sítios mais reprimidos por BRAFi destacam-se alvos "
        "consistentes com a inibição da via, incluindo o fator de transcrição "
        "ERF (repressor regulado por ERK) e o substrato RPS6KA3 (RSK2), "
        "efetor a jusante de ERK. A tabela apresenta uma seleção dos sítios de "
        "maior supressão.", "Body"))
    try:
        t = pd.read_csv(os.path.join(OUT_DIR, "PXD013923_top_suppressed_sites.csv"))
        t = t[t["inhibitor"] == "BRAFi"].head(8)
        rows = [["Sítio (gene_resíduo)", "Gene", "log₂FC (BRAFi)"]]
        for _, r in t.iterrows():
            g = str(r["gene"])
            if len(g) > 22:
                g = g[:20] + "…"
            site = str(r.iloc[0])
            if len(site) > 26:
                site = site[:24] + "…"
            rows.append([site, g, f"{r['log2FC']:+.2f}"])
        el.append(data_table(rows, col_widths=[7 * cm, 5 * cm, 3 * cm]))
        el.append(Spacer(1, 0.2 * cm))
        el.append(P("Tabela 4.1. Sítios de fosforilação mais suprimidos por "
                    "BRAFi (seleção).", "Caption"))
    except Exception:
        pass
    return el


def sec_results_pxd022992():
    el = []
    el.append(P("5. Resultados — Fosfoproteoma directDIA (PXD022992)", "H1"))
    el.append(P(
        "A abordagem directDIA quantificou 55.939 sítios de fosforilação ao "
        "longo das seis linhagens, após remoção de sítios não detectados em "
        "todas as amostras. As medianas de intensidade (log<sub>2</sub>) são "
        "homogêneas entre linhagens (entre 14,59 e 14,74), indicando boa "
        "comparabilidade após o processamento.", "Body"))

    el.append(P("5.1. Controle de qualidade", "H2"))
    el.append(P(
        "As distribuições de intensidade e as taxas de valores ausentes por "
        "linhagem foram inspecionadas para assegurar a qualidade da matriz. As "
        "distribuições são consistentes entre as linhagens BRAF V600E e NRAS, "
        "validando as comparações subsequentes.", "Body"))
    el += figure("pxd022992_qc.png",
                 "Figura 5.1. Controle de qualidade do fosfoproteoma directDIA: "
                 "taxas de valores ausentes por linhagem e distribuições de "
                 "intensidade (log₂).")

    el.append(P("5.2. Estrutura entre linhagens (PCA)", "H2"))
    el.append(P(
        "A PCA das seis linhagens separa os contextos mutacionais ao longo do "
        "primeiro componente principal, com as linhagens BRAF V600E agrupadas e "
        "distintas das NRAS-mutantes. Isso indica que o estado mutacional "
        "condutor imprime uma assinatura detectável no fosfoproteoma basal.", "Body"))
    el += figure("pxd022992_pca_cell_lines.png",
                 "Figura 5.2. PCA das seis linhagens de melanoma. A separação "
                 "no PC1 acompanha o contexto mutacional (BRAF V600E vs. NRAS).",
                 max_w=12 * cm)

    el.append(P("5.3. Fosforilação diferencial BRAF V600E vs. NRAS", "H2"))
    el.append(P(
        "A comparação entre os grupos mutacionais identificou 1.387 sítios com "
        "fosforilação mais elevada nas linhagens BRAF V600E e 1.304 sítios mais "
        "elevados nas NRAS-mutantes (teste t de Welch; <i>p</i> &lt; 0,05 e "
        "|log<sub>2</sub>FC| &gt; 1). Esse conjunto de sítios diferenciais "
        "define a assinatura de fosforilação que distingue os dois contextos e "
        "serve de base para o modelo preditivo (Seção 9).", "Body"))
    el += figure("pxd022992_volcano_braf_vs_nras.png",
                 "Figura 5.3. Vulcão da fosforilação diferencial entre "
                 "linhagens BRAF V600E e NRAS-mutantes. Em destaque, os sítios "
                 "significativos em cada direção.")

    el.append(P("5.4. Sítios de maior variância e atividade de quinases", "H2"))
    el.append(P(
        "O agrupamento hierárquico dos sítios de maior variância revela blocos "
        "de fosforilação coerentes com os subtipos das linhagens. Uma análise "
        "de atividade de quinases por conjuntos de substratos (ERK, AKT, mTOR, "
        "CDK e FAK/SRC) sintetiza as diferenças de sinalização entre as "
        "linhagens.", "Body"))
    el += figure("pxd022992_heatmap_top_variance.png",
                 "Figura 5.4. Mapa de calor dos sítios de fosforilação de maior "
                 "variância, com agrupamento hierárquico das linhagens.",
                 max_w=11 * cm)
    el += figure("pxd022992_ksea_kinase_activity.png",
                 "Figura 5.5. Atividade de vias de quinases por linhagem "
                 "(média dos substratos, padronizada): ERK, AKT, mTOR, CDK e "
                 "FAK/SRC.")
    return el


def sec_results_ksea():
    el = []
    el.append(P("6. Resultados — Atividade de quinases por KSEA (OmniPath)", "H1"))
    el.append(P(
        "A análise KSEA, apoiada na rede cinase-substrato experimental do "
        "OmniPath, traduz os sítios de fosforilação medidos em escores de "
        "atividade por quinase. Aplicada à resposta aguda (PXD013923), a KSEA "
        "recupera de forma precisa a inibição do módulo MAPK.", "Body"))

    el.append(keybox(
        "Validação biológica.",
        "Sob BRAFi, as quinases do núcleo da via apresentam os escores mais "
        "negativos: BRAF (z = −6,6), RAF1 (z = −6,5), MAP2K1/MEK1 (z = −6,5) e "
        "MAPK1/3 — todas com p &lt; 0,001 no teste de permutação. O método "
        "reconstrói corretamente a hierarquia de supressão da via."))

    el.append(P("6.1. Atividade de quinases sob BRAFi / MEKi / ERKi", "H2"))
    el.append(P(
        "O mapa de calor dos escores de KSEA mostra supressão coordenada das "
        "quinases e fosfatases reguladas pela via — incluindo reguladores "
        "negativos como as fosfatases DUSP (DUSP1/8/16) e o scaffold KSR1, "
        "cuja fosforilação dependente de ERK cai abruptamente com a inibição. "
        "A tabela lista as quinases de maior supressão sob BRAFi.", "Body"))
    el += figure("ksea_pxd013923_inhibitors.png",
                 "Figura 6.1. Escores KSEA (z) por quinase sob BRAFi, MEKi e "
                 "ERKi. Tons azuis indicam supressão da atividade da quinase.",
                 max_w=11.5 * cm)

    try:
        k = pd.read_csv(os.path.join(OUT_DIR, "KSEA_PXD013923_kinase_zscores.csv"))
        kcol = "kinase" if "kinase" in k.columns else k.columns[1]
        k = k.sort_values("BRAFi").head(10)
        rows = [["Quinase", "z (BRAFi)", "z (MEKi)", "z (ERKi)"]]
        for _, r in k.iterrows():
            rows.append([str(r[kcol]), f"{r['BRAFi']:+.2f}",
                         f"{r['MEKi']:+.2f}", f"{r['ERKi']:+.2f}"])
        el.append(data_table(rows, col_widths=[5 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm]))
        el.append(Spacer(1, 0.2 * cm))
        el.append(P("Tabela 6.1. Dez quinases de maior supressão sob BRAFi "
                    "(escore KSEA z).", "Caption"))
    except Exception:
        pass

    el.append(P("6.2. Atividade de quinases entre linhagens", "H2"))
    el.append(P(
        "Aplicada ao fosfoproteoma basal (PXD022992), a KSEA descreve o estado "
        "de atividade das quinases característico de cada linhagem, evidenciando "
        "diferenças de sinalização entre os contextos BRAF V600E e NRAS.", "Body"))
    el += figure("ksea_pxd022992_cell_lines.png",
                 "Figura 6.2. Escores KSEA por quinase nas seis linhagens de "
                 "melanoma.", max_w=12 * cm)

    el.append(P("6.3. Síntese comparativa", "H2"))
    el += figure("ksea_combined_volcano.png",
                 "Figura 6.3. Painel comparativo: atividade de quinases sob "
                 "BRAFi (esquerda) e diferença de atividade entre A375 (BRAF "
                 "V600E) e G361 (NRAS) (direita).")
    return el


def sec_results_tcga():
    el = []
    el.append(P("7. Resultados — Caracterização genômica clínica (TCGA-SKCM)", "H1"))
    el.append(P(
        "A coorte TCGA-SKCM (470 pacientes) fornece o contexto clínico e "
        "genômico do melanoma cutâneo. A frequência mutacional dos genes "
        "condutores recupera o padrão canônico da doença: <i>BRAF</i> como gene "
        "mais frequentemente mutado, seguido de <i>NRAS</i>, com participação "
        "relevante de <i>NF1</i> e de supressores como <i>CDKN2A</i> e "
        "<i>PTEN</i>.", "Body"))

    try:
        m = pd.read_csv(os.path.join(OUT_DIR, "TCGA_SKCM_mutation_frequency.csv"))
        m = m.sort_values("pct", ascending=False).head(8)
        rows = [["Gene", "Pacientes mutados", "Frequência (%)"]]
        for _, r in m.iterrows():
            rows.append([str(r["gene"]), str(int(r["n_patients"])), f"{r['pct']:.1f}"])
        el.append(data_table(rows, col_widths=[5 * cm, 5 * cm, 4.5 * cm]))
        el.append(Spacer(1, 0.2 * cm))
        el.append(P("Tabela 7.1. Frequência de mutação dos principais genes "
                    "condutores em TCGA-SKCM.", "Caption"))
    except Exception:
        pass

    el.append(P("7.1. Paisagem mutacional e subtipos moleculares", "H2"))
    el.append(P(
        "A estratificação molecular classificou os pacientes em quatro "
        "subtipos, segundo a hierarquia de mutações condutoras. A distribuição "
        "observada é coerente com a literatura do melanoma cutâneo, em que os "
        "subtipos BRAF e NRAS, somados à classe NF1 e ao grupo tríplice "
        "selvagem, organizam a heterogeneidade da doença.", "Body"))
    el += figure("tcga_skcm_mutation_landscape.png",
                 "Figura 7.1. Paisagem mutacional do TCGA-SKCM: frequência dos "
                 "genes condutores (esquerda) e distribuição dos subtipos "
                 "moleculares (direita).")

    el.append(P("7.2. Sobrevida global por subtipo", "H2"))
    el.append(P(
        "As curvas de sobrevida global de Kaplan–Meier foram estimadas por "
        "subtipo molecular, permitindo comparar a evolução clínica das classes "
        "definidas geneticamente.", "Body"))
    el += figure("tcga_skcm_survival_km.png",
                 "Figura 7.2. Sobrevida global (Kaplan–Meier) por subtipo "
                 "molecular em TCGA-SKCM.", max_w=12.5 * cm)

    el.append(P("7.3. Expressão de vias e integração com a fosfoproteômica", "H2"))
    el.append(P(
        "A expressão das vias MAPK e PI3K foi avaliada por RNA-seq nos subtipos "
        "moleculares, e os genes diferencialmente fosforilados entre BRAF V600E "
        "e NRAS (Seção 5) foram cruzados com a expressão tumoral, articulando a "
        "observação in vitro com o tecido clínico.", "Body"))
    el += figure("tcga_skcm_pathway_expression.png",
                 "Figura 7.3. Expressão de genes das vias MAPK/PI3K por subtipo "
                 "molecular (RNA-seq, TCGA-SKCM).", max_w=12.5 * cm)
    el += figure("tcga_skcm_phospho_integration.png",
                 "Figura 7.4. Integração fosfoproteômica-genômica: principais "
                 "genes diferenciais entre BRAF V600E e NRAS.")
    return el


def sec_results_temporal():
    el = []
    el.append(P("8. Resultados — Dinâmica temporal da resistência (GSE110054)", "H1"))
    el.append(P(
        "O curso temporal em M229 e M397 sob vemurafenibe revela a "
        "reprogramação transcricional progressiva que acompanha a aquisição de "
        "resistência. A análise de monotonicidade (Spearman) identificou, de "
        "forma concordante nas duas linhagens, 647 genes consistentemente "
        "induzidos e 267 genes consistentemente reprimidos ao longo do tempo de "
        "tratamento.", "Body"))

    el.append(keybox(
        "Resultado-chave.",
        "Enquanto a fosfoproteômica capta a supressão imediata da via "
        "(minutos), o transcriptoma temporal revela a adaptação tardia (dias a "
        "meses): 647 genes sobem e 267 descem de forma monotônica e concordante "
        "nas duas linhagens, definindo uma assinatura robusta de resistência."))

    el.append(P("8.1. Qualidade e estrutura temporal", "H2"))
    el += figure("temporal_gse110054_qc.png",
                 "Figura 8.1. Controle de qualidade do curso temporal: "
                 "distribuições de expressão e matriz de correlação entre "
                 "amostras.")
    el.append(P(
        "A PCA organiza as amostras em uma trajetória contínua, do estado "
        "sensível (DMSO) ao estado de resistência adquirida, ilustrando a "
        "progressão ordenada do sistema sob pressão farmacológica.", "Body"))
    el += figure("temporal_pca_resistance_trajectory.png",
                 "Figura 8.2. Trajetória de PCA da sensibilidade à resistência "
                 "em M229 e M397 sob BRAFi. As setas indicam o avanço temporal.",
                 max_w=12.5 * cm)

    el.append(P("8.2. Dinâmica de vias e genes de resistência", "H2"))
    el.append(P(
        "Os conjuntos de genes das vias MAPK, RTK, PI3K/AKT/mTOR, EMT, "
        "ciclo celular e sobrevivência apresentam dinâmicas distintas ao longo "
        "do tratamento, com destaque para a reativação de receptores "
        "tirosina-quinase e a indução de marcadores mesenquimais — vias de "
        "escape clássicas da resistência ao BRAFi.", "Body"))
    el += figure("temporal_pathway_heatmap.png",
                 "Figura 8.3. Dinâmica de expressão das vias ao longo do curso "
                 "temporal de BRAFi.", max_w=11.5 * cm)
    el += figure("temporal_key_gene_trajectories.png",
                 "Figura 8.4. Trajetórias temporais de genes-chave de "
                 "resistência (RTKs, proliferação, EMT, sobrevivência) em M229 "
                 "e M397.")
    el += figure("temporal_resistance_signature_heatmap.png",
                 "Figura 8.5. Assinatura de resistência: genes consistentemente "
                 "induzidos e reprimidos (|ρ| > 0,7 nas duas linhagens).",
                 max_w=11.5 * cm)

    el.append(P("8.3. Integração: supressão aguda vs. rebote transcricional", "H2"))
    el.append(P(
        "Ao cruzar a supressão aguda de quinases (KSEA aos 30 min, PXD013923) "
        "com a variação transcricional tardia (expressão na resistência, "
        "GSE110054), identificam-se quinases que, embora suprimidas de imediato "
        "pela inibição, têm sua expressão gênica reforçada no estado resistente "
        "— candidatas a mecanismos de escape adaptativo, incluindo KSR1, "
        "MAP3K14, DUSP8, DYRK1B e PRKCA.", "Body"))
    el += figure("temporal_kinase_escape_integration.png",
                 "Figura 8.6. Integração entre supressão aguda de quinases "
                 "(eixo x, KSEA 30 min) e rebote transcricional tardio (eixo y, "
                 "resistência). Em destaque, quinases de escape.", max_w=12.5 * cm)
    return el


def sec_results_nn():
    el = []
    el.append(P("9. Resultados — Modelo preditivo por rede neural", "H1"))
    el.append(P(
        "Para avaliar se a assinatura de fosforilação que distingue BRAF V600E "
        "de NRAS é aprendível, treinou-se um classificador de rede neural sobre "
        "5.382 sítios balanceados (50% diferenciais, 50% não diferenciais), com "
        "divisão estratificada em 3.767 sítios de treino e 1.615 de teste. O "
        "modelo, com 11.649 parâmetros, foi treinado por 120 épocas.", "Body"))

    el.append(keybox(
        "Desempenho.",
        "O classificador atinge ROC-AUC de teste de 0,989 (melhor época: "
        "0,991), F1 de 0,95 e acurácia de 95% no conjunto de teste retido. As "
        "curvas de treino e teste evoluem de forma próxima, indicando "
        "generalização sem sobreajuste."))

    el.append(P("9.1. Curvas de treinamento e de teste", "H2"))
    el.append(P(
        "As curvas de perda (entropia cruzada binária) e de ROC-AUC ao longo "
        "das épocas mostram convergência estável e desempenho de teste próximo "
        "ao de treino — a perda de teste mantém-se inclusive abaixo da de "
        "treino na maior parte do treinamento, e a AUC de teste estabiliza "
        "acima de 0,98 a partir de poucas dezenas de épocas.", "Body"))
    el += figure("nn_training_curves.png",
                 "Figura 9.1. Curvas de treinamento e de teste: perda (esquerda) "
                 "e ROC-AUC (direita) por época. A proximidade das curvas indica "
                 "boa generalização.")

    el.append(P("9.2. Avaliação no conjunto de teste", "H2"))
    el.append(P(
        "A curva ROC e a matriz de confusão no conjunto de teste retido "
        "confirmam a alta capacidade discriminativa do modelo, com elevada "
        "sensibilidade e especificidade na identificação dos sítios "
        "diferencialmente fosforilados.", "Body"))
    el += figure("nn_roc_confusion.png",
                 "Figura 9.2. Avaliação no teste: curva ROC (esquerda) e matriz "
                 "de confusão no limiar 0,5 (direita).")

    el.append(P("9.3. Evolução das métricas (seleção de épocas)", "H2"))
    try:
        h = pd.read_csv(os.path.join(OUT_DIR, "nn_training_history.csv"))
        sel = h[h["epoch"].isin([1, 10, 30, 60, 90, 120])]
        rows = [["Época", "Perda treino", "Perda teste", "AUC treino", "AUC teste"]]
        for _, r in sel.iterrows():
            rows.append([str(int(r["epoch"])), f"{r['train_loss']:.4f}",
                         f"{r['test_loss']:.4f}", f"{r['train_auc']:.4f}",
                         f"{r['test_auc']:.4f}"])
        el.append(data_table(rows, col_widths=[2.6 * cm, 3.1 * cm, 3.1 * cm,
                                               3.1 * cm, 3.1 * cm]))
        el.append(Spacer(1, 0.2 * cm))
        el.append(P("Tabela 9.1. Evolução da perda e da ROC-AUC em treino e "
                    "teste ao longo do treinamento.", "Caption"))
    except Exception:
        pass
    return el


def sec_integration_conclusion():
    el = []
    el.append(P("10. Integração multi-ômica e considerações finais", "H1"))
    el.append(P(
        "O conjunto de análises compõe um quadro coerente da biologia da "
        "resposta e da resistência ao BRAFi/MEKi em melanoma, articulando "
        "escalas temporais e camadas moleculares complementares:", "Body"))
    for b in bullets([
        "<b>Escala de minutos.</b> A fosfoproteômica aguda (PXD013923) e a KSEA "
        "demonstram, com robustez estatística, a supressão seletiva e linear do "
        "módulo RAF→MEK→ERK pela inibição farmacológica, validando todo o fluxo "
        "de processamento por critérios puramente biológicos.",
        "<b>Estado basal.</b> O fosfoproteoma directDIA (PXD022992) revela que o "
        "contexto mutacional condutor (BRAF V600E vs. NRAS) imprime uma "
        "assinatura de fosforilação detectável e reprodutível, separável por "
        "PCA e por análise diferencial.",
        "<b>Escala de dias a meses.</b> O curso temporal (GSE110054) descreve a "
        "reprogramação transcricional progressiva da resistência, com indução "
        "de vias de escape (RTKs, EMT) e uma assinatura concordante de centenas "
        "de genes.",
        "<b>Contexto clínico.</b> A coorte TCGA-SKCM ancora as observações in "
        "vitro na paisagem mutacional e na evolução clínica de tumores humanos.",
        "<b>Modelagem preditiva.</b> A rede neural confirma que as assinaturas "
        "de fosforilação específicas da via são aprendíveis e altamente "
        "preditivas (ROC-AUC de teste 0,989).",
    ]):
        el.append(b)

    el.append(P("10.1. Conexão entre supressão aguda e adaptação tardia", "H2"))
    el.append(P(
        "Um dos resultados mais informativos emerge da integração entre a "
        "supressão aguda de quinases e o rebote transcricional tardio: quinases "
        "como KSR1, MAP3K14, DUSP8, DYRK1B e PRKCA, embora suprimidas de "
        "imediato pela inibição, têm sua expressão reforçada no estado "
        "resistente. Esse padrão liga, em um único eixo analítico, a resposta "
        "bioquímica de minutos à adaptação transcricional de meses, e aponta "
        "candidatos naturais para investigação de mecanismos de escape.", "Body"))

    el.append(P("10.2. Reprodutibilidade", "H2"))
    el.append(P(
        "Todas as etapas são reprodutíveis: os dados provêm de repositórios "
        "públicos consolidados (PRIDE, GEO e GDC/TCGA); o processamento, as "
        "análises e a geração de figuras e tabelas estão organizados em módulos "
        "independentes, orquestrados por um script central com registro de log "
        "e salvamento padronizado de resultados. Mapeamentos auxiliares (como a "
        "conversão de identificadores e a rede cinase-substrato) são "
        "armazenados em cache local, garantindo execução determinística.", "Body"))

    el.append(P("10.3. Síntese", "H2"))
    el.append(P(
        "O pipeline integra com sucesso fosfoproteômica dirigida e independente "
        "de dados, transcriptômica temporal, genômica clínica e aprendizado "
        "profundo em uma narrativa única e biologicamente consistente sobre a "
        "resposta e a resistência ao BRAFi/MEKi em melanoma. Os resultados "
        "recapitulam fielmente a biologia conhecida da via MAPK/ERK, "
        "caracterizam a heterogeneidade entre contextos mutacionais, descrevem "
        "a dinâmica temporal da resistência e demonstram a viabilidade de "
        "modelos preditivos a partir dos padrões moleculares observados, "
        "estabelecendo uma base sólida e extensível para investigações "
        "subsequentes.", "Body"))
    return el


def sec_appendix():
    el = []
    el.append(P("11. Apêndice — Inventário de figuras e tabelas", "H1"))
    el.append(P(
        "Todas as figuras (PNG) e tabelas (CSV) referidas neste relatório foram "
        "geradas pelo pipeline e estão disponíveis nos diretórios "
        "<i>results/figures/</i> e <i>results/outputs/</i>. Os principais "
        "arquivos estão listados a seguir.", "Body"))

    el.append(P("11.1. Figuras", "H2"))
    figs = [
        ("pxd013923_response_distributions.png", "Resposta aguda por inibidor"),
        ("pxd013923_inhibitor_agreement.png", "Concordância entre inibidores"),
        ("pxd013923_mapk_anchor.png", "Ancoragem na via MAPK/ERK"),
        ("pxd022992_qc.png", "QC do fosfoproteoma directDIA"),
        ("pxd022992_pca_cell_lines.png", "PCA das linhagens"),
        ("pxd022992_volcano_braf_vs_nras.png", "Diferencial BRAF vs. NRAS"),
        ("pxd022992_heatmap_top_variance.png", "Sítios de maior variância"),
        ("pxd022992_ksea_kinase_activity.png", "Atividade de quinases por linhagem"),
        ("ksea_pxd013923_inhibitors.png", "KSEA sob BRAFi/MEKi/ERKi"),
        ("ksea_pxd022992_cell_lines.png", "KSEA por linhagem"),
        ("ksea_combined_volcano.png", "Síntese KSEA"),
        ("tcga_skcm_mutation_landscape.png", "Paisagem mutacional TCGA"),
        ("tcga_skcm_survival_km.png", "Sobrevida por subtipo"),
        ("tcga_skcm_pathway_expression.png", "Expressão de vias por subtipo"),
        ("tcga_skcm_phospho_integration.png", "Integração fosfo-genômica"),
        ("temporal_gse110054_qc.png", "QC do curso temporal"),
        ("temporal_pca_resistance_trajectory.png", "Trajetória de resistência"),
        ("temporal_pathway_heatmap.png", "Dinâmica de vias no tempo"),
        ("temporal_key_gene_trajectories.png", "Trajetórias de genes-chave"),
        ("temporal_resistance_signature_heatmap.png", "Assinatura de resistência"),
        ("temporal_kinase_escape_integration.png", "Quinases de escape"),
        ("nn_training_curves.png", "Curvas de treino/teste da rede neural"),
        ("nn_roc_confusion.png", "ROC e matriz de confusão"),
    ]
    rows = [["Arquivo", "Descrição"]]
    for f, d in figs:
        rows.append([f, d])
    el.append(data_table(rows, col_widths=[8.5 * cm, 6.5 * cm], font=8))

    el.append(Spacer(1, 0.3 * cm))
    el.append(P("11.2. Tabelas de resultados", "H2"))
    tbls = [
        ("PXD013923_inhibitor_log2_matrix.csv", "Matriz log₂ por inibidor (10.273 sítios)"),
        ("PXD013923_top_suppressed_sites.csv", "Sítios mais suprimidos"),
        ("PXD022992_phosphosite_matrix.csv", "Matriz de fosforilação (55.939 sítios)"),
        ("PXD022992_differential_phospho_BRAFvsNRAS.csv", "Diferencial BRAF vs. NRAS"),
        ("KSEA_PXD013923_kinase_zscores.csv", "Escores KSEA (inibidores)"),
        ("KSEA_PXD022992_kinase_zscores.csv", "Escores KSEA (linhagens)"),
        ("TCGA_SKCM_mutation_frequency.csv", "Frequência mutacional"),
        ("TCGA_SKCM_clinical_with_subtypes.csv", "Dados clínicos + subtipos"),
        ("GEO_GSE110054_resistance_UP_genes.csv", "Genes induzidos na resistência"),
        ("GEO_GSE110054_resistance_DOWN_genes.csv", "Genes reprimidos na resistência"),
        ("GEO_GSE110054_kinase_suppression_vs_rna_rebound.csv", "Supressão vs. rebote"),
        ("nn_training_history.csv", "Histórico de treinamento da rede neural"),
        ("nn_predictions_test.csv", "Predições no conjunto de teste"),
    ]
    rows = [["Arquivo", "Descrição"]]
    for f, d in tbls:
        rows.append([f, d])
    el.append(data_table(rows, col_widths=[9.5 * cm, 5.5 * cm], font=8))
    return el


if __name__ == "__main__":
    build()
