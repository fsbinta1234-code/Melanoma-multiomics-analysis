# -*- coding: utf-8 -*-
"""
generate_full_report_en.py
==============================================================================
Generates an EXTENSIVE and DETAILED technical-scientific report (English) of the
entire multi-omics analysis project on BRAFi/MEKi resistance in melanoma, in PDF.

For each public dataset used, the document covers:
  • dataset description and provenance;
  • detailed methodology of processing and analysis;
  • quantitative results, figures and tables.

Integrated datasets:
  - PXD013923  (acute BRAFi/MEKi/ERKi phosphoproteome, SILAC, A375)
  - PXD022992  (directDIA phosphoproteome, 6 melanoma cell lines)
  - OmniPath   (kinase-substrate network for KSEA)
  - TCGA-SKCM  (clinical genomics/transcriptomics, 470 patients)
  - GSE110054  (temporal transcriptome of BRAFi resistance)
  - Neural network (differential phosphosite classifier)

Output: Relatorio_Completo_Melanoma_MultiOmica.pdf  (English content)
==============================================================================
"""
import os
import datetime

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, KeepTogether, NextPageTemplate,
)
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── fonts with full Unicode coverage (subscripts, Greek, arrows) ─────────────
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

# ── institutional color palette ──────────────────────────────────────────────
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
# Styles
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
    ss.add(ParagraphStyle(
        "Note", parent=ss["Normal"], fontSize=8.5, leading=12,
        alignment=TA_JUSTIFY, textColor=GREY, fontName=FONT_I,
        leftIndent=8, spaceBefore=2, spaceAfter=10))

    # Apply the DejaVu (Unicode) font to every defined style; styles that
    # already fix a variant (italic/bold) are preserved above.
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
# Content helpers
# =============================================================================
def P(text, style="Body"):
    return Paragraph(text, STYLES[style])


def bullets(items, style="MyBullet"):
    return [Paragraph(f"•&nbsp;&nbsp;{t}", STYLES[style]) for t in items]


def figure(path, caption, max_w=15.5 * cm, max_h=20 * cm):
    """Insert a figure fitted to the width, preserving aspect ratio."""
    full = os.path.join(FIG_DIR, path)
    if not os.path.exists(full):
        return [P(f"<i>[missing figure: {path}]</i>", "Caption")]
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
    """Styled table from a list of lists (strings)."""
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


def note(text):
    return Paragraph(f"<b>Data window.</b> {text}", STYLES["Note"])


# =============================================================================
# Header / footer
# =============================================================================
def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(MIDGREY)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.4 * cm, w - 2 * cm, 1.4 * cm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.0 * cm,
                      "Multi-omics Analysis of BRAFi/MEKi Resistance in Melanoma")
    canvas.drawRightString(w - 2 * cm, 1.0 * cm, f"Page {doc.page}")
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
# Document assembly
# =============================================================================
def build():
    doc = BaseDocTemplate(
        PDF_NAME, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Report — Multi-omics Analysis of Melanoma",
        author="Melanoma Multi-omics Pipeline")

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
    story += sec_phase1_acute()
    story.append(PageBreak())
    story += sec_phase2_rewiring()
    story.append(PageBreak())
    story += sec_phase3_consolidated()
    story.append(PageBreak())
    story += sec_integration_conclusion()
    story.append(PageBreak())
    story += sec_appendix()

    doc.build(story)
    print(f"[SUCCESS] Report generated: {PDF_NAME}")


# =============================================================================
# Sections
# =============================================================================
def cover():
    el = []
    el.append(Spacer(1, 2.2 * cm))
    el.append(P("TECHNICAL-SCIENTIFIC REPORT", "CoverSubtitle"))
    el.append(Spacer(1, 0.4 * cm))
    el.append(P("Multi-omics Analysis of BRAF/MEK Inhibitor "
                "Resistance in Melanoma", "CoverTitle"))
    el.append(Spacer(1, 0.6 * cm))
    el.append(P("Quantification of kinase activity by targeted and "
                "data-independent phosphoproteomics, temporal transcriptional "
                "profiles of acquired resistance, clinical genomic "
                "characterization, and predictive modeling by deep learning",
                "CoverSubtitle"))
    el.append(Spacer(1, 1.6 * cm))

    box = [
        ["Datasets", "PXD013923 · PXD022992 · TCGA-SKCM · GSE110054 · OmniPath"],
        ["Modalities", "Phosphoproteomics · Proteomics · Transcriptomics · Genomics"],
        ["Phosphosites", "10,273 (SILAC) + 55,939 (DIA)"],
        ["Clinical patients", "470 (TCGA-SKCM)"],
        ["Time course", "DMSO → 3 d → 21 d → 90 d (2 cell lines)"],
        ["Predictive model", "Neural network — test ROC-AUC 0.989"],
    ]
    rows = [[Paragraph(f"<b>{k}</b>", STYLES["TblCell"]),
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
    today = datetime.date.today().strftime("%B %d, %Y")
    el.append(P(f"Document generated automatically from pipeline results · "
                f"{today}", "CoverMeta"))
    return el


def summary_and_toc():
    el = []
    el.append(P("Executive Summary", "H1"))
    el.append(P(
        "This report provides a detailed description of an integrated "
        "multi-omics analysis pipeline aimed at studying the response and "
        "resistance of melanoma cells to BRAF and MEK kinase inhibitors "
        "(BRAFi/MEKi), the central therapeutic axis in <i>BRAF</i> V600-mutant "
        "melanoma. The work brings together five public and complementary data "
        "sources, spanning from the immediate biochemical response of the "
        "MAPK/ERK pathway (minutes scale) to the transcriptional reprogramming "
        "that accompanies the acquisition of resistance (months scale), through "
        "the mutational and clinical landscape of a human tumor cohort.", "Body"))
    el.append(P(
        "The analysis of the acute response to BRAFi, MEKi and ERKi (dataset "
        "PXD013923) demonstrates selective and statistically robust suppression "
        "of phosphorylation at canonical MAPK/ERK pathway sites, with strong "
        "agreement between the three inhibitors (Pearson r between 0.71 and "
        "0.76), confirming the linearity of the RAF→MEK→ERK module. Kinase-"
        "Substrate Enrichment Analysis (KSEA), supported by the experimental "
        "kinase-substrate network from OmniPath, accurately recovers the "
        "suppression of BRAF (z = −6.6), RAF1 (z = −6.5), MAP2K1 (z = −6.5) and "
        "MAPK1/3, biologically validating the entire processing workflow.", "Body"))
    el.append(P(
        "The characterization of the basal phosphoproteome of six cell lines by "
        "data-independent acquisition (directDIA, PXD022992) clearly separates "
        "the <i>BRAF</i> V600E lines from the <i>NRAS</i>-mutant ones and "
        "identifies thousands of differentially phosphorylated sites. The "
        "clinical cohort TCGA-SKCM (470 patients) provides the mutation "
        "frequency of the main driver genes and the molecular stratification of "
        "melanoma, while the GSE110054 time course reveals the progressive "
        "transcriptional signatures of acquired resistance. Finally, a neural "
        "network classifier trained on real data reaches a test ROC-AUC of "
        "0.989, demonstrating that the pathway-specific phosphorylation patterns "
        "are predictive and learnable.", "Body"))

    el.append(P(
        "The report is organized around the three biological phases of the "
        "signaling-transition-kinetics model of combined BRAFi + MEKi treatment: "
        "<b>Phase 1 — Acute Suppression</b> (6–24 h), <b>Phase 2 — Adaptive "
        "Rewiring</b> (48–72 h), and <b>Phase 3 — Consolidated Resistance</b> "
        "(3–90 days). Each dataset and analysis is presented within the phase it "
        "informs, so the results follow the temporal arc from immediate target "
        "blockade to stable, invasive resistance.", "Body"))

    el.append(P("Table of Contents", "H2"))
    toc = [
        "1.&nbsp;&nbsp;Introduction and biological context",
        "2.&nbsp;&nbsp;Datasets and provenance",
        "3.&nbsp;&nbsp;Methodology",
        "4.&nbsp;&nbsp;Phase 1 — Acute Suppression (6–24 h)",
        "5.&nbsp;&nbsp;Phase 2 — Adaptive Rewiring (48–72 h)",
        "6.&nbsp;&nbsp;Phase 3 — Consolidated Resistance (3–90 days)",
        "7.&nbsp;&nbsp;Multi-omics integration and concluding remarks",
        "8.&nbsp;&nbsp;Appendix — Inventory of figures and tables",
    ]
    for t in toc:
        el.append(Paragraph(t, STYLES["TOCItem"]))
    return el


def sec_introduction():
    el = []
    el.append(P("1. Introduction and biological context", "H1"))
    el.append(P(
        "Cutaneous melanoma is the most lethal skin tumor, and its molecular "
        "biology is dominated by the hyperactivation of the MAPK/ERK signaling "
        "pathway (RAS→RAF→MEK→ERK). Approximately half of cutaneous melanomas "
        "carry activating mutations in the <i>BRAF</i> gene, largely dominated "
        "by the V600 substitution (especially V600E), which keeps the BRAF "
        "kinase constitutively active and drives cell proliferation "
        "independently of external stimuli.", "Body"))
    el.append(P(
        "This molecular dependency made the MAPK/ERK pathway a prime "
        "therapeutic target. BRAF inhibitors (BRAFi, e.g. vemurafenib and "
        "dabrafenib) and MEK inhibitors (MEKi, e.g. trametinib and "
        "cobimetinib), particularly in the BRAFi+MEKi combination, produce "
        "striking initial clinical responses. The durability of this response, "
        "however, is limited by the nearly universal emergence of "
        "<b>acquired resistance</b>, frequently associated with reactivation of "
        "the MAPK pathway itself and the activation of escape routes such as "
        "the reprogramming of receptor tyrosine kinases (RTKs) and the "
        "transition to a mesenchymal-invasive state.", "Body"))
    el.append(P(
        "Understanding resistance requires observing the system across multiple "
        "time scales and multiple molecular layers. Protein phosphorylation is "
        "the most direct readout of kinase activity state and responds within "
        "minutes to pharmacological inhibition; transcriptional reprogramming, "
        "in turn, develops over days to months and consolidates the resistant "
        "phenotype; and the tumor genomic landscape defines the mutational "
        "context in which these adaptations take place. This project "
        "articulates these layers within a single reproducible pipeline.", "Body"))

    el.append(P("1.1. Project objectives", "H2"))
    for b in bullets([
        "Quantify the activity state of MAPK/ERK pathway kinases from "
        "high-resolution phosphoproteomics, using targeted and data-independent "
        "acquisition.",
        "Evaluate the dynamic response to kinase inhibition, from acute "
        "suppression (minutes) to late transcriptional adaptation (months).",
        "Characterize molecular heterogeneity between cell lines and between "
        "clinical tumor subtypes (BRAF V600E vs. NRAS-mutant).",
        "Build predictive models of the molecular state associated with the BRAF "
        "pathway, integrating network-based (KSEA) and machine-learning "
        "approaches.",
        "Validate the entire workflow with public data from established "
        "repositories (PRIDE, GEO and GDC/TCGA).",
    ]):
        el.append(b)

    el.append(P("1.2. The three-phase signaling-transition-kinetics model", "H2"))
    el.append(P(
        "The response of a <i>BRAF</i> V600E melanoma cell to combined BRAFi + "
        "MEKi is not a single event but a temporal trajectory. Following the "
        "signaling-transition-kinetics model, this report organizes the evidence "
        "into three consecutive phases along the treatment timeline "
        "(6 h → 24 h → 48 h → 72 h → 3 d → 11 d → 21 d → 90 d):", "Body"))

    el += figure("phase2_model_diagram.png",
                 "Figure 1.1. The signaling-transition-kinetics reference model "
                 "of combined BRAFi + MEKi treatment in melanoma: Phase 1 (acute "
                 "suppression), Phase 2 (adaptive rewiring) and Phase 3 "
                 "(consolidated resistance), along the 6 h → 90 d timeline. This "
                 "report is structured to follow this model.")

    rows = [
        ["Phase", "Window", "Hallmark", "Evidence in this report"],
        ["Phase 1 — Acute Suppression", "6–24 h",
         "MAPK pathway shutdown; KSEA scores and ERK1/2 activation drop; full target blockade",
         "PXD013923 acute inhibition; KSEA on BRAFi/MEKi/ERKi (§4)"],
        ["Phase 2 — Adaptive Rewiring", "48–72 h",
         "Loss of negative feedback; emergency activation of parallel routes; PI3K/AKT/mTOR and RTK (AXL, PDGFR) rebound",
         "Rebounding kinases; RTK reactivation; cross-genotype kinase activity (§5)"],
        ["Phase 3 — Consolidated Resistance", "3–90 d",
         "Stable, fully integrated alternative signaling; concordant multi-omics signatures; phenotypic switch to invasive cells",
         "GSE110054 time course; PXD022992 genotype signature; TCGA-SKCM; predictive model (§6)"],
    ]
    rows = [[Paragraph(c, STYLES["TblHeader"] if i == 0 else STYLES["TblCell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(rows, colWidths=[3.6 * cm, 1.7 * cm, 5.6 * cm, 4.6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.6, MIDGREY),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, MIDGREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTGREY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    el.append(t)
    el.append(Spacer(1, 0.2 * cm))
    el.append(P("Table 1.1. The three phases of combined BRAFi + MEKi response, "
                "with their temporal window, biological hallmark, and the "
                "datasets that provide evidence for each in this report.",
                "Caption"))
    return el


def sec_datasets():
    el = []
    el.append(P("2. Datasets and provenance", "H1"))
    el.append(P(
        "Five public datasets were integrated, selected to complementarily "
        "cover the different time scales and molecular layers of the problem. "
        "The table below summarizes each source; the subsections detail the "
        "experimental design and format.", "Body"))

    raw = [
        ["Accession", "Type", "Content", "Scale / dimension"],
        ["PXD013923", "SILAC phosphoproteome", "BRAFi/MEKi/ERKi in A375", "30 min · 10,273 sites"],
        ["PXD022992", "directDIA phosphoproteome", "6 melanoma cell lines", "basal · 55,939 sites"],
        ["OmniPath", "Kinase-substrate network", "Enzyme→site relations", "39,037 relations"],
        ["TCGA-SKCM", "Genomics + clinical", "Cutaneous melanoma", "470 patients"],
        ["GSE110054", "Temporal RNA-seq", "BRAFi time course", "25,222 genes · 10 samples"],
    ]
    el.append(data_table(raw, col_widths=[2.6 * cm, 4.0 * cm, 4.4 * cm, 4 * cm]))
    el.append(Spacer(1, 0.3 * cm))

    el.append(P("2.1. PXD013923 — Acute response to RAF/MEK/ERK inhibitors", "H2"))
    el.append(P(
        "Dataset deposited in the PRIDE/EBI repository, corresponding to an "
        "in-depth study of the RAF–MEK–ERK module and its immediate downstream "
        "effectors. A375 melanoma cells (<i>BRAF</i> V600E) were treated for "
        "30 minutes with a RAF inhibitor (BRAFi/dabrafenib), a MEK inhibitor "
        "(MEKi/trametinib) or an ERK inhibitor (ERKi/SCH772984), in a SILAC "
        "isotopic labeling design, in which the heavy/light ratios "
        "(<i>Ratio H/L normalized</i>) express the change in phosphorylation "
        "between the treated condition and control. Spectral processing was "
        "performed in MaxQuant; protein identifiers are of the Ensembl (ENSP) "
        "type, mapped to gene symbols via the mygene.info service "
        "(2,921 of 3,540 identifiers; 82.5% coverage). The very short treatment "
        "time captures the immediate biochemical response, before any "
        "transcriptional reprogramming.", "Body"))

    el.append(P("2.2. PXD022992 — directDIA phosphoproteome of six cell lines", "H2"))
    el.append(P(
        "Label-free phosphoproteome profile obtained by data-independent "
        "acquisition in directDIA mode (a member of the guidedDIA strategy "
        "family), spanning six melanoma cell lines: A375, SH-4, SK-MEL-28 and "
        "RPMI-7951 (carrying <i>BRAF</i> V600E) and G361 and SK-MEL-31 "
        "(carrying <i>NRAS</i> mutations). Each cell line was measured in two "
        "technical replicates. The quantification report (Spectronaut format, "
        "TSV content) contains columns for gene (<i>PG.Genes</i>), modified "
        "sequence (<i>EG.ModifiedSequence</i>), PTM site location "
        "(<i>EG.ProteinPTMLocations</i>) and intensity per sample "
        "(<i>EG.TotalQuantity</i>). This dataset provides the basal "
        "phosphorylation landscape and the basis for comparing the BRAF and "
        "NRAS mutational contexts.", "Body"))

    el.append(P("2.3. OmniPath — Kinase-substrate network for KSEA", "H2"))
    el.append(P(
        "To infer kinase activity from the measured phosphorylation sites, the "
        "OmniPath enzyme-substrate network was used, which aggregates curated "
        "relations from established databases (including PhosphoSitePlus, "
        "SIGNOR and others). The version used contains 39,037 phosphorylation "
        "relations covering 1,648 kinases, with annotation of enzyme gene, "
        "substrate gene, residue and position — allowing direct mapping to the "
        "<i>GENE_residueposition</i> site format used in the phosphorylation "
        "matrices.", "Body"))

    el.append(P("2.4. TCGA-SKCM — Genomic and clinical cohort", "H2"))
    el.append(P(
        "Data from the TCGA Skin Cutaneous Melanoma project, obtained via the "
        "public NCI Genomic Data Commons (GDC) API. Clinical and survival data "
        "were retrieved for 470 patients, along with the catalog of somatic "
        "mutations of the main melanoma driver genes "
        "(<i>BRAF, NRAS, NF1, PTEN, CDKN2A, KIT, MAP2K1, RAC1, PPP6C, PREX2, "
        "IDH1</i>) and gene expression quantifications by RNA-seq "
        "(STAR-Counts workflow) for a representative subset of samples, "
        "stratified by molecular subtype. This dataset anchors the in vitro "
        "observations in the context of human tumors.", "Body"))

    el.append(P("2.5. GSE110054 — Time course of BRAFi resistance", "H2"))
    el.append(P(
        "Gene expression series (RNA-seq) deposited in GEO, corresponding to a "
        "time-course experiment in which the melanoma cell lines M229 and M397 "
        "(both <i>BRAF</i> V600E) were treated with vemurafenib and sampled at "
        "multiple time points: control (DMSO), 3 days, 11–21 days and up to "
        "73–90 days of continuous treatment. The processed expression matrix "
        "(FPKM, 25,222 genes) allows tracking of the progressive transition "
        "from the sensitive to the resistant state, complementing, on the "
        "days-to-months scale, the acute response observed in the "
        "phosphoproteomics.", "Body"))
    return el


def sec_methodology():
    el = []
    el.append(P("3. Methodology", "H1"))
    el.append(P(
        "The pipeline was implemented in Python, with numerical processing in "
        "<i>NumPy</i>/<i>pandas</i>, statistics in <i>SciPy</i>, machine "
        "learning in <i>scikit-learn</i>, deep learning in <i>PyTorch</i> and "
        "visualization in <i>Matplotlib</i>/<i>Seaborn</i>. Each step is "
        "executed as an independent module and orchestrated by a central "
        "script, with logging and standardized saving of figures and tables. "
        "The following subsections describe the methodology by step.", "Body"))

    el.append(P("3.1. Phosphoproteomics processing and quality control", "H2"))
    el.append(P(
        "Phosphorylation sites were filtered by site localization probability "
        "(threshold ≥ 0.75), with removal of reverse (decoy) proteins and "
        "potential contaminants. Intensities were log<sub>2</sub>-transformed; "
        "null or negative SILAC ratios were treated as missing before the "
        "logarithmic transformation. For the directDIA matrix, undetected "
        "values (labeled <i>Filtered</i>) were converted to missing and, when "
        "required for analyses needing a complete matrix, imputed by a "
        "Perseus-style approach (replacement by the lower tail of each sample's "
        "distribution, at <i>mean − 1.8 × standard deviation</i>).", "Body"))

    el.append(P("3.2. Biological anchoring of the SILAC signal", "H2"))
    el.append(P(
        "Since the orientation of SILAC ratios can be ambiguous, the signal was "
        "anchored to known biology: inhibition of the MAPK module must "
        "<b>reduce</b> phosphorylation of the pathway's canonical sites. An "
        "anchor set of approximately 30 MAPK/ERK pathway genes and their "
        "canonical substrates was defined; the sign convention was fixed so "
        "that the median of the anchor sites is negative (suppression). This "
        "check simultaneously provides a quality control independent of the "
        "processing.", "Body"))

    el.append(P("3.3. Kinase-Substrate Enrichment Analysis (KSEA)", "H2"))
    el.append(P(
        "The activity of each kinase was estimated by KSEA: for a kinase "
        "<i>k</i> with a set of measured substrates <i>S<sub>k</sub></i>, the "
        "score is the standardized deviation of the substrate mean relative to "
        "the global mean, in the form <i>z = (m<sub>S</sub> − m<sub>global</sub>) "
        "/ (σ<sub>global</sub> / √n)</i>. Significance was assessed by a "
        "permutation test (1,000 resamplings of substrate sets of the same "
        "size), and only kinases with at least five measured substrates were "
        "scored. The substrate sets come from the OmniPath network.", "Body"))

    el.append(P("3.4. Differential analysis and statistics", "H2"))
    el.append(P(
        "Comparisons between groups (e.g. BRAF V600E vs. NRAS) were performed by "
        "Welch's t-test site by site, with effect magnitude expressed as the "
        "difference of means in log<sub>2</sub> (log<sub>2</sub>FC). Sites were "
        "considered differentially phosphorylated with <i>p</i> &lt; 0.05 and "
        "|log<sub>2</sub>FC| &gt; 1. For the time course, the monotonicity of "
        "the response was measured by Spearman correlation between treatment "
        "time and expression, with Bonferroni correction; resistance signatures "
        "were defined as genes with |ρ| &gt; 0.7 and adjusted <i>p</i> &lt; "
        "0.01 concordantly in both cell lines.", "Body"))

    el.append(P("3.5. Dimensionality reduction and stratification", "H2"))
    el.append(P(
        "Principal Component Analysis (PCA) was applied, after standardization, "
        "to visualize the sample structure — separation between cell lines and "
        "the resistance trajectory over time. Molecular stratification of the "
        "TCGA-SKCM cohort followed the priority <i>BRAF</i> V600 &gt; "
        "<i>NRAS</i> &gt; <i>NF1</i> &gt; triple wild-type, and overall survival "
        "was estimated by the Kaplan–Meier method per subtype.", "Body"))

    el.append(P("3.6. Predictive modeling by neural network", "H2"))
    el.append(P(
        "A classifier was built to predict whether a phosphorylation site is "
        "differentially phosphorylated between the BRAF V600E and NRAS contexts, "
        "from the intensities across the six cell lines. The set was balanced "
        "(equal number of differential and non-differential sites) and split "
        "into training and test in a stratified manner (70/30); standardization "
        "was fit only on the training set to avoid information leakage. The "
        "architecture is a multilayer perceptron (6 → 128 → 64 → 32 → 1) with "
        "batch normalization, ReLU activation and 0.3 <i>dropout</i>; training "
        "used binary cross-entropy loss, the AdamW optimizer with weight decay "
        "and a cosine-annealing learning-rate scheduler, over 120 epochs. The "
        "training and test loss and ROC-AUC curves were recorded per epoch.", "Body"))
    return el

def _phase_header(el, title, window, hallmark):
    el.append(P(title, "H1"))
    el.append(Paragraph(
        f"<b>Treatment window:</b> {window} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Hallmark:</b> {hallmark}", STYLES["KeyBox"]))


def sec_phase1_acute():
    el = []
    _phase_header(
        el, "4. Phase 1 — Acute Suppression (6–24 h)", "6 h – 24 h",
        "MAPK pathway shutdown; KSEA scores and ERK1/2 activation drop; the "
        "cell is driven into full target blockade.")
    el.append(P(
        "The first phase of combined BRAFi + MEKi treatment is the immediate, "
        "on-target collapse of MAPK/ERK signaling. Evidence comes from the acute "
        "(30-minute) inhibitor phosphoproteome of A375 cells (PXD013923) and from "
        "Kinase-Substrate Enrichment Analysis (KSEA) over the OmniPath network, "
        "which together confirm that the drugs hit the intended module and shut "
        "it down.", "Body"))
    el.append(note(
        "The available acute phosphoproteome measures a 30-minute treatment, "
        "which lies just before the diagram's 6–24 h window; it captures the "
        "on-target biochemical shutdown that initiates the acute-suppression "
        "phase. No 6–24 h phosphoproteome exists for this system, so this "
        "earliest available time point is used as the Phase-1 readout."))

    el.append(keybox(
        "Key result.",
        "Canonical MAPK/ERK phosphosites are selectively suppressed (anchor "
        "median log<sub>2</sub> ≈ −0.29 to −0.31 across the three inhibitors) "
        "while the rest of the phosphoproteome stays near zero, and KSEA scores "
        "the core kinases as strongly negative — the signature of full acute "
        "pathway shutdown."))

    # ---- 4.1 PXD013923 ----
    el.append(P("4.1. Acute phosphoproteome response (PXD013923)", "H2"))
    el.append(P(
        "After quality filtering, 10,273 phosphorylation sites were quantified; "
        "for each inhibitor the mean change in phosphorylation (log<sub>2</sub>) "
        "relative to control was computed. The distribution is centered near "
        "zero but has a pronounced negative tail: 346 (BRAFi), 331 (MEKi) and "
        "290 (ERKi) sites are strongly suppressed (log<sub>2</sub> &lt; −1), "
        "against far fewer induced sites.", "Body"))
    el += figure("pxd013923_response_distributions.png",
                 "Figure 4.1. Distribution of phosphorylation changes "
                 "(log₂, inhibitor/control) for BRAFi, MEKi and ERKi in A375 "
                 "(30 min). The negative tail concentrates the suppressed sites.")

    el.append(P("4.2. Linearity of the RAF→MEK→ERK module", "H2"))
    el.append(P(
        "The responses to the three inhibitors are strongly correlated: Pearson "
        "r of 0.756 (BRAFi vs. MEKi), 0.705 (BRAFi vs. ERKi) and 0.760 (MEKi "
        "vs. ERKi). This is the expected signature of a linear module, in which "
        "inhibition at any level propagates similar effects downstream.", "Body"))
    el += figure("pxd013923_inhibitor_agreement.png",
                 "Figure 4.2. Pairwise agreement between the responses to "
                 "BRAFi, MEKi and ERKi. The strong positive correlation reflects "
                 "the linearity of the RAF→MEK→ERK module.")

    el.append(P("4.3. Selective suppression of the pathway (anchor)", "H2"))
    el.append(P(
        "Against the rest of the phosphoproteome, the MAPK/ERK pathway sites "
        "shift systematically negative across all three inhibitors (median of "
        "−0.293 for BRAFi, −0.267 for MEKi and −0.310 for ERKi; anchor set of "
        "81 sites, overall median −0.295), while the remaining sites stay "
        "centered at zero — the suppression is specific to the target pathway. "
        "Among the most repressed BRAFi targets are the ERK-regulated repressor "
        "ERF and the ERK effector RPS6KA3 (RSK2).", "Body"))
    el += figure("pxd013923_mapk_anchor.png",
                 "Figure 4.3. MAPK/ERK pathway sites (red) vs. other sites "
                 "(grey). The suppression concentrates in the pathway's "
                 "canonical sites.", max_w=13 * cm)
    try:
        t = pd.read_csv(os.path.join(OUT_DIR, "PXD013923_top_suppressed_sites.csv"))
        t = t[t["inhibitor"] == "BRAFi"].head(8)
        rows = [["Site (gene_residue)", "Gene", "log₂FC (BRAFi)"]]
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
        el.append(P("Table 4.1. Phosphorylation sites most suppressed by BRAFi "
                    "(selection).", "Caption"))
    except Exception:
        pass

    # ---- 4.4 KSEA on inhibitors ----
    el.append(P("4.4. Kinase-activity collapse (KSEA)", "H2"))
    el.append(P(
        "KSEA over the OmniPath kinase-substrate network translates the measured "
        "sites into per-kinase activity scores. Under BRAFi, the core of the "
        "pathway shows the most negative scores — BRAF (z = −6.6), RAF1 "
        "(z = −6.5), MAP2K1/MEK1 (z = −6.5) and MAPK1/3 — all with p &lt; 0.001 "
        "in the permutation test, reconstructing the pathway's suppression "
        "hierarchy. Negative regulators such as the DUSP phosphatases "
        "(DUSP1/8/16) and the scaffold KSR1, whose phosphorylation is "
        "ERK-dependent, drop the most sharply.", "Body"))
    el += figure("ksea_pxd013923_inhibitors.png",
                 "Figure 4.4. KSEA scores (z) per kinase under BRAFi, MEKi and "
                 "ERKi. Blue tones indicate suppression of kinase activity.",
                 max_w=11.5 * cm)
    el += figure("phase1_ksea_dropping_bars.png",
                 "Figure 4.5. Phase-1 signature: KSEA scores of the MAPK core "
                 "drop sharply under BRAFi — the quantitative form of the "
                 "'KSEA scores drop / MAPK pathway shutdown' panel of the model.",
                 max_w=12.5 * cm)
    try:
        k = pd.read_csv(os.path.join(OUT_DIR, "KSEA_PXD013923_kinase_zscores.csv"))
        kcol = "kinase" if "kinase" in k.columns else k.columns[1]
        k = k.sort_values("BRAFi").head(10)
        rows = [["Kinase", "z (BRAFi)", "z (MEKi)", "z (ERKi)"]]
        for _, r in k.iterrows():
            rows.append([str(r[kcol]), f"{r['BRAFi']:+.2f}",
                         f"{r['MEKi']:+.2f}", f"{r['ERKi']:+.2f}"])
        el.append(data_table(rows, col_widths=[5 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm]))
        el.append(Spacer(1, 0.2 * cm))
        el.append(P("Table 4.2. Ten most-suppressed kinases under BRAFi "
                    "(KSEA z-score).", "Caption"))
    except Exception:
        pass
    return el


def sec_phase2_rewiring():
    el = []
    _phase_header(
        el, "5. Phase 2 — Adaptive Rewiring (48–72 h)", "48 h – 72 h",
        "Loss of negative feedback triggers emergency activation of parallel "
        "routes; PI3K/AKT/mTOR and RTK (AXL, PDGFR) activities rebound; the cell "
        "begins to bypass the block.")
    el.append(P(
        "Once the acute block is established, the cell escapes it. The loss of "
        "ERK-dependent negative feedback releases upstream receptors, and "
        "alternative kinase routes are activated to sustain survival and "
        "proliferation. This phase gathers three complementary readouts of that "
        "rewiring: kinases that rebound after acute suppression, the "
        "reactivation of receptor tyrosine kinases, and the parallel-pathway "
        "kinase activity that characterizes MAPK-independent signaling states.",
        "Body"))

    el.append(keybox(
        "Key result.",
        "Several kinases that are acutely suppressed by inhibition (KSR1, "
        "MAP3K14, DUSP8, DYRK1B, PRKCA) have their expression reinforced in the "
        "resistant state, and receptor tyrosine kinases (AXL, PDGFRB, MET, EGFR) "
        "are progressively reactivated — the molecular signature of emergency "
        "activation of parallel routes."))
    el.append(note(
        "The rewiring window (48–72 h) has no dedicated dataset; its molecular "
        "signature is reconstructed from the earliest arms of the resistance "
        "time course (GSE110054) and from the cross-genotype baseline "
        "(PXD022992). The pathway-score trajectories below (Figure 5.3) show the "
        "rise of parallel routes beginning within the first days of treatment."))

    # ---- 5.1 rebounding kinases ----
    el.append(P("5.1. Rebounding kinase activities", "H2"))
    el.append(P(
        "Cross-referencing the acute kinase suppression (KSEA at 30 min, "
        "PXD013923) with the late transcriptional change (expression in the "
        "resistant state, GSE110054) isolates kinases that, although suppressed "
        "immediately by the drug, have their gene expression reinforced once "
        "resistance sets in — natural candidates for the adaptive escape that "
        "drives rewiring. The most prominent are KSR1, MAP3K14, DUSP8, DYRK1B "
        "and PRKCA.", "Body"))
    el += figure("temporal_kinase_escape_integration.png",
                 "Figure 5.1. Acute kinase suppression (x-axis, KSEA 30 min) vs. "
                 "late transcriptional rebound (y-axis, resistant state). Escape "
                 "kinases — suppressed acutely but re-expressed late — are "
                 "highlighted.", max_w=12.5 * cm)

    # ---- 5.2 RTK reactivation ----
    el.append(P("5.2. Receptor tyrosine-kinase reactivation", "H2"))
    el.append(P(
        "The hallmark of adaptive rewiring is the reactivation of receptor "
        "tyrosine kinases (RTKs) that feed the parallel PI3K/AKT and RAS "
        "branches. Along the treatment time course, RTK genes — including AXL, "
        "PDGFRB, MET, EGFR and FGFR1 — rise progressively, together with the "
        "compensatory increase of AKT/mTOR-axis members, consistent with the "
        "emergency activation of survival routes as the MAPK block is bypassed.",
        "Body"))
    el += figure("temporal_key_gene_trajectories.png",
                 "Figure 5.2. Temporal trajectories of key resistance genes in "
                 "M229 and M397 under BRAFi — receptor tyrosine kinases (AXL, "
                 "PDGFRB, MET, EGFR, FGFR1), proliferation, EMT and survival "
                 "markers rise as the cell rewires.")
    el.append(P(
        "Aggregating these genes into pathway scores (mean change vs. the DMSO "
        "baseline) reproduces the model's rising panels directly: the "
        "PI3K/AKT/mTOR and RTK scores climb above baseline within days, while "
        "MAPK output — sharply suppressed at the first treated time point — "
        "rebounds later, the transcriptional counterpart of the phosphoproteomic "
        "rebound. EMT/invasion scores rise in parallel.", "Body"))
    el += figure("signaling_kinetics_pathway_trajectories.png",
                 "Figure 5.3. Pathway-activity trajectories over the BRAFi time "
                 "course (Δlog2 vs. DMSO, GSE110054): MAPK output drops then "
                 "rebounds; PI3K/AKT/mTOR, RTK (AXL/PDGFR/EGFR/MET) and "
                 "EMT/invasion rise — the diagram's rising panels realized in "
                 "real data.")

    # ---- 5.4 parallel-pathway kinase activity across genotypes ----
    el.append(P("5.4. Parallel-pathway signaling across genotypes (PXD022992)", "H2"))
    el.append(P(
        "The directDIA phosphoproteome of six melanoma cell lines provides a "
        "static model of the rewired, MAPK-independent state: the NRAS-mutant "
        "lines survive without a druggable BRAF V600E node and therefore "
        "exemplify the parallel-route signaling that BRAFi-resistant cells "
        "acquire. Kinase-activity scores by substrate set (ERK, AKT, mTOR, CDK "
        "and FAK/SRC) summarize how signaling is distributed differently across "
        "the genotypes, with the PI3K/AKT/mTOR and FAK/SRC axes prominent in the "
        "non-BRAF-V600E context.", "Body"))
    el += figure("pxd022992_ksea_kinase_activity.png",
                 "Figure 5.4. Kinase-pathway activity per cell line "
                 "(substrate mean, standardized): ERK, AKT, mTOR, CDK and "
                 "FAK/SRC.")
    el += figure("ksea_pxd022992_cell_lines.png",
                 "Figure 5.5. KSEA scores per kinase across the six melanoma "
                 "cell lines.", max_w=12 * cm)
    el += figure("ksea_combined_volcano.png",
                 "Figure 5.6. Comparative panel: kinase activity under BRAFi "
                 "(left) and the difference in kinase activity between A375 "
                 "(BRAF V600E) and G361 (NRAS) (right).")
    return el


def sec_phase3_consolidated():
    el = []
    _phase_header(
        el, "6. Phase 3 — Consolidated Resistance (3–90 days)", "3 d – 90 d",
        "Stable, fully integrated alternative signaling; concordant multi-omics "
        "signatures; phosphoproteomic rebound and phenotypic switching toward "
        "invasive cells.")
    el.append(P(
        "In the final phase the rewired state is consolidated into a stable, "
        "heritable resistant phenotype. The evidence spans the long-term "
        "transcriptional time course (GSE110054), the stable genotype-specific "
        "phosphoproteome (PXD022992), the clinical tumor landscape (TCGA-SKCM), "
        "and a predictive model that shows the resistant signature is learnable.",
        "Body"))

    el.append(keybox(
        "Key result.",
        "The resistant state carries a robust, concordant signature — 647 genes "
        "consistently induced and 267 repressed across two cell lines — with "
        "induction of EMT/invasion programs (phenotypic switching). The stable "
        "genotype-specific phosphoproteome is strong enough that a neural "
        "network predicts differential sites at test ROC-AUC 0.989."))
    el.append(note(
        "This is the one phase whose data window matches the model exactly: the "
        "GSE110054 time course spans 3 days to 90 days. The MAPK-output rebound "
        "seen in Figure 5.3 — sharp suppression at the first treated point "
        "followed by recovery toward or above baseline — is the transcriptional "
        "readout of the model's 'phosphoproteomic rebound'. A direct phospho "
        "time course is not available for this system."))

    # ---- 6.1 trajectory to resistance ----
    el.append(P("6.1. Trajectory to the resistant state (GSE110054)", "H2"))
    el.append(P(
        "The time course in M229 and M397 under vemurafenib (DMSO → 3 d → "
        "11–21 d → 73–90 d) traces the progressive transcriptional "
        "reprogramming of resistance. Principal-component analysis organizes the "
        "samples along a continuous trajectory from the sensitive state (DMSO) "
        "to acquired resistance, and the quality control confirms consistent, "
        "well-correlated samples.", "Body"))
    el += figure("temporal_gse110054_qc.png",
                 "Figure 6.1. Quality control of the time course: expression "
                 "distributions and sample correlation matrix.")
    el += figure("temporal_pca_resistance_trajectory.png",
                 "Figure 6.2. PCA trajectory from sensitivity to resistance in "
                 "M229 and M397 under BRAFi. The arrows indicate temporal "
                 "progression.", max_w=12.5 * cm)

    # ---- 6.2 stable signature + phenotypic switch ----
    el.append(P("6.2. Stable, concordant signature and phenotypic switch", "H2"))
    el.append(P(
        "A monotonicity analysis (Spearman) identifies, concordantly in both "
        "cell lines, 647 genes consistently induced and 267 consistently "
        "repressed over treatment (|ρ| &gt; 0.7). The pathway dynamics show the "
        "reactivation of RTK and PI3K/AKT/mTOR programs and the induction of "
        "EMT/invasion markers — the transcriptional face of the phenotypic "
        "switch toward invasive cells that defines consolidated resistance.",
        "Body"))
    el += figure("temporal_pathway_heatmap.png",
                 "Figure 6.3. Pathway expression dynamics over the BRAFi time "
                 "course.", max_w=11.5 * cm)
    el += figure("temporal_resistance_signature_heatmap.png",
                 "Figure 6.4. Resistance signature: genes consistently induced "
                 "and repressed (|ρ| > 0.7 in both cell lines).",
                 max_w=11.5 * cm)

    # ---- 6.3 stable genotype phosphoproteome ----
    el.append(P("6.3. Consolidated genotype-specific phosphoproteome (PXD022992)", "H2"))
    el.append(P(
        "The basal phosphoproteome of the six cell lines (55,939 sites; "
        "intensity medians homogeneous between 14.59 and 14.74) captures the "
        "stable signaling state of each genotype. The comparison between BRAF "
        "V600E and NRAS-mutant lines identifies 1,387 sites with higher "
        "phosphorylation in BRAF V600E and 1,304 higher in NRAS "
        "(Welch's t-test; <i>p</i> &lt; 0.05 and |log<sub>2</sub>FC| &gt; 1), "
        "and PCA separates the two contexts along PC1 — a reproducible, stable "
        "phosphorylation signature of the resistant genotype.", "Body"))
    el += figure("pxd022992_volcano_braf_vs_nras.png",
                 "Figure 6.5. Volcano plot of differential phosphorylation "
                 "between BRAF V600E and NRAS-mutant cell lines.")
    el += figure("pxd022992_pca_cell_lines.png",
                 "Figure 6.6. PCA of the six melanoma cell lines; PC1 follows "
                 "the mutational context (BRAF V600E vs. NRAS).", max_w=12 * cm)
    el += figure("pxd022992_heatmap_top_variance.png",
                 "Figure 6.7. Heatmap of the highest-variance phosphorylation "
                 "sites, with hierarchical clustering of the cell lines.",
                 max_w=11 * cm)

    # ---- 6.4 clinical anchoring ----
    el.append(P("6.4. Clinical anchoring (TCGA-SKCM)", "H2"))
    el.append(P(
        "The TCGA-SKCM cohort (470 patients) anchors the consolidated resistant "
        "state in human tumors. The driver-gene mutation frequencies recover the "
        "canonical melanoma pattern — <i>BRAF</i> most frequent, followed by "
        "<i>NRAS</i>, with <i>NF1</i>, <i>CDKN2A</i> and <i>PTEN</i> — and the "
        "molecular subtypes organize the disease's heterogeneity.", "Body"))
    try:
        m = pd.read_csv(os.path.join(OUT_DIR, "TCGA_SKCM_mutation_frequency.csv"))
        m = m.sort_values("pct", ascending=False).head(8)
        rows = [["Gene", "Mutated patients", "Frequency (%)"]]
        for _, r in m.iterrows():
            rows.append([str(r["gene"]), str(int(r["n_patients"])), f"{r['pct']:.1f}"])
        el.append(data_table(rows, col_widths=[5 * cm, 5 * cm, 4.5 * cm]))
        el.append(Spacer(1, 0.2 * cm))
        el.append(P("Table 6.1. Mutation frequency of the main driver genes in "
                    "TCGA-SKCM.", "Caption"))
    except Exception:
        pass
    el += figure("tcga_skcm_mutation_landscape.png",
                 "Figure 6.8. Mutational landscape of TCGA-SKCM: driver-gene "
                 "frequency (left) and molecular-subtype distribution (right).")
    el += figure("tcga_skcm_survival_km.png",
                 "Figure 6.9. Overall survival (Kaplan–Meier) by molecular "
                 "subtype in TCGA-SKCM.", max_w=12.5 * cm)
    el += figure("tcga_skcm_pathway_expression.png",
                 "Figure 6.10. Expression of MAPK/PI3K pathway genes by "
                 "molecular subtype (RNA-seq, TCGA-SKCM).", max_w=12.5 * cm)
    el += figure("tcga_skcm_phospho_integration.png",
                 "Figure 6.11. Phosphoproteomics–genomics integration: main "
                 "differential genes between BRAF V600E and NRAS.")

    # ---- 6.5 predictive model ----
    el.append(P("6.5. Predictive model of the resistant signature", "H2"))
    el.append(P(
        "To test whether the stable phosphorylation signature distinguishing "
        "BRAF V600E from NRAS is learnable, a neural network was trained on "
        "5,382 balanced sites (stratified 3,767 train / 1,615 test; 11,649 "
        "parameters; 120 epochs). It reaches a test ROC-AUC of 0.989 (best "
        "epoch 0.991), F1 of 0.95 and 95% accuracy, with training and test "
        "curves that evolve closely — the resistant signature is not only stable "
        "but predictable.", "Body"))
    el += figure("nn_training_curves.png",
                 "Figure 6.12. Neural-network training and test curves: loss "
                 "(left) and ROC-AUC (right) per epoch. The closeness of the "
                 "curves indicates good generalization.")
    el += figure("nn_roc_confusion.png",
                 "Figure 6.13. Test evaluation: ROC curve (left) and confusion "
                 "matrix at the 0.5 threshold (right).")
    try:
        h = pd.read_csv(os.path.join(OUT_DIR, "nn_training_history.csv"))
        sel = h[h["epoch"].isin([1, 10, 30, 60, 90, 120])]
        rows = [["Epoch", "Train loss", "Test loss", "Train AUC", "Test AUC"]]
        for _, r in sel.iterrows():
            rows.append([str(int(r["epoch"])), f"{r['train_loss']:.4f}",
                         f"{r['test_loss']:.4f}", f"{r['train_auc']:.4f}",
                         f"{r['test_auc']:.4f}"])
        el.append(data_table(rows, col_widths=[2.6 * cm, 3.1 * cm, 3.1 * cm,
                                               3.1 * cm, 3.1 * cm]))
        el.append(Spacer(1, 0.2 * cm))
        el.append(P("Table 6.2. Evolution of loss and ROC-AUC on training and "
                    "test over the course of training.", "Caption"))
    except Exception:
        pass
    return el


def sec_integration_conclusion():
    el = []
    el.append(P("7. Multi-omics integration and concluding remarks", "H1"))
    el.append(P(
        "Taken together, the analyses trace a single coherent arc through the "
        "three phases of the signaling-transition-kinetics model, connecting the "
        "immediate drug effect to the stable resistant phenotype:", "Body"))
    for b in bullets([
        "<b>Phase 1 — Acute Suppression (6–24 h).</b> The acute phosphoproteomics "
        "(PXD013923) and KSEA demonstrate, with statistical robustness, the "
        "selective and linear suppression of the RAF→MEK→ERK module by "
        "inhibition, validating the entire processing workflow on purely "
        "biological grounds.",
        "<b>Phase 2 — Adaptive Rewiring (48–72 h).</b> Kinases suppressed "
        "acutely rebound transcriptionally (KSR1, MAP3K14, DUSP8, DYRK1B, "
        "PRKCA), receptor tyrosine kinases (AXL, PDGFRB, MET) are reactivated, "
        "and cross-genotype kinase activity exposes the PI3K/AKT/mTOR and "
        "FAK/SRC parallel routes — the emergency activation that bypasses the "
        "block.",
        "<b>Phase 3 — Consolidated Resistance (3–90 d).</b> The time course "
        "(GSE110054) yields a stable, concordant signature (647 up, 267 down) "
        "with EMT/invasion induction (phenotypic switching); the stable "
        "genotype-specific phosphoproteome (PXD022992) and the clinical "
        "TCGA-SKCM cohort anchor the resistant state in reproducible and human "
        "tumor contexts.",
        "<b>Predictive modeling.</b> The neural network confirms that the "
        "consolidated phosphorylation signature is learnable and highly "
        "predictive (test ROC-AUC 0.989).",
    ]):
        el.append(b)

    el.append(P("7.1. Linking acute suppression and late adaptation", "H2"))
    el.append(P(
        "One of the most informative results emerges from the integration "
        "between acute kinase suppression and late transcriptional rebound: "
        "kinases such as KSR1, MAP3K14, DUSP8, DYRK1B and PRKCA, although "
        "suppressed immediately by inhibition, have their expression reinforced "
        "in the resistant state. This pattern links, along a single analytical "
        "axis, the minutes-scale biochemical response to the months-scale "
        "transcriptional adaptation, and points to natural candidates for the "
        "investigation of escape mechanisms.", "Body"))

    el.append(P("7.2. Reproducibility", "H2"))
    el.append(P(
        "All steps are reproducible: the data come from established public "
        "repositories (PRIDE, GEO and GDC/TCGA); the processing, analyses and "
        "generation of figures and tables are organized into independent "
        "modules, orchestrated by a central script with logging and "
        "standardized saving of results. Auxiliary mappings (such as identifier "
        "conversion and the kinase-substrate network) are stored in a local "
        "cache, ensuring deterministic execution.", "Body"))

    el.append(P("7.3. Synthesis", "H2"))
    el.append(P(
        "The pipeline successfully integrates targeted and data-independent "
        "phosphoproteomics, temporal transcriptomics, clinical genomics and "
        "deep learning into a single, biologically consistent narrative about "
        "the response and resistance to BRAFi/MEKi in melanoma. The results "
        "faithfully recapitulate the known biology of the MAPK/ERK pathway, "
        "characterize the heterogeneity across mutational contexts, describe the "
        "temporal dynamics of resistance, and demonstrate the feasibility of "
        "predictive models from the observed molecular patterns, establishing a "
        "solid and extensible foundation for subsequent investigations.", "Body"))
    return el


def sec_appendix():
    el = []
    el.append(P("8. Appendix — Inventory of figures and tables", "H1"))
    el.append(P(
        "All figures (PNG) and tables (CSV) referenced in this report were "
        "generated by the pipeline and are available in the "
        "<i>results/figures/</i> and <i>results/outputs/</i> directories. The "
        "main files are listed below.", "Body"))

    el.append(P("8.1. Figures", "H2"))
    figs = [
        ("phase2_model_diagram.png", "Signaling-transition-kinetics reference model"),
        ("phase1_ksea_dropping_bars.png", "Phase 1 — KSEA scores drop (MAPK core)"),
        ("signaling_kinetics_pathway_trajectories.png", "Pathway-activity trajectories over time"),
        ("pxd013923_response_distributions.png", "Acute response per inhibitor"),
        ("pxd013923_inhibitor_agreement.png", "Agreement between inhibitors"),
        ("pxd013923_mapk_anchor.png", "Anchoring in the MAPK/ERK pathway"),
        ("pxd022992_qc.png", "directDIA phosphoproteome QC"),
        ("pxd022992_pca_cell_lines.png", "PCA of cell lines"),
        ("pxd022992_volcano_braf_vs_nras.png", "Differential BRAF vs. NRAS"),
        ("pxd022992_heatmap_top_variance.png", "Highest-variance sites"),
        ("pxd022992_ksea_kinase_activity.png", "Kinase activity per cell line"),
        ("ksea_pxd013923_inhibitors.png", "KSEA under BRAFi/MEKi/ERKi"),
        ("ksea_pxd022992_cell_lines.png", "KSEA per cell line"),
        ("ksea_combined_volcano.png", "KSEA synthesis"),
        ("tcga_skcm_mutation_landscape.png", "TCGA mutational landscape"),
        ("tcga_skcm_survival_km.png", "Survival by subtype"),
        ("tcga_skcm_pathway_expression.png", "Pathway expression by subtype"),
        ("tcga_skcm_phospho_integration.png", "Phospho-genomics integration"),
        ("temporal_gse110054_qc.png", "Time-course QC"),
        ("temporal_pca_resistance_trajectory.png", "Resistance trajectory"),
        ("temporal_pathway_heatmap.png", "Pathway dynamics over time"),
        ("temporal_key_gene_trajectories.png", "Key gene trajectories"),
        ("temporal_resistance_signature_heatmap.png", "Resistance signature"),
        ("temporal_kinase_escape_integration.png", "Escape kinases"),
        ("nn_training_curves.png", "Neural-network training/test curves"),
        ("nn_roc_confusion.png", "ROC and confusion matrix"),
    ]
    rows = [["File", "Description"]]
    for f, d in figs:
        rows.append([f, d])
    el.append(data_table(rows, col_widths=[8.5 * cm, 6.5 * cm], font=8))

    el.append(Spacer(1, 0.3 * cm))
    el.append(P("8.2. Result tables", "H2"))
    tbls = [
        ("PXD013923_inhibitor_log2_matrix.csv", "log₂ matrix per inhibitor (10,273 sites)"),
        ("PXD013923_top_suppressed_sites.csv", "Most-suppressed sites"),
        ("PXD022992_phosphosite_matrix.csv", "Phosphorylation matrix (55,939 sites)"),
        ("PXD022992_differential_phospho_BRAFvsNRAS.csv", "Differential BRAF vs. NRAS"),
        ("KSEA_PXD013923_kinase_zscores.csv", "KSEA scores (inhibitors)"),
        ("KSEA_PXD022992_kinase_zscores.csv", "KSEA scores (cell lines)"),
        ("signaling_kinetics_trajectory_scores.csv", "Pathway-score trajectories over time"),
        ("TCGA_SKCM_mutation_frequency.csv", "Mutation frequency"),
        ("TCGA_SKCM_clinical_with_subtypes.csv", "Clinical data + subtypes"),
        ("GEO_GSE110054_resistance_UP_genes.csv", "Genes induced in resistance"),
        ("GEO_GSE110054_resistance_DOWN_genes.csv", "Genes repressed in resistance"),
        ("GEO_GSE110054_kinase_suppression_vs_rna_rebound.csv", "Suppression vs. rebound"),
        ("nn_training_history.csv", "Neural-network training history"),
        ("nn_predictions_test.csv", "Predictions on the test set"),
    ]
    rows = [["File", "Description"]]
    for f, d in tbls:
        rows.append([f, d])
    el.append(data_table(rows, col_widths=[9.5 * cm, 5.5 * cm], font=8))
    return el


if __name__ == "__main__":
    build()
