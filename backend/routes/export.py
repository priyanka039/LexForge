# ─────────────────────────────────────────────
# routes/export.py
# Feature: Export Report as PDF
#
# Takes the last generated output (research
# answer, IRAC arguments, or opposition
# analysis) and creates a downloadable
# professional PDF report.
#
# Endpoint:
#   POST /api/export/report
# ─────────────────────────────────────────────

import io
from datetime import datetime

from fastapi          import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic         import BaseModel
from typing           import Optional, List

from reportlab.lib.pagesizes   import A4
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units       import cm
from reportlab.lib             import colors
from reportlab.platypus        import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, Table, TableStyle
)

router = APIRouter()


# ── Colour palette matching LexForge UI ──────
NAVY   = colors.HexColor('#0a0f1e')
GOLD   = colors.HexColor('#c9a84c')
CREAM  = colors.HexColor('#f5f0e8')
MUTED  = colors.HexColor('#6b7a99')
WHITE  = colors.white
RED    = colors.HexColor('#c0392b')
GREEN  = colors.HexColor('#16a34a')


# ── Request shapes ────────────────────────────
class PrecedentItem(BaseModel):
    case_name: str
    court:     Optional[str] = "Unknown"
    year:      Optional[str] = "Unknown"
    snippet:   Optional[str] = ""
    score:     Optional[float] = 0.0
    binding:   Optional[str] = "Persuasive"

class IracBlock(BaseModel):
    issue_title: str
    area_of_law: str
    priority:    str
    irac: dict   # {issue, rule, application, conclusion}
    precedents:  Optional[List[dict]] = []

class ExportRequest(BaseModel):
    report_type:  str          # "research" | "argument" | "opposition" | "debate"
    title:        str          # e.g. "Wrongful Termination Analysis"
    jurisdiction: Optional[str] = "High Court of Delhi"

    # Research fields
    query:        Optional[str] = None
    answer:       Optional[str] = None
    precedents:   Optional[List[PrecedentItem]] = []

    # Argument fields
    facts:        Optional[str] = None
    arguments:    Optional[List[IracBlock]] = []

    # Opposition fields
    argument:     Optional[str] = None
    risk_level:   Optional[str] = None
    weaknesses:   Optional[List[dict]] = []
    counter_args: Optional[List[dict]] = []
    strategy:     Optional[List[dict]] = []

    # Debate fields
    round1:       Optional[dict] = None
    round2:       Optional[dict] = None
    summary:      Optional[dict] = None


# ═════════════════════════════════════════════
# PDF BUILDER
# ═════════════════════════════════════════════
def build_pdf(req: ExportRequest) -> bytes:
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer,
        pagesize     = A4,
        rightMargin  = 2.5 * cm,
        leftMargin   = 2.5 * cm,
        topMargin    = 2.5 * cm,
        bottomMargin = 2.5 * cm,
    )

    styles = getSampleStyleSheet()

    # ── Custom styles ─────────────────────────
    title_style = ParagraphStyle(
        'LexTitle',
        parent    = styles['Title'],
        fontName  = 'Helvetica-Bold',
        fontSize  = 20,
        textColor = NAVY,
        spaceAfter= 6,
    )
    subtitle_style = ParagraphStyle(
        'LexSub',
        parent    = styles['Normal'],
        fontName  = 'Helvetica',
        fontSize  = 10,
        textColor = MUTED,
        spaceAfter= 4,
    )
    section_style = ParagraphStyle(
        'LexSection',
        parent    = styles['Heading2'],
        fontName  = 'Helvetica-Bold',
        fontSize  = 13,
        textColor = NAVY,
        spaceBefore=14,
        spaceAfter= 6,
    )
    body_style = ParagraphStyle(
        'LexBody',
        parent    = styles['Normal'],
        fontName  = 'Helvetica',
        fontSize  = 10,
        textColor = colors.HexColor('#1a1a2e'),
        leading   = 16,
        spaceAfter= 8,
    )
    label_style = ParagraphStyle(
        'LexLabel',
        parent    = styles['Normal'],
        fontName  = 'Helvetica-Bold',
        fontSize  = 9,
        textColor = GOLD,
        spaceAfter= 2,
    )
    irac_letter_style = ParagraphStyle(
        'IracLetter',
        parent    = styles['Normal'],
        fontName  = 'Helvetica-Bold',
        fontSize  = 22,
        textColor = GOLD,
        leading   = 26,
    )
    case_style = ParagraphStyle(
        'LexCase',
        parent    = styles['Normal'],
        fontName  = 'Helvetica-Bold',
        fontSize  = 10,
        textColor = NAVY,
        spaceAfter= 2,
    )
    meta_style = ParagraphStyle(
        'LexMeta',
        parent    = styles['Normal'],
        fontName  = 'Helvetica',
        fontSize  = 8,
        textColor = MUTED,
        spaceAfter= 3,
    )
    snippet_style = ParagraphStyle(
        'LexSnippet',
        parent    = styles['Normal'],
        fontName  = 'Helvetica-Oblique',
        fontSize  = 9,
        textColor = colors.HexColor('#4a4a6a'),
        leading   = 14,
        spaceAfter= 6,
    )

    elements = []

    # ── HEADER ────────────────────────────────
    # LexForge branding bar
    header_data = [[
        Paragraph('<font color="#c9a84c"><b>LEX</b></font><font color="#0a0f1e">FORGE</font>', ParagraphStyle('brand', fontName='Helvetica-Bold', fontSize=16)),
        Paragraph(f'<font color="#6b7a99">{datetime.now().strftime("%d %B %Y")}</font>', ParagraphStyle('date', fontName='Helvetica', fontSize=9, alignment=2))
    ]]
    header_table = Table(header_data, colWidths=[10*cm, 6*cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), CREAM),
        ('PADDING',     (0,0), (-1,-1), 10),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('ROUNDEDCORNERS', [4]),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 0.5*cm))

    # ── REPORT TITLE ──────────────────────────
    elements.append(Paragraph(req.title, title_style))
    elements.append(Paragraph(
        f'Jurisdiction: {req.jurisdiction}  ·  Generated by LexForge AI  ·  {req.report_type.title()} Report',
        subtitle_style
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=GOLD, spaceAfter=12))

    # ═══════════════════════════════════════════
    # RESEARCH REPORT
    # ═══════════════════════════════════════════
    if req.report_type == "research" and req.answer:
        elements.append(Paragraph("Legal Query", section_style))
        elements.append(Paragraph(req.query or "—", body_style))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))

        elements.append(Paragraph("Synthesized Answer", section_style))
        # Clean markdown from answer for PDF
        clean_answer = (req.answer
            .replace('**', '')
            .replace('##', '')
            .replace('[SOURCE ', '(Source ')
            .replace(']', ')')
        )
        for para in clean_answer.split('\n\n'):
            if para.strip():
                elements.append(Paragraph(para.strip(), body_style))

        if req.precedents:
            elements.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))
            elements.append(Paragraph("Retrieved Precedents", section_style))
            for p in req.precedents:
                elements.append(Paragraph(f"{p.case_name}", case_style))
                elements.append(Paragraph(
                    f"{p.court}  ·  {p.year}  ·  Relevance: {p.score}  ·  {p.binding}",
                    meta_style
                ))
                if p.snippet:
                    elements.append(Paragraph(f'"{p.snippet[:200]}..."', snippet_style))
                elements.append(Spacer(1, 0.2*cm))

    # ═══════════════════════════════════════════
    # ARGUMENT BUILDER REPORT
    # ═══════════════════════════════════════════
    elif req.report_type == "argument" and req.arguments:
        if req.facts:
            elements.append(Paragraph("Case Facts", section_style))
            elements.append(Paragraph(req.facts, body_style))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))

        for idx, arg in enumerate(req.arguments):
            elements.append(Paragraph(
                f"Issue {idx+1} of {len(req.arguments)}  ·  {arg.area_of_law}  ·  {arg.priority.upper()} PRIORITY",
                label_style
            ))
            elements.append(Paragraph(arg.issue_title, section_style))

            irac = arg.irac
            for letter, label, content in [
                ("I", "ISSUE — Central Legal Question",       irac.get('issue', '—')),
                ("R", "RULE — Applicable Law & Precedents",   irac.get('rule', '—')),
                ("A", "APPLICATION — Applying Law to Facts",  irac.get('application', '—')),
                ("C", "CONCLUSION — Outcome & Remedy",        irac.get('conclusion', '—')),
            ]:
                irac_data = [[
                    Paragraph(letter, irac_letter_style),
                    [
                        Paragraph(label, label_style),
                        Paragraph(
                            (content or '—')
                                .replace('**', '')
                                .replace('[SOURCE ', '(Source ')
                                .replace(']', ')'),
                            body_style
                        )
                    ]
                ]]
                irac_table = Table(irac_data, colWidths=[1.2*cm, 13.8*cm])
                irac_table.setStyle(TableStyle([
                    ('BACKGROUND',  (0,0), (-1,-1), colors.HexColor('#f9f7f2')),
                    ('OUTLINE',     (0,0), (-1,-1), 0.5, colors.HexColor('#ddd8cc')),
                    ('PADDING',     (0,0), (-1,-1), 8),
                    ('VALIGN',      (0,0), (-1,-1), 'TOP'),
                    ('TOPPADDING',  (0,0), (0,-1), 14),
                    ('ROUNDEDCORNERS', [4]),
                ]))
                elements.append(irac_table)
                elements.append(Spacer(1, 0.3*cm))

            # Cases used
            if arg.precedents:
                cases_str = "  ·  ".join([p.get('case_name','') for p in arg.precedents])
                elements.append(Paragraph(f"Cases Referenced: {cases_str}", meta_style))

            elements.append(Spacer(1, 0.5*cm))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))

    # ═══════════════════════════════════════════
    # OPPOSITION REPORT
    # ═══════════════════════════════════════════
    elif req.report_type == "opposition":
        if req.argument:
            elements.append(Paragraph("Argument Analysed", section_style))
            elements.append(Paragraph(req.argument, body_style))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))

        elements.append(Paragraph(
            f"Risk Level: {req.risk_level or 'MODERATE'}",
            ParagraphStyle('risk', fontName='Helvetica-Bold', fontSize=14,
                           textColor=RED if req.risk_level == 'HIGH' else GOLD)
        ))
        elements.append(Spacer(1, 0.3*cm))

        if req.weaknesses:
            elements.append(Paragraph("Identified Weaknesses", section_style))
            for w in req.weaknesses:
                elements.append(Paragraph(
                    f"{w.get('id','W')}  [{w.get('severity','MODERATE')} RISK]",
                    label_style
                ))
                elements.append(Paragraph(w.get('description', ''), body_style))

        if req.counter_args:
            elements.append(Paragraph("Defense Counter-Arguments", section_style))
            for c in req.counter_args:
                elements.append(Paragraph(f"• {c.get('point', '')}", body_style))
                if c.get('source'):
                    elements.append(Paragraph(f"  Citation: {c['source']}", meta_style))

        if req.strategy:
            elements.append(Paragraph("Strategy Recommendations", section_style))
            for s in req.strategy:
                icon  = "✓" if s.get('type') == 'DO' else "⚠"
                color = '#16a34a' if s.get('type') == 'DO' else '#c0392b'
                elements.append(Paragraph(
                    f'<font color="{color}"><b>{icon} {s.get("type", "DO")}</b></font>  {s.get("advice","")}',
                    body_style
                ))

    # ═══════════════════════════════════════════
    # DEBATE REPORT
    # ═══════════════════════════════════════════
    elif req.report_type == "debate":
        if req.round1:
            elements.append(Paragraph("Round 1 — Opening Arguments", section_style))
            debate_data = [
                [
                    Paragraph('<b>⚖ PLAINTIFF</b>', ParagraphStyle('ph', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#3b6fd4'))),
                    Paragraph('<b>⚔ DEFENSE</b>',   ParagraphStyle('dh', fontName='Helvetica-Bold', fontSize=10, textColor=RED))
                ]
            ]
            p_args = req.round1.get('plaintiff', [])
            d_args = req.round1.get('defense',   [])
            for i in range(max(len(p_args), len(d_args))):
                p_text = p_args[i].get('point', '') if i < len(p_args) else ''
                d_text = d_args[i].get('point', '') if i < len(d_args) else ''
                debate_data.append([
                    Paragraph(p_text, body_style),
                    Paragraph(d_text, body_style)
                ])
            debate_table = Table(debate_data, colWidths=[7.5*cm, 7.5*cm])
            debate_table.setStyle(TableStyle([
                ('BACKGROUND',   (0,0), (0,-1), colors.HexColor('#f0f4ff')),
                ('BACKGROUND',   (1,0), (1,-1), colors.HexColor('#fff4f4')),
                ('OUTLINE',      (0,0), (-1,-1), 0.5, colors.HexColor('#ddd8cc')),
                ('INNERGRID',    (0,0), (-1,-1), 0.3, colors.HexColor('#e0ddd8')),
                ('PADDING',      (0,0), (-1,-1), 8),
                ('VALIGN',       (0,0), (-1,-1), 'TOP'),
            ]))
            elements.append(debate_table)
            elements.append(Spacer(1, 0.5*cm))

        if req.round2:
            elements.append(Paragraph("Round 2 — Rebuttals", section_style))
            p_reb = req.round2.get('plaintiff', [])
            d_reb = req.round2.get('defense',   [])
            reb_data = [
                [
                    Paragraph('<b>⚖ PLAINTIFF REBUTTAL</b>', ParagraphStyle('pr', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#3b6fd4'))),
                    Paragraph('<b>⚔ DEFENSE REBUTTAL</b>',   ParagraphStyle('dr', fontName='Helvetica-Bold', fontSize=10, textColor=RED))
                ]
            ]
            for i in range(max(len(p_reb), len(d_reb))):
                p_text = p_reb[i].get('point', '') if i < len(p_reb) else ''
                d_text = d_reb[i].get('point', '') if i < len(d_reb) else ''
                reb_data.append([Paragraph(p_text, body_style), Paragraph(d_text, body_style)])
            reb_table = Table(reb_data, colWidths=[7.5*cm, 7.5*cm])
            reb_table.setStyle(TableStyle([
                ('BACKGROUND',  (0,0), (0,-1), colors.HexColor('#f0f4ff')),
                ('BACKGROUND',  (1,0), (1,-1), colors.HexColor('#fff4f4')),
                ('OUTLINE',     (0,0), (-1,-1), 0.5, colors.HexColor('#ddd8cc')),
                ('INNERGRID',   (0,0), (-1,-1), 0.3, colors.HexColor('#e0ddd8')),
                ('PADDING',     (0,0), (-1,-1), 8),
                ('VALIGN',      (0,0), (-1,-1), 'TOP'),
            ]))
            elements.append(reb_table)
            elements.append(Spacer(1, 0.5*cm))

        if req.summary:
            elements.append(HRFlowable(width="100%", thickness=2, color=GOLD, spaceAfter=8))
            elements.append(Paragraph("Final Strategy Summary", section_style))
            sm = req.summary
            for label, key in [
                ("Overall Assessment",       "overall_assessment"),
                ("Likely Outcome",           "likely_outcome"),
                ("Strategic Recommendation", "strategic_recommendation"),
            ]:
                if sm.get(key):
                    elements.append(Paragraph(f"<b>{label}</b>", label_style))
                    elements.append(Paragraph(sm[key], body_style))

    # ── FOOTER ────────────────────────────────
    elements.append(Spacer(1, 1*cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=GOLD))
    elements.append(Spacer(1, 0.2*cm))
    elements.append(Paragraph(
        'Generated by <b>LexForge</b>  ·  AI-Powered Indian Legal Research  ·  '
        'This report is for research purposes only and does not constitute legal advice.',
        ParagraphStyle('footer', fontName='Helvetica', fontSize=7, textColor=MUTED, alignment=1)
    ))

    doc.build(elements)
    return buffer.getvalue()


# ═════════════════════════════════════════════
# ROUTE — EXPORT REPORT
# POST /api/export/report
# ═════════════════════════════════════════════
@router.post("/api/export/report")
def export_report(req: ExportRequest):
    """
    Generate a professional PDF report from LexForge output.
    Returns the PDF as a downloadable file stream.
    """
    try:
        pdf_bytes = build_pdf(req)
        filename  = f"LexForge_{req.report_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length":      str(len(pdf_bytes))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")