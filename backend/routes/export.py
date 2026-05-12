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
from xml.sax.saxutils import escape

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

from routes.export_pdf_assets import register_lexforge_pdf_fonts, FONT_BODY, FONT_BOLD

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

    # PDF headings/footer: "en" or "hi" (body text always preserves user language; Devanagari needs Nirmala/Noto)
    document_locale: Optional[str] = "en"


def _xr(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = escape(str(text).strip('\ufeff')).replace('\r\n', '\n').replace('\r', '\n')
    return s.replace('\n', '<br/>')


def _P(raw: Optional[str], style) -> Paragraph:
    if raw is None or str(raw).strip() == '':
        return Paragraph('—', style)
    return Paragraph(_xr(str(raw)), style)


def _labels(locale: Optional[str]) -> dict:
    if (locale or "en").lower().startswith("hi"):
        return {
            "research": "अनुसंधान विवरण · Research memorandum",
            "argument": "लिखित दलीलें · Written submissions (IRAC)",
            "opposition": "विरोध विश्लेषण · Opposition analysis",
            "debate": "अदालती सिमुलेशन · Hearing simulation",
            "legal_query": "कानूनी प्रश्न",
            "synthesized": "संश्लेषित उत्तर",
            "precedents": "उद्धृत निर्णय",
            "case_facts": "प्रकरण के तथ्य",
            "issue": "मुद्दा",
            "argument_analysed": "विश्लेषित दलील",
            "risk": "जोखिम स्तर",
            "weaknesses": "कमजोरियाँ",
            "counters": "प्रतिदलीलें",
            "strategy": "रणनीति",
            "r1": "पहला दौर — प्रारंभिक दलीलें",
            "r2": "दूसरा दौर — प्रत्युत्तर",
            "summary": "अंतिम निष्कर्ष",
            "plaintiff": "याचिकाकर्ता",
            "defense": "प्रतिवादी",
            "footer": (
                "यह रिपोर्ट केवल अनुसंधान हेतु है — कानूनी सलाह नहीं। "
                "LexForge द्वारा उत्पन्न।"
            ),
            "jurisdiction": "क्षेत्राधिकार",
            "cite": "उद्धरण",
            "sm_assess": "कुल मूल्यांकन",
            "sm_outcome": "संभावित परिणाम",
            "sm_strat": "रणनीतिक सुझाव",
            "reb_p": "प्रत्युत्तर",
            "reb_d": "पुनः प्रत्युत्तर",
        }
    return {
        "research": "Research memorandum",
        "argument": "Written submissions (IRAC)",
        "opposition": "Opposition analysis",
        "debate": "Hearing simulation (moot court transcript)",
        "legal_query": "Legal query",
        "synthesized": "Synthesized answer",
        "precedents": "Retrieved precedents",
        "case_facts": "Case facts",
        "issue": "Issue",
        "argument_analysed": "Argument analysed",
        "risk": "Risk level",
        "weaknesses": "Identified weaknesses",
        "counters": "Counter-arguments",
        "strategy": "Strategy recommendations",
        "r1": "Round 1 — Opening arguments",
        "r2": "Round 2 — Rebuttals",
        "summary": "Judicial assessment & strategy",
        "plaintiff": "Petitioner / Plaintiff",
        "defense": "Respondent / Defense",
        "footer": (
            "Generated by LexForge — AI-assisted legal research. "
            "For research only; not legal advice."
        ),
        "jurisdiction": "Jurisdiction",
        "cite": "Citation",
        "sm_assess": "Overall assessment",
        "sm_outcome": "Likely outcome",
        "sm_strat": "Strategic recommendation",
        "reb_p": "Rebuttal",
        "reb_d": "Sur-rebuttal",
    }


# ═════════════════════════════════════════════
# PDF BUILDER
# ═════════════════════════════════════════════
def build_pdf(req: ExportRequest) -> bytes:
    register_lexforge_pdf_fonts()
    loc = (req.document_locale or "en").lower()
    if not loc.startswith("hi"):
        loc = "en"
    L = _labels(loc)

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

    # ── Custom styles (Unicode-capable when system/Noto fonts register) ─────────
    title_style = ParagraphStyle(
        'LexTitle',
        parent    = styles['Title'],
        fontName  = FONT_BOLD,
        fontSize  = 20,
        textColor = NAVY,
        spaceAfter= 6,
    )
    subtitle_style = ParagraphStyle(
        'LexSub',
        parent    = styles['Normal'],
        fontName  = FONT_BODY,
        fontSize  = 10,
        textColor = MUTED,
        spaceAfter= 4,
    )
    section_style = ParagraphStyle(
        'LexSection',
        parent    = styles['Heading2'],
        fontName  = FONT_BOLD,
        fontSize  = 13,
        textColor = NAVY,
        spaceBefore=14,
        spaceAfter= 6,
    )
    body_style = ParagraphStyle(
        'LexBody',
        parent    = styles['Normal'],
        fontName  = FONT_BODY,
        fontSize  = 10,
        textColor = colors.HexColor('#1a1a2e'),
        leading   = 16,
        spaceAfter= 8,
    )
    label_style = ParagraphStyle(
        'LexLabel',
        parent    = styles['Normal'],
        fontName  = FONT_BOLD,
        fontSize  = 9,
        textColor = GOLD,
        spaceAfter= 2,
    )
    irac_letter_style = ParagraphStyle(
        'IracLetter',
        parent    = styles['Normal'],
        fontName  = FONT_BOLD,
        fontSize  = 22,
        textColor = GOLD,
        leading   = 26,
    )
    case_style = ParagraphStyle(
        'LexCase',
        parent    = styles['Normal'],
        fontName  = FONT_BOLD,
        fontSize  = 10,
        textColor = NAVY,
        spaceAfter= 2,
    )
    meta_style = ParagraphStyle(
        'LexMeta',
        parent    = styles['Normal'],
        fontName  = FONT_BODY,
        fontSize  = 8,
        textColor = MUTED,
        spaceAfter= 3,
    )
    snippet_style = ParagraphStyle(
        'LexSnippet',
        parent    = styles['Normal'],
        fontName  = FONT_BODY,
        fontSize  = 9,
        textColor = colors.HexColor('#4a4a6a'),
        leading   = 14,
        spaceAfter= 6,
    )

    elements = []

    type_line = L.get(req.report_type, L["research"])
    jur_line  = f'{escape(L["jurisdiction"])}: {escape(str(req.jurisdiction or ""))} · LexForge · {escape(type_line)}'

    # ── HEADER ────────────────────────────────
    header_data = [[
        Paragraph('<font color="#c9a84c"><b>LEX</b></font><font color="#0a0f1e">FORGE</font>', ParagraphStyle('brand', fontName=FONT_BOLD, fontSize=16)),
        Paragraph(f'<font color="#6b7a99">{escape(datetime.now().strftime("%d %B %Y"))}</font>', ParagraphStyle('date', fontName=FONT_BODY, fontSize=9, alignment=2))
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

    elements.append(_P(req.title, title_style))
    elements.append(Paragraph(f'<font color="#6b7a99">{jur_line}</font>', subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=2, color=GOLD, spaceAfter=12))

    # ═══════════════════════════════════════════
    # RESEARCH REPORT
    # ═══════════════════════════════════════════
    if req.report_type == "research" and req.answer:
        elements.append(Paragraph(_xr(L["legal_query"]), section_style))
        elements.append(_P(req.query or "—", body_style))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))

        elements.append(Paragraph(_xr(L["synthesized"]), section_style))
        clean_answer = (req.answer
            .replace('**', '')
            .replace('##', '')
            .replace('[SOURCE ', '(Source ')
            .replace(']', ')'))
        for para in clean_answer.split('\n\n'):
            if para.strip():
                elements.append(_P(para.strip(), body_style))

        if req.precedents:
            elements.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))
            elements.append(Paragraph(_xr(L["precedents"]), section_style))
            for p in req.precedents:
                elements.append(_P(str(p.case_name), case_style))
                elements.append(Paragraph(_xr(f"{p.court}  ·  {p.year}  ·  Relevance: {p.score}  ·  {p.binding}"), meta_style))
                if p.snippet:
                    sn = str(p.snippet)
                    snippet_txt = '"' + sn[:280] + ('…"' if len(sn) > 280 else '"')
                    elements.append(_P(snippet_txt, snippet_style))
                elements.append(Spacer(1, 0.2*cm))

    # ═══════════════════════════════════════════
    # ARGUMENT BUILDER REPORT
    # ═══════════════════════════════════════════
    elif req.report_type == "argument" and req.arguments:
        if req.facts:
            elements.append(Paragraph(_xr(L["case_facts"]), section_style))
            elements.append(_P(req.facts, body_style))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))

        for idx, arg in enumerate(req.arguments):
            elements.append(Paragraph(_xr(f"Issue {idx+1} of {len(req.arguments)}  ·  {arg.area_of_law}  ·  {arg.priority.upper()} PRIORITY"), label_style))
            elements.append(_P(arg.issue_title, section_style))

            irac = arg.irac or {}
            for letter, label, content in [
                ("I", "ISSUE — Central Legal Question",       irac.get('issue', '—')),
                ("R", "RULE — Applicable Law & Precedents",   irac.get('rule', '—')),
                ("A", "APPLICATION — Applying Law to Facts",  irac.get('application', '—')),
                ("C", "CONCLUSION — Outcome & Remedy",        irac.get('conclusion', '—')),
            ]:
                body_cell = _P(
                    str(content or '—').replace('**', '').replace('[SOURCE ', '(Source ').replace(']', ')'),
                    body_style)
                irac_data = [[
                    Paragraph(_xr(letter), irac_letter_style),
                    [Paragraph(_xr(label), label_style), body_cell],
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
                cases_str = "  ·  ".join(str(p.get('case_name', '')) for p in arg.precedents)
                elements.append(_P(f"Cases referenced: {cases_str}", meta_style))

            elements.append(Spacer(1, 0.5*cm))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))

    # ═══════════════════════════════════════════
    # OPPOSITION REPORT
    # ═══════════════════════════════════════════
    elif req.report_type == "opposition":
        if req.argument:
            elements.append(Paragraph(_xr(L["argument_analysed"]), section_style))
            elements.append(_P(req.argument, body_style))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))

        risk_txt = f'{L["risk"]}: {escape(str(req.risk_level or "MODERATE"))}'
        elements.append(Paragraph(
            risk_txt,
            ParagraphStyle('risk', fontName=FONT_BOLD, fontSize=14,
                           textColor=RED if req.risk_level == 'HIGH' else GOLD)
        ))
        elements.append(Spacer(1, 0.3*cm))

        if req.weaknesses:
            elements.append(Paragraph(_xr(L["weaknesses"]), section_style))
            for w in req.weaknesses:
                wid = escape(str(w.get('id', 'W')))
                sev = escape(str(w.get('severity', 'MODERATE')))
                elements.append(Paragraph(_xr(f"{wid}  [{sev} RISK]"), label_style))
                elements.append(_P(w.get('description', ''), body_style))

        if req.counter_args:
            elements.append(Paragraph(_xr(L["counters"]), section_style))
            for c in req.counter_args:
                pt = str(c.get('point', '') or '')
                cit = str(c.get('source') or c.get('authority') or '')
                elements.append(_P('• ' + pt, body_style))
                if cit.strip():
                    elements.append(_P(f"   {L['cite']}: {cit}", meta_style))

        if req.strategy:
            elements.append(Paragraph(_xr(L["strategy"]), section_style))
            for s in req.strategy:
                icon  = "✓ " if s.get('type') == 'DO' else "⚠ "
                color = '#16a34a' if s.get('type') == 'DO' else '#c0392b'
                st = escape(str(s.get('type', 'DO')))
                ad = escape(str(s.get('advice', '') or ''))
                elements.append(Paragraph(
                    f'<font color="{color}"><b>{icon}{st}</b></font> {ad}',
                    body_style
                ))

    elif req.report_type == "debate":
        ph = ParagraphStyle('ph', fontName=FONT_BOLD, fontSize=10, textColor=colors.HexColor('#2563eb'))
        dh = ParagraphStyle('dh', fontName=FONT_BOLD, fontSize=10, textColor=RED)

        if req.round1:
            elements.append(Paragraph(_xr(L["r1"]), section_style))
            debate_data = [[
                Paragraph(f'<b>{escape(L["plaintiff"])}</b>', ph),
                Paragraph(f'<b>{escape(L["defense"])}</b>', dh),
            ]]
            p_args = req.round1.get('plaintiff', [])
            d_args = req.round1.get('defense',   [])
            for i in range(max(len(p_args), len(d_args))):
                p_text = p_args[i].get('point', '') if i < len(p_args) else ''
                d_text = d_args[i].get('point', '') if i < len(d_args) else ''
                debate_data.append([
                    _P(str(p_text), body_style),
                    _P(str(d_text), body_style),
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
            elements.append(Paragraph(_xr(L["r2"]), section_style))
            p_reb = req.round2.get('plaintiff', [])
            d_reb = req.round2.get('defense',   [])
            pr = ParagraphStyle('pr', fontName=FONT_BOLD, fontSize=10, textColor=colors.HexColor('#2563eb'))
            dr = ParagraphStyle('dr', fontName=FONT_BOLD, fontSize=10, textColor=RED)
            reb_data = [[
                Paragraph(f'<b>{escape(L["plaintiff"])} — {escape(L["reb_p"])}</b>', pr),
                Paragraph(f'<b>{escape(L["defense"])} — {escape(L["reb_d"])}</b>', dr),
            ]]
            for i in range(max(len(p_reb), len(d_reb))):
                p_text = p_reb[i].get('point', '') if i < len(p_reb) else ''
                d_text = d_reb[i].get('point', '') if i < len(d_reb) else ''
                reb_data.append([_P(str(p_text), body_style), _P(str(d_text), body_style)])
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
            elements.append(Paragraph(_xr(L["summary"]), section_style))
            sm = req.summary
            triple = [
                (L["sm_assess"], "overall_assessment"),
                (L["sm_outcome"], "likely_outcome"),
                (L["sm_strat"], "strategic_recommendation"),
            ]
            for slabel, key in triple:
                if sm.get(key):
                    elements.append(Paragraph(f"<b>{escape(slabel)}</b>", label_style))
                    elements.append(_P(str(sm[key]), body_style))

    elements.append(Spacer(1, 1*cm))
    elements.append(HRFlowable(width="100%", thickness=1, color=GOLD))
    elements.append(Spacer(1, 0.2*cm))
    footer_style = ParagraphStyle(
        'LexFooter',
        parent=styles['Normal'],
        fontName=FONT_BODY,
        fontSize=7,
        textColor=MUTED,
        alignment=1,
        leading=11,
    )
    elements.append(_P(L["footer"], footer_style))

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