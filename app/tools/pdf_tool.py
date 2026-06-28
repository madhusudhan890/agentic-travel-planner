"""
app/tools/pdf_tool.py
──────────────────────
PDF generation tool using ReportLab (pure Python, no API needed).

WHY PDF GENERATION:
  Professional deliverables matter. A beautifully formatted PDF
  is what separates a demo from a real product. Users want to
  print their itinerary or share it with travel companions.

  Alternative: WeasyPrint (HTML → PDF, easier for complex layouts)
  Alternative: Pandoc (Markdown → PDF, but requires system installation)
  We use ReportLab because: no system dependencies, pure Python, free.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool


@tool
def generate_pdf_itinerary(
    itinerary_markdown: str,
    destination: str,
    session_id: str,
    output_dir: str = "./data/pdfs",
) -> str:
    """Generate a beautifully formatted PDF from a travel itinerary.

    Converts the markdown itinerary into a downloadable PDF document.
    Use this as the final step after the itinerary has been reviewed and approved.

    Args:
        itinerary_markdown: Full itinerary in markdown format
        destination: Destination name for the PDF filename
        session_id: Session ID for unique filename
        output_dir: Directory to save the PDF

    Returns:
        Path to the generated PDF file or error message.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.platypus import Table, TableStyle

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        safe_dest = re.sub(r"[^\w\-]", "_", destination)[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"itinerary_{safe_dest}_{timestamp[:8]}_{session_id[:8]}.pdf"
        filepath = output_path / filename

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        story = []

        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=22,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=12,
        )
        h1_style = ParagraphStyle(
            "CustomH1",
            parent=styles["Heading1"],
            fontSize=16,
            textColor=colors.HexColor("#16213e"),
            spaceBefore=16,
            spaceAfter=8,
        )
        h2_style = ParagraphStyle(
            "CustomH2",
            parent=styles["Heading2"],
            fontSize=13,
            textColor=colors.HexColor("#0f3460"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "CustomBody",
            parent=styles["Normal"],
            fontSize=10,
            leading=16,
            textColor=colors.HexColor("#2d2d2d"),
            spaceAfter=6,
        )

        # Header
        story.append(Paragraph(f"🌍 Travel Itinerary", title_style))
        story.append(Paragraph(destination, h1_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')} | Powered by AI Travel Planner Enterprise",
            body_style,
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0f3460")))
        story.append(Spacer(1, 0.5 * cm))

        # Parse and render markdown
        lines = itinerary_markdown.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 0.2 * cm))
                continue

            # Clean markdown for ReportLab (escape XML special chars)
            clean = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Bold: **text**
            clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", clean)
            # Italic: *text*
            clean = re.sub(r"\*(.+?)\*", r"<i>\1</i>", clean)

            if stripped.startswith("# "):
                story.append(Paragraph(clean[2:], title_style))
            elif stripped.startswith("## "):
                story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
                story.append(Paragraph(clean[3:], h1_style))
            elif stripped.startswith("### "):
                story.append(Paragraph(clean[4:], h2_style))
            elif stripped.startswith("- ") or stripped.startswith("* "):
                story.append(Paragraph(f"&bull; {clean[2:]}", body_style))
            elif stripped.startswith("|"):
                # Skip markdown table rows (simplified)
                story.append(Paragraph(clean, body_style))
            else:
                story.append(Paragraph(clean, body_style))

        # Footer
        story.append(Spacer(1, 1 * cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e0e0e0")))
        story.append(Paragraph(
            "Generated by AI Travel Planner Enterprise — Verify all prices and requirements before booking.",
            ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey),
        ))

        doc.build(story)
        return f"✅ PDF generated: {filepath}\nFile size: {filepath.stat().st_size // 1024}KB"

    except ImportError:
        return "⚠️ PDF generation requires reportlab: run 'uv add reportlab' to install."
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return f"⚠️ PDF generation failed: {exc}"
