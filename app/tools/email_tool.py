"""
app/tools/email_tool.py
────────────────────────
Email delivery tool using Python's smtplib (no external service needed).

WHY SMTPLIB OVER SENDGRID:
  smtplib is Python standard library — zero dependencies.
  Works with Gmail, Outlook, any SMTP server.
  For production at scale: SendGrid (100 emails/day free), Mailgun, AWS SES.

CONFIGURATION:
  Set in .env:
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your@gmail.com
    SMTP_PASSWORD=your_app_password  (Gmail: create App Password)
    SMTP_FROM=your@gmail.com

  Gmail App Password: myaccount.google.com → Security → App Passwords
"""

from __future__ import annotations

import asyncio
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from langchain_core.tools import tool


@tool
def send_itinerary_email(
    recipient_email: str,
    destination: str,
    itinerary_markdown: str,
    pdf_path: str = "",
    smtp_host: str = "",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_password: str = "",
    smtp_from: str = "",
) -> str:
    """Send the travel itinerary via email with optional PDF attachment.

    Use this as the final step after PDF generation, if the user provided an email.

    PRODUCTION: Replace smtplib with SendGrid API for:
    - Deliverability tracking
    - Open/click analytics
    - 100 emails/day free at sendgrid.com

    Args:
        recipient_email: Destination email address
        destination: Trip destination name (for subject line)
        itinerary_markdown: Full itinerary text
        pdf_path: Path to PDF attachment (optional)
        smtp_host, smtp_port, smtp_user, smtp_password, smtp_from: SMTP config

    Returns:
        Success message or error description.
    """
    if not smtp_host or not smtp_user:
        return (
            "⚠️ Email not configured. Set SMTP_HOST, SMTP_USER, SMTP_PASSWORD "
            "in .env file to enable email delivery.\n"
            "Guide: Use Gmail App Passwords or set up SendGrid (free tier)."
        )

    if not recipient_email or "@" not in recipient_email:
        return "⚠️ Invalid recipient email address."

    try:
        # Build email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🌍 Your AI Travel Itinerary: {destination}"
        msg["From"] = smtp_from or smtp_user
        msg["To"] = recipient_email

        # HTML body
        html_body = _markdown_to_html(itinerary_markdown, destination)
        msg.attach(MIMEText(itinerary_markdown, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        # PDF attachment
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as f:
                attach = MIMEApplication(f.read(), _subtype="pdf")
                attach.add_header(
                    "Content-Disposition", "attachment",
                    filename=f"itinerary_{destination.replace(' ', '_')[:20]}.pdf"
                )
                msg.attach(attach)

        # Send
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from or smtp_user, recipient_email, msg.as_string())

        attach_note = " with PDF attachment" if pdf_path else ""
        return f"✅ Email sent{attach_note} to {recipient_email}"

    except asyncio.CancelledError:
        raise
    except smtplib.SMTPAuthenticationError:
        return "⚠️ Email authentication failed. Check SMTP_USER and SMTP_PASSWORD in .env"
    except Exception as exc:
        return f"⚠️ Email send failed: {exc}"


def _markdown_to_html(markdown: str, destination: str) -> str:
    """Simple markdown to HTML conversion for email."""
    import re

    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; max-width: 700px; margin: 0 auto; padding: 20px; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }}
  h2 {{ color: #16213e; border-left: 4px solid #e94560; padding-left: 10px; }}
  h3 {{ color: #0f3460; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #0f3460; color: white; }}
  blockquote {{ background: #f0f4ff; border-left: 4px solid #0f3460; margin: 10px 0; padding: 10px 15px; }}
  .footer {{ font-size: 12px; color: #888; margin-top: 30px; border-top: 1px solid #eee; padding-top: 15px; }}
</style>
</head>
<body>
"""
    lines = markdown.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            html += "<br>"
        elif line.startswith("# "):
            html += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith("## "):
            html += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith("### "):
            html += f"<h3>{line[4:]}</h3>\n"
        elif line.startswith("> "):
            html += f"<blockquote>{line[2:]}</blockquote>\n"
        elif line.startswith("- ") or line.startswith("* "):
            html += f"<li>{line[2:]}</li>\n"
        else:
            line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
            line = re.sub(r"\*(.+?)\*", r"<i>\1</i>", line)
            html += f"<p>{line}</p>\n"

    html += """
<div class="footer">
  Generated by AI Travel Planner Enterprise — Verify all prices and visa requirements before booking.
</div>
</body></html>"""
    return html
