import io
import sqlite3
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


def build_daily_report_pdf(db_path: str, date_str: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Heading1'], alignment=TA_CENTER, textColor=colors.HexColor('#1a1a2e'))
    elements = []

    elements.append(Paragraph("Industrial Monitoring Report", title))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Date: <b>{date_str}</b>", styles['Normal']))
    elements.append(Paragraph(f"Generated: <b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</b>", styles['Normal']))
    elements.append(Spacer(1, 18))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*), AVG(reading), MAX(reading), MIN(reading), SUM(is_anomaly) FROM dial_readings WHERE DATE(timestamp) = ?", (date_str,))
    total, avg, mx, mn, anoms = cur.fetchone()
    total = total or 0; avg = avg or 0; mx = mx or 0; mn = mn or 0; anoms = anoms or 0

    stats = [["Metric", "Value"], ["Total Readings", total], ["Average Reading", f"{avg:.2f}"],
             ["Max Reading", f"{mx:.2f}"], ["Min Reading", f"{mn:.2f}"], ["Anomalies", anoms],
             ["Anomaly Rate", f"{(anoms/total*100):.1f}%" if total else "0.0%"]]
    t = Table(stats, colWidths=[200, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.8, colors.black),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
    ]))
    elements.append(Paragraph("<b>Daily Statistics</b>", styles['Heading2']))
    elements.append(Spacer(1, 6))
    elements.append(t)
    elements.append(PageBreak())

    cur.execute("SELECT camera_id, COUNT(*), AVG(reading), MAX(reading), MIN(reading), SUM(is_anomaly) FROM dial_readings WHERE DATE(timestamp) = ? GROUP BY camera_id",
                (date_str,))
    rows = cur.fetchall()
    data = [["Camera", "Total", "Avg", "Max", "Min", "Anomalies"]]
    for r in rows:
        data.append([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}", r[5]])
    if len(data) > 1:
        t2 = Table(data)
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.8, colors.black),
        ]))
        elements.append(Paragraph("<b>Camera Performance</b>", styles['Heading2']))
        elements.append(Spacer(1, 6))
        elements.append(t2)

    conn.close()
    doc.build(elements)
    return buf.getvalue()