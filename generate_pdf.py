"""
generate_pdf.py
---------------
Convertit rapport_ML_Projet.md en PDF avec markdown-pdf (pure Python).

Usage : python generate_pdf.py
Sortie : rapport_ML_Projet.pdf
"""

from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

ROOT = Path(__file__).parent
md_path  = ROOT / "rapport_ML_Projet.md"
pdf_path = ROOT / "rapport_ML_Projet.pdf"

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

pdf = MarkdownPdf(toc_level=0, optimize=True)
pdf.meta["title"] = "Eco-Smart Classifier — Rapport ML"
pdf.meta["author"] = "Bahloul Fares · Ben Rhaiem Mohamed"
pdf.meta["subject"] = "Pipeline ML Multimodal pour la classification de déchets"
pdf.meta["keywords"] = "ML, classification, régression, NLP, FastAPI, MLflow, DVC"

# La table des matières interne du markdown utilise des liens vers les sections.
# markdown-pdf génère un PDF par Section, ce qui casse les liens inter-sections.
# Solution : retirer les liens markdown de la TOC interne (les laisser comme texte simple).
import re
text = re.sub(r"\[([^\]]+)\]\(#[^)]+\)", r"\1", text)

css = """
body { font-family: 'Helvetica', sans-serif; line-height: 1.5; color: #1a1a1a; font-size: 10pt; }
h1   { color: #246124; border-bottom: 2px solid #2f7a2f; padding-bottom: 6px; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h2   { color: #2f7a2f; margin-top: 20px; border-bottom: 1px solid #ddd; padding-bottom: 3px; }
h3   { color: #444; margin-top: 14px; }
code { background: #f4f4f0; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre  { background: #f4f4f0; padding: 8px; border-radius: 4px; overflow-x: auto; font-size: 9pt; }
table { border-collapse: collapse; margin: 8px 0; font-size: 9pt; }
th, td { border: 1px solid #ddd; padding: 4px 8px; }
th { background: #f0f0eb; }
img { max-width: 100%; border: 1px solid #e2e2dc; border-radius: 4px; margin: 8px 0; }
blockquote { border-left: 3px solid #2f7a2f; padding-left: 10px; color: #555; font-style: italic; }
hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
"""

# Découpe sur les ---  pour faire des Sections (page-break entre chaque)
parts = [p.strip() for p in text.split("\n---\n") if p.strip()]
print(f"Sections détectées : {len(parts)}")

for i, part in enumerate(parts):
    pdf.add_section(Section(part, paper_size="A4"), user_css=css)
    print(f"  + Section {i+1}/{len(parts)} ({len(part)} chars)")

pdf.save(str(pdf_path))
size_kb = pdf_path.stat().st_size / 1024
print(f"\n✓ {pdf_path.name} ({size_kb:.0f} KB)")
