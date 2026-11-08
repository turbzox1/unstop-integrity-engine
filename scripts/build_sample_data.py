"""Build the synthetic evaluation corpus in ./sample_submissions.

All text is synthetic and written for this assessment (no real student data).
Requires: reportlab, python-docx, ffmpeg and espeak-ng (only for re-generating MP4s;
the generated files are committed so the pipeline can be run without these tools).

    python scripts/build_sample_data.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample_submissions"

# --------------------------------------------------------------------------------------
# Shared ~150-word competition prompt (Scenario 4: benign common boilerplate)
# --------------------------------------------------------------------------------------
PROMPT = (
    "Case Prompt: GreenCart Quick Commerce Expansion. GreenCart is a mid-sized grocery "
    "delivery company operating in three Indian metro cities. The founders are considering "
    "an expansion into ten tier-two cities over the next eighteen months. Participants must "
    "analyse the market opportunity, evaluate the unit economics of dark stores, identify the "
    "main operational risks, and recommend a go-to-market strategy. Answers should address "
    "customer acquisition cost, average order value, delivery time commitments, and supplier "
    "relationships. Teams must state all assumptions clearly and support every recommendation "
    "with quantitative reasoning. The submission should not exceed five pages and must include "
    "an executive summary, a market analysis, a financial assessment, and a risk mitigation plan. "
    "Cite all external sources using a consistent referencing style. Submissions will be judged "
    "on analytical depth, feasibility, clarity of communication, and originality of insight. "
    "Late submissions will not be evaluated by the panel."
)

# --------------------------------------------------------------------------------------
# Original analyses
# --------------------------------------------------------------------------------------
ORIG_A = [
    "Our team believes GreenCart should not enter all ten cities at once. Tier-two cities differ "
    "sharply in density, and a dark store only breaks even when it fulfils roughly nine hundred "
    "orders a day. We therefore propose a staggered rollout that begins with Indore and Coimbatore, "
    "where apartment clusters are dense and smartphone adoption is already high.",
    "The financial assessment rests on three numbers. We assume an average order value of four "
    "hundred and twenty rupees, a contribution margin of eighteen percent before delivery, and a "
    "delivery cost of thirty eight rupees per order. Under these assumptions a store needs fourteen "
    "months to recover its setup capital, which is acceptable only if customer churn stays below six "
    "percent per month.",
    "The largest operational risk is rider availability during festivals and monsoon weeks. We "
    "recommend a hybrid fleet where forty percent of riders are salaried and the rest are paid per "
    "drop, so that service levels hold when demand spikes. Supplier risk is lower because regional "
    "dairy cooperatives already run cold chains into these cities.",
]

ORIG_B = [
    "Rather than copying the ten minute promise used in metros, GreenCart should compete on "
    "reliability and price in smaller cities. Our customer interviews with twenty households in "
    "Nashik showed that families plan weekly purchases and value a predictable thirty minute window "
    "more than speed.",
    "We model a scheduled-slot service with larger baskets. If the average basket grows to six "
    "hundred rupees and orders are batched into routes of four drops, the delivery cost per order "
    "falls by almost half. This turns the unit economics positive within the first eight months.",
    "Marketing spend should flow through local kirana partnerships and resident welfare associations "
    "instead of paid digital ads, which are expensive relative to order value in these markets. The "
    "key risk is that a metro competitor undercuts prices, so GreenCart must lock in exclusive "
    "supply agreements with two or three regional brands early.",
]

ORIG_C = [
    "We frame the expansion decision as a portfolio of options rather than a single bet. Each city is "
    "scored on population density, digital payment penetration, rental costs for warehouse space, and "
    "the strength of existing local delivery players. Only cities scoring above the median receive a "
    "dark store in year one.",
    "Our sensitivity analysis shows that profitability is most sensitive to order frequency, not "
    "basket size. A household ordering three times a week is worth almost four times as much as a "
    "household ordering once, because fixed delivery and picking costs are spread across more value.",
    "We therefore recommend a subscription plan priced at ninety nine rupees per month that removes "
    "delivery fees. Retention, not acquisition, should be the north star metric for the first year, "
    "and city managers should be rewarded on repeat purchase rate.",
]

# Heavily formatted original PDF (tables, header/footer, many short lines)
ORIG_D = [
    "This report evaluates GreenCart through a supply chain lens. We map every step from farm gate "
    "to doorstep and estimate where spoilage and waiting time occur. In tier-two cities, produce often "
    "passes through two wholesale mandis, adding a day of delay and roughly seven percent wastage.",
    "Our recommendation is to source vegetables directly from farmer producer organisations within a "
    "hundred kilometre radius. Direct procurement shortens the chain, improves freshness scores, and "
    "gives GreenCart a story that local customers trust more than national brand advertising.",
]

# --------------------------------------------------------------------------------------
# Paraphrase pair (Scenario 2)
# --------------------------------------------------------------------------------------
PARA_SRC = [
    "Electric vehicle adoption in Indian cities is held back mainly by the lack of public charging "
    "infrastructure. Drivers worry that they will run out of battery before finding a charger, and "
    "this range anxiety outweighs the lower running cost of electric cars.",
    "Battery swapping offers a practical solution for two-wheelers and three-wheelers. Instead of "
    "waiting an hour to recharge, a rider exchanges a depleted battery for a full one in under two "
    "minutes, which keeps commercial vehicles on the road and earning money.",
    "Government subsidies should shift from the purchase price of vehicles toward building charging "
    "and swapping stations. Once infrastructure is visible in every neighbourhood, consumer confidence "
    "will rise and private demand will follow without permanent subsidies.",
]
PARA_DST = [
    "The main barrier stopping urban Indians from switching to electric cars is that there are too few "
    "public places to charge them. Motorists fear their battery will die before they reach a charging "
    "point, and this fear about driving distance matters more to them than the cheaper cost of running "
    "an electric car.",
    "For scooters and auto rickshaws, exchanging batteries is a workable answer. A driver can trade an "
    "empty battery for a charged one in less than two minutes rather than spending an hour plugged in, "
    "so delivery and taxi vehicles stay in service and keep generating income.",
    "State incentives ought to move away from lowering the sticker price of vehicles and instead fund "
    "the construction of charging and battery exchange points. When people see this infrastructure in "
    "every locality, their trust will grow and buyers will come forward on their own without the need "
    "for subsidies forever.",
]

# --------------------------------------------------------------------------------------
# Cross-modal pair (Scenario 1): paragraph on page 2 of the PDF is read aloud in the MP4
# --------------------------------------------------------------------------------------
CM_PAGE1 = [
    "Section one introduces the cold chain problem facing pharmaceutical distributors in rural "
    "districts. Vaccines and insulin must stay between two and eight degrees Celsius, yet many "
    "primary health centres lose grid power for several hours a day.",
    "We surveyed published reports from state health departments and found that temperature "
    "excursions are rarely logged, which makes it hard to know how much medicine is actually spoiled.",
]
CM_PAGE2_SPOKEN = (
    "Solar powered refrigerators with phase change batteries can keep vaccines cold for up to five "
    "days without electricity. Each unit costs about three thousand dollars, but it removes the need "
    "for diesel generators and cuts spoilage to almost zero. We recommend that every district deploy "
    "these units at the ten health centres with the worst power supply first."
)
CM_PAGE2_REST = [
    "A low cost temperature logger with a cellular connection should be attached to every unit, so "
    "that district officials receive an alert whenever the temperature leaves the safe range.",
]
PITCH1_INTRO = "Hello judges, our team worked on rural health logistics."

PITCH2_ORIGINAL = (
    "Hi everyone, this is our pitch for a campus food waste app. Canteens throw away a lot of cooked "
    "food every evening. Our app lets students reserve leftover meals at half price thirty minutes "
    "before closing, so the canteen earns money and less food goes to waste."
)

# --------------------------------------------------------------------------------------
# External / web-copied submission (Scenario: external match). Source text lives in
# reference_corpus/ (a simulated "web" knowledge base of synthetic encyclopedia-style articles).
# --------------------------------------------------------------------------------------
WEB_ARTICLE = (
    "Dark stores are small warehouses that are closed to the public and exist only to fulfil online "
    "orders. They are usually located in dense residential neighbourhoods so that riders can reach "
    "customers within a few kilometres. Inventory in a dark store is limited to a few thousand fast "
    "moving products, which allows pickers to assemble an order in under three minutes. The model "
    "became popular during the pandemic, when demand for contactless grocery delivery grew rapidly. "
    "Critics argue that dark stores create traffic congestion and noise in residential areas, and "
    "several city councils have introduced zoning rules that restrict where they can operate."
)
WEB_COPY_DOC = [
    "Background. " + WEB_ARTICLE,
    "In our view GreenCart can use this model in tier-two cities if rent stays low, but we did not "
    "complete a full financial model for this submission.",
]

# --------------------------------------------------------------------------------------
# LLM-generated submission (written by an LLM assistant for this test set)
# --------------------------------------------------------------------------------------
AI_TEXT = [
    "In today's rapidly evolving digital landscape, sustainable urban mobility has emerged as a "
    "critical priority for cities around the world. It is important to note that a comprehensive "
    "approach is essential to address the multifaceted challenges involved.",
    "Furthermore, leveraging innovative technologies such as data analytics and smart infrastructure "
    "can significantly enhance operational efficiency. Additionally, fostering collaboration between "
    "public and private stakeholders plays a pivotal role in driving meaningful change.",
    "Moreover, it is crucial to consider the diverse needs of all users. By prioritizing inclusivity "
    "and accessibility, cities can ensure that the benefits of sustainable mobility are shared "
    "equitably across communities. This holistic strategy not only reduces emissions but also "
    "improves overall quality of life.",
    "In conclusion, sustainable urban mobility represents a transformative opportunity. By embracing "
    "a strategic, forward-thinking approach, stakeholders can navigate the complexities of this "
    "landscape and unlock significant long-term value for society as a whole.",
]

REFERENCE_CORPUS = {
    "web_dark_stores.txt": ("https://example-encyclopedia.org/wiki/Dark_store (simulated)", WEB_ARTICLE),
    "web_battery_swapping.txt": (
        "https://example-encyclopedia.org/wiki/Battery_swapping (simulated)",
        "Battery swapping is a method of refuelling electric vehicles in which a discharged battery is "
        "removed and replaced with a fully charged one at a dedicated station. The approach is common "
        "for electric scooters in Asia, where operators run networks of swap kiosks. Supporters say it "
        "removes charging delays, while critics point to the need for standardised battery packs "
        "across manufacturers.",
    ),
    "web_vaccine_cold_chain.txt": (
        "https://example-encyclopedia.org/wiki/Vaccine_cold_chain (simulated)",
        "The vaccine cold chain is the system of storage and transport that keeps vaccines within a "
        "recommended temperature range from manufacture to administration. Most routine vaccines must "
        "be stored between two and eight degrees Celsius. Solar direct drive refrigerators are used in "
        "regions with unreliable electricity because they do not require batteries or grid power.",
    ),
    "web_food_waste.txt": (
        "https://example-encyclopedia.org/wiki/Food_waste (simulated)",
        "Food waste refers to food that is fit for consumption but is discarded at the retail or "
        "consumer level. Institutional kitchens such as hospitals, schools and university canteens "
        "contribute a significant share because meals are prepared in advance of uncertain demand.",
    ),
}


# --------------------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------------------
def write_pdf(path: Path, pages: list[list[str]], title: str, header_footer: bool = True,
              table: bool = False) -> None:
    styles = getSampleStyleSheet()

    def decorate(canvas, doc):
        if not header_footer:
            return
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 815, "Case Competition 2026 | Confidential Submission")
        canvas.drawString(40, 25, f"Page {doc.page}")
        canvas.restoreState()

    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for i, page in enumerate(pages):
        for para in page:
            story.append(Paragraph(para, styles["BodyText"]))
            story.append(Spacer(1, 6))
        if table and i == 0:
            story.append(Table([["Metric", "Value"], ["Wastage", "7%"], ["Mandis", "2"],
                                ["Radius", "100 km"]]))
        if i < len(pages) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4).build(story, onFirstPage=decorate,
                                                    onLaterPages=decorate)


def write_docx(path: Path, paragraphs: list[str], title: str) -> None:
    doc = Document()
    doc.add_heading(title, level=1)
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))


def write_mp4(path: Path, text: str | None, seconds: int = 12) -> None:
    """Text-to-speech (espeak-ng) over a solid background. text=None -> silent video."""
    with tempfile.TemporaryDirectory() as tmp:
        if text is None:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            f"color=c=navy:s=320x240:d={seconds}", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", str(path)], check=True)
            return
        wav = Path(tmp) / "speech.wav"
        subprocess.run(["espeak-ng", "-s", "140", "-v", "mb-us2", "-w", str(wav), text], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=darkgreen:s=320x240", "-i", str(wav), "-shortest",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
                       check=True)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    ref_dir = ROOT / "reference_corpus"
    ref_dir.mkdir(exist_ok=True)
    for name, (url, text) in REFERENCE_CORPUS.items():
        (ref_dir / name).write_text(f"SOURCE_URL: {url}\n\n{text}\n", encoding="utf-8")

    write_pdf(OUT / "sub_01_original_a.pdf", [[PROMPT], ORIG_A], "GreenCart Expansion - Team Alpha")
    write_docx(OUT / "sub_02_exact_copy_of_01.docx", [PROMPT] + ORIG_A, "GreenCart - Team Beta")
    write_pdf(OUT / "sub_03_original_b.pdf", [[PROMPT], ORIG_B], "GreenCart Expansion - Team Gamma")
    write_docx(OUT / "sub_04_ev_original.docx", PARA_SRC, "EV Adoption in Indian Cities")
    write_docx(OUT / "sub_05_ev_paraphrase.docx", PARA_DST, "Why Urban India Is Slow to Go Electric")
    write_pdf(OUT / "sub_06_cold_chain.pdf", [CM_PAGE1, [CM_PAGE2_SPOKEN] + CM_PAGE2_REST],
              "Rural Vaccine Cold Chain")
    write_docx(OUT / "sub_07_ai_generated.docx", AI_TEXT, "Sustainable Urban Mobility")
    write_pdf(OUT / "sub_08_web_copy.pdf", [WEB_COPY_DOC], "Dark Store Note - Team Delta")
    write_docx(OUT / "sub_09_original_c.docx", [PROMPT] + ORIG_C, "GreenCart - Team Epsilon")
    write_pdf(OUT / "sub_10_heavy_format.pdf", [[PROMPT], ORIG_D], "GreenCart Supply Chain - Team Zeta",
              table=True)
    (OUT / "sub_11_empty.pdf").write_bytes(b"")
    write_mp4(OUT / "pitch_01_reads_sub06.mp4", f"{PITCH1_INTRO} {CM_PAGE2_SPOKEN}")
    write_mp4(OUT / "pitch_02_original.mp4", PITCH2_ORIGINAL)
    write_mp4(OUT / "pitch_03_silent.mp4", None, seconds=10)

    manifest = {
        "description": "Synthetic ground truth for the integrity engine. 'pairs' lists document "
                       "pairs that SHOULD be flagged; every other pair should not be flagged.",
        "files": {
            "sub_01_original_a.pdf": {"label": "original", "contains_template": True},
            "sub_02_exact_copy_of_01.docx": {"label": "exact_copy", "source": "sub_01_original_a.pdf",
                                             "contains_template": True},
            "sub_03_original_b.pdf": {"label": "original", "contains_template": True},
            "sub_04_ev_original.docx": {"label": "original"},
            "sub_05_ev_paraphrase.docx": {"label": "paraphrase", "source": "sub_04_ev_original.docx"},
            "sub_06_cold_chain.pdf": {"label": "original"},
            "sub_07_ai_generated.docx": {"label": "ai_generated"},
            "sub_08_web_copy.pdf": {"label": "external_copy",
                                    "external_source": "web_dark_stores.txt"},
            "sub_09_original_c.docx": {"label": "original", "contains_template": True},
            "sub_10_heavy_format.pdf": {"label": "original", "contains_template": True,
                                        "note": "header/footer on every page + table"},
            "sub_11_empty.pdf": {"label": "corrupted", "note": "zero-byte file"},
            "pitch_01_reads_sub06.mp4": {"label": "cross_modal_copy", "source": "sub_06_cold_chain.pdf",
                                         "note": "reads page-2 paragraph aloud after a short intro"},
            "pitch_02_original.mp4": {"label": "original"},
            "pitch_03_silent.mp4": {"label": "corrupted", "note": "no audio track"},
        },
        "pairs": [
            {"a": "sub_01_original_a.pdf", "b": "sub_02_exact_copy_of_01.docx", "type": "exact_copy"},
            {"a": "sub_04_ev_original.docx", "b": "sub_05_ev_paraphrase.docx", "type": "paraphrase"},
            {"a": "sub_06_cold_chain.pdf", "b": "pitch_01_reads_sub06.mp4", "type": "cross_modal"},
        ],
        "external": [{"doc": "sub_08_web_copy.pdf", "source": "web_dark_stores.txt"}],
        "ai_generated": ["sub_07_ai_generated.docx"],
        "template_docs": ["sub_01_original_a.pdf", "sub_02_exact_copy_of_01.docx",
                          "sub_03_original_b.pdf", "sub_09_original_c.docx", "sub_10_heavy_format.pdf"],
        "corrupted": ["sub_11_empty.pdf", "pitch_03_silent.mp4"],
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
