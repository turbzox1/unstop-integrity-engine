# Batch integrity summary

Embedding: `tfidf(word1-2+char3-5)` (tfidf), index: `faiss.IndexFlatIP`, peer threshold 0.45, external threshold 0.45, AI detector: heuristic. 12/14 documents usable, 33 chunks, 4 reference sources.

| Document | Risk | Peer % | External % | AI % | Top match | Status |
|---|---|---|---|---|---|---|
| [pitch_01_reads_sub06.mp4](pitch_01_reads_sub06.mp4.md) | 🔴 HIGH | 85.7 | 0.0 | 1.5 | sub_06_cold_chain.pdf (0.474) | ok |
| [sub_01_original_a.pdf](sub_01_original_a.pdf.md) | 🔴 HIGH | 97.2 | 0.0 | 0.7 | sub_02_exact_copy_of_01.docx (1.0) | ok |
| [sub_02_exact_copy_of_01.docx](sub_02_exact_copy_of_01.docx.md) | 🔴 HIGH | 97.8 | 0.0 | 0.6 | sub_01_original_a.pdf (1.0) | ok |
| [sub_06_cold_chain.pdf](sub_06_cold_chain.pdf.md) | 🔴 HIGH | 55.8 | 0.0 | 7.2 | pitch_01_reads_sub06.mp4 (0.474) | ok |
| [sub_07_ai_generated.docx](sub_07_ai_generated.docx.md) | 🔴 HIGH | 0.0 | 0.0 | 99.7 | - | ok |
| [sub_08_web_copy.pdf](sub_08_web_copy.pdf.md) | 🔴 HIGH | 0.0 | 100.0 | 12.1 | - | ok |
| [pitch_03_silent.mp4](pitch_03_silent.mp4.md) | ⚪ NEEDS MANUAL REVIEW | n/a | n/a | n/a | - | no_audio |
| [sub_11_empty.pdf](sub_11_empty.pdf.md) | ⚪ NEEDS MANUAL REVIEW | n/a | n/a | n/a | - | empty |
| [pitch_02_original.mp4](pitch_02_original.mp4.md) | 🟢 LOW | 0.0 | 0.0 | n/a | - | ok |
| [sub_03_original_b.pdf](sub_03_original_b.pdf.md) | 🟢 LOW | 0.0 | 0.0 | 2.4 | - | ok |
| [sub_04_ev_original.docx](sub_04_ev_original.docx.md) | 🟢 LOW | 0.0 | 0.0 | 10.7 | - | ok |
| [sub_05_ev_paraphrase.docx](sub_05_ev_paraphrase.docx.md) | 🟢 LOW | 0.0 | 0.0 | 19.2 | - | ok |
| [sub_09_original_c.docx](sub_09_original_c.docx.md) | 🟢 LOW | 0.0 | 0.0 | 15.0 | - | ok |
| [sub_10_heavy_format.pdf](sub_10_heavy_format.pdf.md) | 🟢 LOW | 0.0 | 0.0 | 5.5 | - | ok |

## Pairwise peer-coverage matrix

Cell = % of the **row** document's non-template words matched in the **column** document.

| row \ col | pitch_01 | pitch_02 | sub_01 | sub_02 | sub_03 | sub_04 | sub_05 | sub_06 | sub_07 | sub_08 | sub_09 | sub_10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pitch_01 | · | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **85.7** | 0.0 | 0.0 | 0.0 | 0.0 |
| pitch_02 | 0.0 | · | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sub_01 | 0.0 | 0.0 | · | **97.2** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sub_02 | 0.0 | 0.0 | **97.8** | · | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sub_03 | 0.0 | 0.0 | 0.0 | 0.0 | · | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sub_04 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | · | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sub_05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | · | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| sub_06 | **55.8** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | · | 0.0 | 0.0 | 0.0 | 0.0 |
| sub_07 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | · | 0.0 | 0.0 | 0.0 |
| sub_08 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | · | 0.0 | 0.0 |
| sub_09 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | · | 0.0 |
| sub_10 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | · |
