# LegalBERT India — deployment starter

This repository includes the original research files plus a small Streamlit demo interface.

## Run locally

1. Install Python 3.10 or 3.11.
2. Open a terminal in this folder.
3. Create and activate a virtual environment.
4. Install dependencies: `pip install -r requirements.txt`
5. Start the app: `streamlit run app.py`

## Current demo features

- Extract text from text-based PDF, DOCX, and TXT files.
- Display regex-based entity hints.
- Display keyword-based clause hints.
- Search records in `legalbertindia-main/precedents.db`.

## Important limitations

The uploaded project archive does not contain the fine-tuned LegalBERT model weights or tokenizer files. The current precedent database contains only two demo records, and their summaries are placeholders. This starter does not claim to run the fine-tuned model, OCR, or a complete legal risk engine. Verify legal outputs against authoritative sources; this is not legal advice.

## Free deployment

Push this folder to a GitHub repository, then deploy `app.py` on Streamlit Community Cloud using `requirements.txt`. Choose this folder/repository root as the app location. Free hosting limits and provider availability can change.
