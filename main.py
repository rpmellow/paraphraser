import os
# Set Huggingface & Torch cache dirs BEFORE importing transformers or nltk
os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["TORCH_HOME"] = "/tmp/torch_cache"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk

# Add paths where punkt will be located (downloaded at build time or fallback)
nltk.data.path.append("/root/nltk_data")  # Leapcell build step location
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))  # local folder if needed
nltk.data.path.append("/tmp/nltk_data")  # fallback location

import pathlib
# Download punkt if missing
if not pathlib.Path("/root/nltk_data/tokenizers/punkt").exists():
    nltk.download("punkt", download_dir="/tmp/nltk_data")

app = FastAPI()

tokenizer = None
model = None

class ParagraphRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Welcome to the Paraphrasing API! Use POST /paraphrase to paraphrase your text."}

def load_model():
    """
    Load tokenizer and model only once (lazy loading to save memory).
    """
    global tokenizer, model
    if tokenizer is None or model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "Vamsi/T5_Paraphrase_Paws"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def paraphrase_paragraph(paragraph: str) -> str:
    """
    Split paragraph into sentences, paraphrase each, then join back.
    """
    load_model()
    from nltk import sent_tokenize

    sentences = sent_tokenize(paragraph)
    paraphrased_sentences = []

    for sent in sentences:
        text = f"paraphrase: {sent} </s>"
        encoding = tokenizer.encode_plus(
            text,
            padding="longest",
            return_tensors="pt"
        )

        outputs = model.generate(
            encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            temperature=2.0,
        )

        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased)

    return " ".join(paraphrased_sentences)

@app.post("/paraphrase")
def paraphrase_api(request: ParagraphRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        result = paraphrase_paragraph(request.text)
    except LookupError as e:
        raise HTTPException(status_code=500, detail=f"NLTK resource missing: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    return {
        "original": request.text,
        "paraphrased": result
    }
