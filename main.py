from fastapi import FastAPI
from pydantic import BaseModel
import nltk
import os

# Add path where punkt will be located (downloaded at build time)
nltk.data.path.append("/root/nltk_data")  # Leapcell build step location
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))  # local folder if needed

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
    result = paraphrase_paragraph(request.text)
    return {
        "original": request.text,
        "paraphrased": result
    }
