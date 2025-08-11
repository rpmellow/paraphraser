from fastapi import FastAPI
from pydantic import BaseModel
import nltk

nltk.download("punkt")

app = FastAPI()

tokenizer = None
model = None

class ParagraphRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Welcome to the Paraphrasing API! Use POST /paraphrase to paraphrase your text."}

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "Vamsi/T5_Paraphrase_Paws"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def paraphrase_paragraph(paragraph: str) -> str:
    load_model()
    from nltk import sent_tokenize
    sentences = sent_tokenize(paragraph)
    paraphrased_sentences = []

    for sent in sentences:
        text = "paraphrase: " + sent + " </s>"
        encoding = tokenizer.encode_plus(
            text, padding='longest', return_tensors="pt"
        )

        outputs = model.generate(
            encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
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

# Run with: uvicorn main:app --host 0.0.0.0 --port 8080
