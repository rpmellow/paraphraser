from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# Download punkt tokenizer for sentence splitting
nltk.download('punkt')

# Load model and tokenizer
model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# FastAPI app
app = FastAPI()

# Request body schema
class ParagraphRequest(BaseModel):
    text: str

# Paraphrase function
def paraphrase_paragraph(paragraph: str) -> str:
    sentences = nltk.sent_tokenize(paragraph)
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

# API endpoint
@app.post("/paraphrase")
def paraphrase_api(request: ParagraphRequest):
    result = paraphrase_paragraph(request.text)
    return {
        "original": request.text,
        "paraphrased": result
    }

@app.get("/")
def root():
    return {"message": "Welcome to the Paraphrasing API! Use POST /paraphrase to paraphrase your text."}


# To run: uvicorn filename:app --reload
