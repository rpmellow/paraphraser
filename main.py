from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk

# Download punkt tokenizer (only needed once)
nltk.download('punkt')

# Load model and tokenizer at startup
model_name = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# FastAPI app
app = FastAPI(title="Paraphraser API", description="Paraphrases paragraphs using T5", version="1.0")

# Input schema
class ParagraphInput(BaseModel):
    text: str

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
            do_sample=True,       # enables temperature
            top_k=50,             # sampling limit
            temperature=2.0       # creativity level
        )

        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased)

    return " ".join(paraphrased_sentences)

@app.post("/paraphrase")
async def paraphrase_endpoint(input_data: ParagraphInput):
    output_text = paraphrase_paragraph(input_data.text)
    return {"original": input_data.text, "paraphrased": output_text}
