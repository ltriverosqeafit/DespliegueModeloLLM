from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import load_checkpoint_and_dispatch
import torch

modelo = "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(modelo)

model = AutoModelForCausalLM.from_pretrained(
    modelo,
    device_map="auto",  
    torch_dtype=torch.float16  
)

model = load_checkpoint_and_dispatch(
    model, 
    modelo, 
    device_map="auto", 
    offload_folder="./offload" 
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)

app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(query: Query):
    prompt = f"<s>[INST] {query.prompt} [/INST]"
    result = pipe(prompt, do_sample=True)
    return {"response": result[0]["generated_text"]}
