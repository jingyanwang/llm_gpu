import torch

torch.cuda.get_device_name()

torch.cuda.device_count()

device = 7

model_id = "decapoda-research/llama-7b-hf"

###

print("loading model")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

model = model.to(device)

###

print("inference")

context = "My name is Jim."

question = "What is my name?"

prompt = f"""
{context}

Question: {question}

Answer:
"""

inputs = tokenizer(
    prompt, 
    return_tensors="pt")     

inputs = inputs.to(device) if device is not None else inputs

outputs = model.generate(**inputs)

result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(result)

print("continues inference")

while(True):
    outputs = model.generate(**inputs)
