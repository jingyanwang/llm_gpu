import torch

torch.cuda.get_device_name()

torch.cuda.device_count()

device = 3

model_id = "EleutherAI/gpt-j-6B"

###

print("loading model")

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    )

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    )


model = model.to(device)

###

print("inference")

context = "My name is Jim."

question = "What is my name?"

prompt = f"""
{context}

Question: {question}
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
