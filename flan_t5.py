'''
python flan_t5.py --gpu_id 7 &
'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gpu_id')

args = parser.parse_args()

print(f"gpu {args.gpu_id}")


import torch

torch.cuda.get_device_name()

torch.cuda.device_count()

device = int(args.gpu_id)

###

print("loading model")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base",
    )

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base",
    )


model = model.to(device)

###

print("inference")

context = "My name is Jim."

question = "What is my name?"

prompt = f"""
{context}

Q: {question}
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
