from fastapi import FastAPI, Response
from pydantic import BaseModel

import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")


class Body(BaseModel):
    phrase: str


app = FastAPI()


@app.get('/')
def root():
    return Response("<h1>A self-documenting API to interact with an ONNX model</h1>")


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()


@app.post('/predict')
def predict(body: Body):
    input_ids = torch.tensor(
        tokenizer.encode(body.phrase, add_special_tokens=True)
    ).unsqueeze(
        0
    )

    inputs = {session.get_inputs()[0].name: to_numpy(input_ids)}
    out = session.run(None, inputs)

    result = np.argmax(out)
    return {'positive': bool(result)}
