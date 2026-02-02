from fastapi import FastAPI
import torch
from model import GNNLSTM
import numpy as np

app = FastAPI()

model = GNNLSTM(num_nodes=207)
model.load_state_dict(torch.load("model.pt"))
model.eval()


@app.post("/predict")
def predict(data: list):
    x = torch.FloatTensor(np.array(data)).unsqueeze(0)
    with torch.no_grad():
        pred = model(x, edge_index=None)
    return {"prediction": pred.tolist()}
