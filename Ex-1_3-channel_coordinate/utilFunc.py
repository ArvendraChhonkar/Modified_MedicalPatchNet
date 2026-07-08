import csv
import gzip
import json
import torch

def readCSV(inpFile,delimiter=","):
    with open(inpFile, 'r') as read_obj: return list(csv.reader(read_obj,delimiter=delimiter))

def readCompressedJson(inputFile):
    with gzip.open(inputFile, "rt", encoding="utf-8") as fIn:
        return json.load(fIn)
    
def generate_coord_grid(H, W):
    x = torch.linspace(-1, 1, W).repeat(H, 1)   # shape (H, W)
    y = torch.linspace(-1, 1, H).unsqueeze(1).repeat(1, W)   # shape (H, W)
    return torch.stack([x, y], dim=0)            # shape (2, H, W)