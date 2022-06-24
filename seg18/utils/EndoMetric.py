
from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def general_dice(y_true, y_pred):
    result = []
    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [[instrument_id, dice(y_true == instrument_id, y_pred == instrument_id)]]
    return result


def general_jaccard(y_true, y_pred):
    result = []
    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [[instrument_id,jaccard(y_true == instrument_id, y_pred == instrument_id)]]
    return result

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)