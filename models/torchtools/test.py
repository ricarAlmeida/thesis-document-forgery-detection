""" Module that implements functions used for testing """

from typing import Union, Optional
from pathlib import Path
import json
from tqdm import tqdm
import timeit

from torch.utils.data import DataLoader
import torch

from .metrics import Metrics
from .model import ModelRunner


def evaluate(
    model: ModelRunner, 
    loader: DataLoader, 
    metrics: Metrics, 
    device: str, 
    save_path: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Evaluate a given data loader

    Parameters:
        model: model to be evaluated
        loader: data loader
        metrics: collection of metrics to be tested
        device: hardware to use. Options are cpu or cuda
        save_path: save result path
    """
    
    run_times = []
    metrics_times = []
    
    model.model.to(device)
    metrics.to(device)
    model.model.eval()

    number_batches = len(loader)

    with torch.no_grad():
    
        for batch_idx, batch in tqdm(enumerate(loader)):
            print(f"Batch: {batch_idx}/{number_batches}")

            t_run0 = timeit.default_timer()
            outputs = model.logits(batch, device, None)
            t_run = timeit.default_timer() - t_run0
            run_times.append(t_run)
            print("Time to run model: ", t_run)
            logits = outputs.logits
            labels = outputs.labels

            t_metrics0 = timeit.default_timer()
            values = metrics.add_batch(logits, labels)
            t_metrics = timeit.default_timer() - t_metrics0
            metrics_times.append(t_metrics)
            print("Time to calculate metrics: ", t_metrics)
            print("Batch Metrics: ", values)

    result = metrics.evaluate()

    result = {
        "metrics": result,
        "run_times": run_times,
        "metrics_times": metrics_times,
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(result, f)
            
    return result