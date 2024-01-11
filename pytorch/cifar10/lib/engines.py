import tqdm
import torch

from torchmetrics.aggregation import MeanMetric


def train_one_epoch(model, loader, metric_fn, loss_fn, device, optimizer, scheduler):
    model.train()
    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()    
    
    for inputs, targets in tqdm.tqdm(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        accuracy = metric_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_epoch.update(loss.to('cpu'))
        accuracy_epoch.update(accuracy.to('cpu'))
    
    summary = {
        'loss': loss_epoch.compute(),
        'accuracy': accuracy_epoch.compute(),
    }

    return summary


def eval_one_epoch(model, loader, metric_fn, loss_fn, device):
    model.eval()
    loss_epoch = MeanMetric()
    accuracy_epoch = MeanMetric()    
    
    for inputs, targets in tqdm.tqdm(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        accuracy = metric_fn(outputs, targets)
        
        loss_epoch.update(loss.to('cpu'))
        accuracy_epoch.update(accuracy.to('cpu'))
    
    summary = {
        'loss': loss_epoch.compute(),
        'accuracy': accuracy_epoch.compute(),
    }

    return summary