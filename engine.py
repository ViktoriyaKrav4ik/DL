import logging
import os
import numpy as np
from tqdm.auto import tqdm
from utils import RunningAverage, load_checkpoint, save_checkpoint, save_dict_to_json


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model using batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop
    summ = []
    # a running average object for loss
    loss_avg = RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)

            # compute model output and loss
            output_batch = model(train_batch)
            output_batch = output_batch.flatten()

            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                #print(labels_batch)
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    return metrics_mean
def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model using batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    # a running average object for loss
    loss_avg = RunningAverage()

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:
        # compute model output
        output_batch = model(data_batch)
        output_batch = output_batch.flatten()
        loss = loss_fn(output_batch, labels_batch)

        # update the average loss
        loss_avg.update(loss.item())

        # move data to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, loss_fn,
                       metrics, params, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (generator) a generator that generates batches of data and labels
        test_dataloader: (generator) a generator that generates batches of data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            params.model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    train_stats = []
    val_stats = []

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # Run training loop
        train_metrics = train(model, optimizer, loss_fn, train_dataloader,
                              metrics, params)

        # Run evaluation once
        val_metrics = evaluate(
            model, loss_fn, test_dataloader, metrics, params)

        # TODO: modify according to the metric of choice
        val_acc = val_metrics['Mean Squared Error']
        is_best = val_acc >= best_val_acc

        # Save weights
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'optim_dict': optimizer.state_dict()},
                        is_best=is_best,
                        checkpoint=params.model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                params.model_dir, "metrics_val_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            params.model_dir, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)

        train_stats.append(train_metrics)
        val_stats.append(val_metrics)

    return train_stats, val_stats