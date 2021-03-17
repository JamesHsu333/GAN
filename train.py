import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")

def train(discriminator, generator, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, dataloader, metrics, params):
    # Set model to training mode
    discriminator.train()
    generator.train()

    # Summary for current training loop and a running average object for loss
    summ = []
    d_loss_avg = utils.RunningAverage()
    g_loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(
                    non_blocking=True), labels_batch.cuda(non_blocking=True)
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # Discriminator Forward
            real_outputs = discriminator(train_batch).view(-1,1)
            real_label = torch.ones(train_batch.shape[0], 1).to('cuda:0')

            # Generator Forward
            noise = torch.randn(params.batch_size, 64, 1, 1)
            noise = noise.to('cuda:0')
            fake_inputs = generator(noise)
            fake_outputs = discriminator(fake_inputs).view(-1,1)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to('cuda:0')

            outputs = torch.cat((real_outputs, fake_outputs), 0)
            targets = torch.cat((real_label, fake_label), 0)
            
            # Backward
            d_optimizer.zero_grad()
            d_loss = d_loss_fn(outputs, targets)
            d_loss.backward()
            d_optimizer.step()

            noise = torch.randn(params.batch_size, 64, 1, 1)
            noise = noise.to('cuda:0')

            fake_inputs = generator(noise)
            fake_outputs = discriminator(fake_inputs)

            g_loss = g_loss_fn(fake_outputs)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # update the average loss
            d_loss_avg.update(d_loss.item())
            g_loss_avg.update(g_loss.item())

            t.set_postfix(d_loss='{:05.3f}'.format(d_loss_avg()), g_loss='{:05.3f}'.format(g_loss_avg()))
            t.update()

def train_and_evaluate(discriminator, generator, train_dataloader, val_dataloader, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, metrics, params, model_dir, restore_file, p):
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, generator, g_optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(discriminator, generator,d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, train_dataloader, metrics, params)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': generator.state_dict(),
                               'optim_dict': g_optimizer.state_dict()},
                              is_best=True,
                              checkpoint=model_dir)

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    
    params.cuda = torch.cuda.is_available()

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
    
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading the datasets...")

    if args.data_dir != 'data/':
        dataloader = data_loader.fetch_dataloader(
            ['train', 'val'], args.data_dir, params)
    else:
        dataloader = data_loader.mnist_dataloader(
            ['train', 'val'], params)

    train_dl = dataloader['train']
    val_dl = dataloader['val']

    logging.info("-done")

    discriminator = net.Discriminator().cuda() if params.cuda else net.Discriminator()
    d_optimizer = optim.Adam(discriminator.parameters(), lr=params.learning_rate, betas=(0.5, 0.999))

    generator = net.Generator().cuda() if params.cuda else net.Generator()
    g_optimizer = optim.Adam(generator.parameters(), lr=params.learning_rate, betas=(0.5, 0.999))

    d_loss_fn = net.d_loss_fn
    g_loss_fn = net.g_loss_fn
    metrics = None

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(discriminator, generator, train_dl, val_dl, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, metrics, params, args.model_dir, args.restore_file)
