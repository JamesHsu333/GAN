import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import matplotlib.pyplot as plt
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--show_images', default='no', help="Show image")

def evaluate(model, loss_fn, metrics, params):
    model.eval()

    summ = []
    args = parser.parse_args()

    noise = torch.randn(128, 64, 1, 1)
    noise = noise.to('cuda:0')

    outputs = model(noise)
    outputs =  outputs.data.cpu().numpy()

    # Show image
    if args.show_images != 'no':
        show_images(outputs)
        plt.show()



# Show images
def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(28, 28))
        plt.axis('off')

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

    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    model = net.Generator().cuda() if params.cuda else net.Generator()

    loss_fn = net.g_loss_fn
    metrics = None

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    evaluate(model, loss_fn, metrics, params)

    logging.info("- done.")