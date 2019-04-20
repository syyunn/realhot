import os
import time
import torch
import logging
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict

from models.ae import AE


def main(args, ts, logPath):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MNIST(
        root='../data', train=True, transform=transforms.ToTensor(),
        download=True)
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)

    def loss_fn(recon_x, x):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')

        return BCE / x.size(0)

    ae = AE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(ae.parameters(),
                                 lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, z = ae(x, y)
            else:
                recon_x, z = ae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                if z.shape[1] != 1:
                    tracker_epoch[id]['y'] = z[i, 1].item()
                else:
                    tracker_epoch[id]['y'] = 0
                tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                logContent = "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item())
                print(logContent)
                with open(logPath, 'a') as log:
                    for arg in vars(args):
                        log.write(logContent + "\n")

                if args.conditional:
                    c = torch.arange(0, 10).long().unsqueeze(1)
                    x = ae.inference(n=c.size(0), c=c)
                else:
                    x = ae.inference(n=10)

                plt.figure()
                plt.figure(figsize=(5, 10))
                for p in range(10):
                    plt.subplot(5, 2, p+1)
                    if args.conditional:
                        plt.text(
                            0, 0, "c={:d}".format(c[p].item()), color='black',
                            backgroundcolor='white', fontsize=8)
                    plt.imshow(x[p].view(28, 28).data.numpy())
                    plt.axis('off')

                if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    if not(os.path.exists(os.path.join(args.fig_root))):
                        os.mkdir(os.path.join(args.fig_root))
                    os.mkdir(os.path.join(args.fig_root, str(ts)))

                plt.savefig(
                    os.path.join(args.fig_root, str(ts),
                                 "E{:d}I{:d}.png".format(epoch, iteration)),
                    dpi=300)
                plt.clf()
                plt.close('all')

        df = pd.DataFrame.from_dict(tracker_epoch, orient='index')

        g = sns.lmplot(
            x='x', y='y', hue='label', data=df.groupby('label').head(100),
            fit_reg=False, legend=True)

        g.savefig(os.path.join(
            args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
            dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='ae/figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    ts = time.time()

    logPath = os.path.join(args.fig_root, str(ts), 'log.txt')
    os.makedirs(os.path.dirname(logPath))
    with open(logPath, 'a') as log:
        for arg in vars(args):
            log.write(arg + ": " + str(getattr(args, arg)) + "\n")

    main(args, ts, logPath)
