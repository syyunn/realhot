import torch
import torch.nn as nn

from utils import idx2onehot


class AE(nn.Module):

    def __init__(self,
                 encoder_layer_sizes,
                 latent_size,
                 decoder_layer_sizes,
                 conditional=False,
                 num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes,
            latent_size,
            conditional,
            num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes,
            latent_size,
            conditional,
            num_labels)

    def forward(self,
                x,
                c=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        z = self.encoder(x, c)

        recon_x = self.decoder(z, c)

        return recon_x, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size,
                         self.latent_size])

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self,
                 layer_sizes,
                 latent_size,
                 conditional,
                 num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1],
                                                    layer_sizes[1:])):
            print(i, ": ", in_size, out_size)
            self.MLP.add_module(
                name="L{:d}".format(i),
                module=nn.Linear(in_size, out_size))
            print("ReLU added @ Encoder")
            self.MLP.add_module(name="A{:d}".format(i),
                                module=nn.ReLU())

        self.linear = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=10)
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        z = self.linear(x)

        return z


class Decoder(nn.Module):

    def __init__(self,
                 layer_sizes,
                 latent_size,
                 conditional,
                 num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(
                zip([input_size]+layer_sizes[:-1], layer_sizes)):
            print(i, ": ", in_size, out_size)
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                print("ReLU added @ Decoder")
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                print("Sig step")
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=10)
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
