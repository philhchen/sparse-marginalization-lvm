import argparse
import pathlib

import torch
import torch.nn.functional as F
from torchvision import datasets
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from lvmhelpers.marg import \
    ExplicitWrapper, Marginalizer
from lvmhelpers.sfe import \
    BitVectorReinforceWrapper, ReinforceDeterministicWrapper, \
    BitVectorScoreFunctionEstimator
from lvmhelpers.gumbel import \
    GumbelSoftmaxWrapper, Gumbel
from lvmhelpers.utils import DeterministicWrapper, populate_common_params

from data import transform
from archs import MLP, CategoricalGenerator
from opts import populate_experiment_params


class VAE(pl.LightningModule):
    def __init__(
            self,
            n_features,
            hidden_size,
            out_rank,
            out_classes,
            budget,
            init,
            mode,
            entropy_coeff,
            latent_size,
            normalizer,
            gs_tau,
            temperature_decay,
            temperature_update_freq,
            straight_through,
            baseline_type,
            topk,
            random_seed,
            batch_size,
            lr,
            weight_decay,
            optimizer):
        super(VAE, self).__init__()

        self.save_hyperparameters()

        inf = MLP(
            dim_in=n_features,
            dim_hid=self.hparams.hidden_size,
            dim_out=self.hparams.latent_size,
            n_layers=0)
        gen = MLP(
            dim_in=self.hparams.latent_size,
            dim_hid=self.hparams.hidden_size,
            dim_out=self.hparams.n_features * self.hparams.out_rank,
            n_layers=0)
        gen = CategoricalGenerator(
            gen,
            n_features=self.hparams.n_features,
            out_rank=self.hparams.out_rank,
            n_classes=256)

        loss_fun = reconstruction_loss

        if self.hparams.mode == 'sfe':
            inf = BitVectorReinforceWrapper(inf, baseline_type=self.hparams.baseline_type)
            gen = ReinforceDeterministicWrapper(gen)
            lvm_method = BitVectorScoreFunctionEstimator
        elif self.hparams.mode == 'gs':
            inf = GumbelSoftmaxWrapper(
                inf,
                temperature=self.hparams.gs_tau,
                straight_through=self.hparams.straight_through)
            gen = DeterministicWrapper(gen)
            lvm_method = Gumbel
        elif self.hparams.mode == 'marg':
            inf = ExplicitWrapper(inf, normalizer=self.hparams.normalizer)
            gen = DeterministicWrapper(gen)
            lvm_method = Marginalizer
        else:
            raise RuntimeError(f"Unknown training mode: {self.hparams.mode}")

        self.lvm_method = lvm_method(
            inf,
            gen,
            loss_fun,
            encoder_entropy_coeff=1.0)

    def forward(self, inf_input, labels):
        return self.lvm_method(inf_input, torch.zeros_like(inf_input), labels)

    def training_step(self, batch, batch_nb):
        inf_input, labels = batch
        if inf_input.dtype == torch.uint8:
            inf_input = inf_input.to(dtype=torch.float) / 255
        training_result = self(inf_input, labels)
        loss = training_result['loss']

        result = pl.TrainResult(minimize=loss)
        elbo = \
            training_result['log']['loss'] + \
            training_result['log']['encoder_entropy'] + \
            self.hparams.latent_size * torch.log(torch.tensor(0.5))
        result.log(
            '-train_elbo',
            -elbo,
            prog_bar=True)

        if 'nonzeros' in training_result['log'].keys():
            result.log(
                'train_nonzeros',
                training_result['log']['nonzeros'],
                prog_bar=True)

        # Update temperature if Gumbel
        if self.hparams.mode == 'gs':
            self.lvm_method.encoder.update_temperature(
                self.global_step,
                self.hparams.temperature_update_freq,
                self.hparams.temperature_decay)
            result.log('temperature', self.lvm_method.encoder.temperature)

        return result

    def validation_step(self, batch, batch_nb):
        inf_input, labels = batch
        if inf_input.dtype == torch.uint8:
            inf_input = inf_input.to(dtype=torch.float) / 255
        validation_result = self(inf_input, labels)
        result = pl.EvalResult(checkpoint_on=validation_result['log']['loss'])
        elbo = \
            validation_result['log']['loss'] + \
            validation_result['log']['encoder_entropy'] + \
            self.hparams.latent_size * torch.log(torch.tensor(0.5))
        result.log(
            '-val_elbo',
            -elbo,
            prog_bar=True)

        if 'nonzeros' in validation_result['log'].keys():
            result.log(
                'val_nonzeros',
                validation_result['log']['nonzeros'],
                prog_bar=True)
        return result

    def test_step(self, batch, batch_nb):
        inf_input, labels = batch
        if inf_input.dtype == torch.uint8:
            inf_input = inf_input.to(dtype=torch.float) / 255
        test_result = self(inf_input, labels)
        result = pl.EvalResult()
        elbo = \
            test_result['log']['loss'] + \
            test_result['log']['encoder_entropy'] + \
            self.hparams.latent_size * torch.log(torch.tensor(0.5))
        result.log(
            '-test_elbo',
            -elbo,
            prog_bar=True)

        if 'nonzeros' in test_result['log'].keys():
            result.log(
                'test_nonzeros',
                test_result['log']['nonzeros'],
                prog_bar=True)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                datasets.FashionMNIST(
                    'data/bit_vector-vae/fmnist_data/',
                    train=True,
                    download=True,
                    transform=transform),
                indices=range(55000)),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                datasets.FashionMNIST(
                    'data/bit_vector-vae/fmnist_data/',
                    train=True,
                    download=True,
                    transform=transform),
                indices=range(55000, 60000)),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                datasets.FashionMNIST(
                    'data/bit_vector-vae/fmnist_data/',
                    train=False,
                    download=True,
                    transform=transform)),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)


def reconstruction_loss(
        _inf_input,
        discrete_latent_z,
        _gen_input,
        gen_output,
        true_labels):
    Xhat_logits = gen_output.permute(0, 2, 1)
    lv = F.cross_entropy(
        Xhat_logits, _inf_input.to(dtype=torch.long), reduction="none"
    )
    return -lv.sum(dim=1), {}


def get_model(opt):
    n_features = 28 * 28
    hidden_size = 128
    out_rank = 5
    out_classes = 256
    game = VAE(
        n_features=n_features,
        hidden_size=hidden_size,
        out_rank=out_rank,
        out_classes=out_classes,
        budget=opt.budget,
        init=not opt.noinit,
        mode=opt.mode,
        entropy_coeff=opt.entropy_coeff,
        latent_size=opt.latent_size,
        normalizer=opt.normalizer,
        gs_tau=opt.gs_tau,
        temperature_decay=opt.temperature_decay,
        temperature_update_freq=opt.temperature_update_freq,
        straight_through=opt.straight_through,
        baseline_type=opt.baseline_type,
        topk=opt.topk,
        random_seed=opt.random_seed,
        batch_size=opt.batch_size,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        optimizer=opt.optimizer)

    return game


def main(params):

    arg_parser = argparse.ArgumentParser()
    arg_parser = populate_experiment_params(arg_parser)
    arg_parser = populate_common_params(arg_parser)
    opts = arg_parser.parse_args(params)

    # fix seed
    pl.seed_everything(opts.random_seed)

    pathlib.Path(
            'data/bit_vector-vae/fmnist_data/').mkdir(parents=True, exist_ok=True)

    bit_vector_vae = get_model(opts)

    experiment_name = 'bit-vector'
    model_name = '%s/%s' % (experiment_name, opts.mode)
    other_info = [
        "lr-{}".format(opts.lr),
        "latent_size-{}".format(opts.latent_size),
    ]
    if opts.mode == "sparsemap":
        if opts.budget > 0:
            other_info.append(f"b{opts.budget}")
        if opts.noinit:
            other_info.append("noinit")

    model_name = '%s/%s' % (model_name, '_'.join(other_info))

    tb_logger = pl_loggers.TensorBoardLogger(
        'logs/',
        name=model_name)

    trainer = pl.Trainer(
        progress_bar_refresh_rate=20,
        logger=tb_logger,
        max_epochs=opts.n_epochs,
        weights_save_path='checkpoints/',
        weights_summary='full',
        gpus=1 if torch.cuda.is_available() else 0,
        resume_from_checkpoint=opts.load_from_checkpoint,
        deterministic=True)
    trainer.fit(bit_vector_vae)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])