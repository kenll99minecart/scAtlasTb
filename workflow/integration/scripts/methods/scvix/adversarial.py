"""Adversarial training plan vendored from the scvix branch of scvi-tools.

Placeholder until SCVIX ships with scvi-tools upstream. Remove this module
together with the rest of this package once that happens.
"""

import warnings
from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scvi import settings
from scvi.module import Classifier
from scvi.module.base import BaseModuleClass
from scvi.train import TrainingPlan

TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]


class AdversarialTrainingPlan(TrainingPlan):
    """Train vaes with adversarial loss option to encourage latent space mixing.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    optimizer
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`),
        or "Custom", which requires a custom optimizer creator callable to be passed via
        `optimizer_creator`.
    optimizer_creator
        A callable taking in parameters and returning a :class:`~torch.optim.Optimizer`.
        This allows using any PyTorch optimizer with custom hyperparameters.
    lr
        Learning rate used for the optimization when `optimizer_creator` is None.
    weight_decay
        Weight decay used in optimization, when `optimizer_creator` is None.
    eps
        eps used for optimization, when `optimizer_creator` is None.
    n_steps_kl_warmup
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None.
    n_epochs_kl_warmup
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`.
    reduce_lr_on_plateau
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus.
    lr_factor
        Factor to reduce the learning rate.
    lr_patience
        Number of epochs with no improvement after which the learning rate will be reduced.
    lr_threshold
        Threshold for measuring the new optimum.
    lr_scheduler_metric
        Which metric to track for learning rate reduction.
    lr_min
        Minimum learning rate allowed
    adversarial_classifier
        Whether to use adversarial classifier in the latent space
    adversarial_key
        Key in setup args to use for adversarial training.
    adversarial_steps
        Number of steps to train the adversarial classifier for each training step.
    scale_adversarial_loss
        Scaling factor on the adversarial components of the loss.
        By default, adversarial loss is scaled from 1 to 0 following the opposite of
        kl warmup.
    compile
        Whether to compile the model for faster training
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        optimizer: Literal["Adam", "AdamW", "Custom"] = "Adam",
        optimizer_creator: TorchOptimizerCreator | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        adversarial_classifier: bool | Classifier = False,
        adversarial_key: str = "batch",
        adversarial_steps: int = 1,
        scale_adversarial_loss: float | Literal["auto"] = "auto",
        compile: bool = False,
        compile_kwargs: dict | None = None,
        **loss_kwargs,
    ):
        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            compile=compile,
            compile_kwargs=compile_kwargs,
            **loss_kwargs,
        )
        self.adversarial_key = adversarial_key
        if adversarial_classifier is True:
            self.adversarial_steps = adversarial_steps
            self.n_adversarial_name = f"n_{adversarial_key}"
            if hasattr(self.module, self.n_adversarial_name):
                self.n_output_classifier = getattr(self.module, self.n_adversarial_name)
            else:
                raise ValueError(
                    f"Adversarial key {adversarial_key} not found in module setup args."
                )
            if self.n_output_classifier == 1:
                warnings.warn(
                    "Disabling adversarial classifier as there is only one class.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
                self.adversarial_classifier = False
            else:
                self.adversarial_classifier = Classifier(
                    n_input=self.module.n_latent + getattr(self.module, "n_adversarial_group", 0),
                    n_hidden=128,
                    n_labels=self.n_output_classifier,
                    n_layers=2,
                    logits=True,
                    use_batch_norm=False,
                    use_layer_norm=True,
                )

        else:
            self.adversarial_classifier = adversarial_classifier
        self.scale_adversarial_loss = scale_adversarial_loss
        self.automatic_optimization = False

    def loss_adversarial_classifier(
        self, z, adversarial_group, batch_index, predict_true_class=True
    ):
        """Loss for adversarial classifier."""
        n_classes = self.n_output_classifier
        n_adv_group = getattr(self.module, "n_adversarial_group", 0)
        if n_adv_group > 0:
            adversarial_group_ = torch.nn.functional.one_hot(
                adversarial_group, num_classes=n_adv_group
            ).float()
            z_cls = torch.cat([z, adversarial_group_], dim=1)
        else:
            z_cls = z
        if predict_true_class:  # train classifier
            z_cls = z_cls.detach()
        z = z_cls
        cls_logits = self.adversarial_classifier(z)

        if predict_true_class:
            cls_target = batch_index.squeeze(-1)
            loss = torch.nn.functional.cross_entropy(cls_logits, cls_target)
        else:
            one_hot_batch = torch.nn.functional.one_hot(batch_index.squeeze(-1), n_classes).float()
            cls_target = (1 - one_hot_batch) / (n_classes - 1)
            loss = (
                -(cls_target * torch.nn.functional.log_softmax(cls_logits, dim=1))
                .sum(dim=1)
                .mean()
            )

        return loss

    def training_step(self, batch, batch_idx):
        """Training step for adversarial training."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
            self.log("kl_weight", self.kl_weight, on_step=True, on_epoch=False)
        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        batch_tensor = batch[self.adversarial_key].long()

        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts

        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        z = inference_outputs["z"]
        adversarial_group = inference_outputs.get("adversarial_group", None)
        if adversarial_group is None:
            adversarial_group = torch.zeros(z.size(0)).to(z.device).long()
        else:
            adversarial_group = adversarial_group.squeeze(-1).long()
        loss = scvi_loss.loss
        orig_loss = loss
        # fool classifier if doing adversarial training
        if kappa > 0 and self.adversarial_classifier is not False:
            fool_loss = self.loss_adversarial_classifier(z, adversarial_group, batch_tensor, False)
            loss += fool_loss * kappa

        # on_step/on_epoch attributes only exist since scvi-tools 1.4
        on_step = getattr(self, "on_step", False)
        self.log("train_loss", loss, on_step=on_step, on_epoch=not on_step, prog_bar=True)
        if on_step:
            self.trainer.logger.log_metrics(
                {"train_loss_step": loss},
                step=self.global_step,
            )
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt1.zero_grad()
        self.manual_backward(loss)
        opt1.step()

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if opt2 is not None:
            loss = 0.0
            for i in range(self.adversarial_steps):
                qz = inference_outputs["qz"]
                z = qz.sample()
                loss_ = kappa * self.loss_adversarial_classifier(
                    z, adversarial_group, batch_tensor, True
                )
                if i > 1 and (loss - loss_) / loss < 1e-3:
                    break
                loss = loss_
                opt2.zero_grad()
                self.manual_backward(loss)
                opt2.step()

        # next part is for the usage of scib-metrics autotune with scvi
        if scvi_loss.extra_metrics is not None and len(scvi_loss.extra_metrics.keys()) > 0:
            self.prepare_scib_autotune(scvi_loss.extra_metrics, "training")

        return orig_loss

    def on_train_epoch_end(self):
        """Update the learning rate via scheduler steps."""
        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def on_validation_epoch_end(self) -> None:
        """Update the learning rate via scheduler steps."""
        if not self.reduce_lr_on_plateau or "validation" not in self.lr_scheduler_metric:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def configure_optimizers(self):
        """Configure optimizers for adversarial training."""
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )

        if self.adversarial_classifier is not False:
            params2 = filter(lambda p: p.requires_grad, self.adversarial_classifier.parameters())
            optimizer2 = torch.optim.Adam(params2, lr=3e-4, eps=1e-4, weight_decay=1e-9)
            config2 = {"optimizer": optimizer2}

            # pytorch lightning requires this way to return
            opts = [config1.pop("optimizer"), config2["optimizer"]]
            if "lr_scheduler" in config1:
                scheds = [config1["lr_scheduler"]]
                return opts, scheds
            else:
                return opts

        return config1

