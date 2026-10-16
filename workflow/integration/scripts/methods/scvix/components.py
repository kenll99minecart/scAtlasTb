import collections
from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch import nn
from torch.distributions import Normal

from scvi.nn._utils import ExpActivation


def _identity(x):
    return x


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, momentum, eps):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(self.num_features, momentum=momentum, eps=eps, affine=False)
        self.embed = nn.Embedding(num_classes, self.num_features * 2)
        self.embed.weight.data[:, : self.num_features].normal_(
            1, 0.02
        )  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, self.num_features :].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y.long().ravel()).chunk(2, 1)
        out = gamma.view(-1, self.num_features) * out + beta.view(-1, self.num_features)

        return out


class ConditionalLayerNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.ln = nn.LayerNorm(self.num_features, elementwise_affine=False)
        self.embed = nn.Embedding(num_classes, self.num_features * 2)
        self.embed.weight.data[:, : self.num_features].normal_(
            1, 0.02
        )  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, self.num_features :].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.ln(x)
        gamma, beta = self.embed(y.long().ravel()).chunk(2, 1)
        out = gamma.view(-1, self.num_features) * out + beta.view(-1, self.num_features)

        return out


class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_continuous
        The dimensionality of the continuous covariates
        including batch embeddings.
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_cont
        The dimensionality of the continuous covariates
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_continuous: int = 0,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
        conditional_norm: bool = False,
        conditional_category: int = 0,
    ):
        super().__init__()
        self.bias = bias
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_activation = use_activation
        self.conditional_norm = conditional_norm
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []
        self.n_continuous = n_continuous

        self.cond_cat = conditional_category
        if conditional_norm and self.n_cat_list[self.cond_cat] == 0:
            raise ValueError(
                "Conditional normalization is not applicable for a categorical variable with only "
                "one category."
            )

        self.n_cov = n_continuous + sum(self.n_cat_list)

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (f"Layer {i}", self._build_layer(n_in, n_out, i))
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:], strict=True)
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def _build_layer(self, n_in: int, n_out: int, layer_num: int) -> nn.Sequential:
        """Build one fully-connected layer: linear (+ optional norm / activation / dropout)."""
        return nn.Sequential(
            nn.Linear(
                n_in + self.n_cov * self.inject_into_layer(layer_num),
                n_out,
                bias=self.bias,
            ),
            # non-default params come from defaults in the original Tensorflow implementation
            ConditionalBatchNorm2d(n_out, self.n_cat_list[self.cond_cat], momentum=0.01, eps=0.001)
            if self.conditional_norm and self.use_batch_norm
            else nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
            if self.use_batch_norm
            else None,
            ConditionalLayerNorm(n_out, self.n_cat_list[self.cond_cat])
            if self.conditional_norm and self.use_layer_norm
            else nn.LayerNorm(n_out, elementwise_affine=False)
            if self.use_layer_norm
            else None,
            self.activation_fn() if self.use_activation else None,
            nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else None,
        )

    def _is_linear_layer(self, layer: nn.Module) -> bool:
        """Whether ``layer`` is a linear map covariate-injection and scArches hooks act on."""
        return isinstance(layer, nn.Linear)

    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[..., -categorical_dims:] = grad[..., -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if self._is_linear_layer(layer):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int, cont_input: torch.Tensor | None = None):
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample
        cont_input
            tensor of continuous covariates with shape ``(n_continuous,)``

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many idxs useless.
        cont_list = [cont_input] if cont_input is not None else []
        cat_list = cat_list or []

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        if (
            self.n_continuous > 0
            and cont_input is not None
            and cont_input.shape[-1] != self.n_continuous
        ):
            raise ValueError("continuous dims provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        cov_list = cont_list + one_hot_cat_list
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, ConditionalBatchNorm2d) or isinstance(
                        layer, ConditionalLayerNorm
                    ):
                        if x.dim() == 3:
                            x = torch.cat(
                                [
                                    (layer(x=slice_x, y=cat_list[self.cond_cat])).unsqueeze(0)
                                    for slice_x in x
                                ],
                                dim=0,
                            )
                        else:
                            x = layer(x=x, y=cat_list[self.cond_cat])
                    elif isinstance(layer, nn.BatchNorm1d):
                        x = self._apply_batch_norm(layer, x)
                    else:
                        x = self._apply_layer(layer, x, cov_list, i)
        return x

    def _apply_batch_norm(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply a batch-norm layer, handling the 3D (and MPS) case."""
        if x.dim() == 3:
            if x.device.type == "mps":  # TODO: remove this when MPS supports for loop.
                return torch.cat([(layer(slice_x.clone())).unsqueeze(0) for slice_x in x], dim=0)
            return torch.cat([layer(slice_x).unsqueeze(0) for slice_x in x], dim=0)
        return layer(x)

    def _apply_layer(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        cov_list: list[torch.Tensor],
        layer_index: int,
    ) -> torch.Tensor:
        """Apply a (non batch-norm) layer, injecting covariates into linear layers."""
        if self._is_linear_layer(layer) and self.inject_into_layer(layer_index):
            if x.dim() == 3:
                # For when we have sampling in inference. x: (n_samples, n_obs, n_hidden)
                cov_list_layer = [
                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1))) for o in cov_list
                ]
            else:
                cov_list_layer = cov_list
            x = torch.cat((x, *cov_list_layer), dim=-1)
        return layer(x)


# Encoder
class Encoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_continuous
        The dimensionality of the continuous covariates
        including batch embeddings.
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :func:`torch.exp`.
    return_dist
        Directly return the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_continuous: int = 0,
        n_cat_list: Iterable[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Callable | None = None,
        return_dist: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_continuous=n_continuous,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(
        self,
        x: torch.Tensor,
        *cat_list: int,
        cont: torch.Tensor | None = None,
    ):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from i.i.d. multivariate normal
            \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample
        cont
            optional tensor with shape (n_continuous,)

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list, cont_input=cont)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent


# Decoder
class DecoderSCVI(nn.Module):
    """Decodes data from a latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_continuous
        The dimensionality of the continuous covariates
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    n_conditions_output
        The number of conditions add to the scale and dropout parameters.
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_continuous: int = 0,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_conditions_output: int = 0,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus", "exp"] = "softmax",
        **kwargs,
    ):
        super().__init__()
        self.n_conditions_output = n_conditions_output
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_continuous=n_continuous,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        elif scale_activation == "exp":
            px_scale_activation = ExpActivation()

        # scale
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden + n_conditions_output, n_output),
            px_scale_activation,
        )
        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden + n_conditions_output, n_output)
        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden + n_conditions_output, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
        cont: torch.Tensor | None = None,
        output_condition: torch.Tensor | None = None,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then the value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z
            tensor with shape ``(n_input,)``
        library
            library size
        cat_list
            list of category membership(s) for this sample
        cont
            tensor with shape ``(n_continuous,)``
        output_condition
            tensor with shape ``(n_input,)`` used for conditioning the output layer

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list, cont_input=cont)
        if output_condition is not None and self.n_conditions_output:
            one_hot_cat = nn.functional.one_hot(
                output_condition.squeeze(-1), self.n_conditions_output
            )
        else:
            one_hot_cat = torch.zeros(px.size(-2), self.n_conditions_output).to(px.device)
        if px.dim() == 3:
            one_hot_cat = one_hot_cat.unsqueeze(0).expand(px.size(0), -1, -1)
        px_cat = torch.cat([px, one_hot_cat], dim=-1)
        px_scale = self.px_scale_decoder(px_cat)
        px_dropout = self.px_dropout_decoder(px_cat)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px_cat) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout
