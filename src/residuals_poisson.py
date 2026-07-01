import torch
from src.grad_utils import *

class ResidualsPoisson:
    """
    Residual evaluation for the 2D Poisson equation, forward-solve setup:
    the model is conditioned on a fixed source field f and predicts u, with
    u = 0 enforced on the boundary by construction of the training data.

        residual(x, y) = laplacian(u_pred)(x, y) - f(x, y)

    Calling convention matches ResidualsDarcy / ResidualsMechanics:
    compute_residual receives `input` as a tuple where
        input[0] = model_input = (x, t)   # x already has f concatenated as an extra channel
        input[1] = conditioning_f         # the known source field, shape [batch, 1, pixels, pixels]
    """
    def __init__(self, model, fd_acc, pixels_per_dim, pixels_at_boundary,
                 device='cpu', domain_length=1., use_ddim_x0=False, ddim_steps=0):
        self.gov_eqs = 'poisson'
        self.model = model
        self.pixels_at_boundary = pixels_at_boundary
        self.pixels_per_dim = pixels_per_dim
        self.device = device

        if self.pixels_at_boundary:
            d0 = domain_length / (pixels_per_dim - 1)
            d1 = domain_length / (pixels_per_dim - 1)
        else:
            d0 = domain_length / pixels_per_dim
            d1 = domain_length / pixels_per_dim

        self.grads = GradientsHelper(d0=d0, d1=d1, fd_acc=fd_acc, periodic=False, device=device)

        self.use_ddim_x0 = use_ddim_x0
        self.ddim_steps = ddim_steps

    def compute_residual(self, input, reduce='none', return_model_out=False,
                          return_optimizer=False, return_inequality=False,
                          sample=False, ddim_func=None, pass_through=False):

        if pass_through:
            assert isinstance(input, torch.Tensor), 'Input is assumed to directly be given output.'
            x0_pred = input
            model_out = x0_pred
            conditioning_f = None  # not available in pass-through mode
        else:
            model_input, conditioning_f = input
            assert len(model_input) == 2 and isinstance(model_input, tuple), \
                'model_input must be a tuple consisting of noisy signal and time.'
            noisy_in, time = iter(model_input)

            if self.use_ddim_x0:
                x0_pred, model_out = ddim_func(noisy_in, time, self.model, noisy_in.shape,
                                                self.ddim_steps, 0.)
            else:
                x0_pred = self.model(noisy_in, time)
                model_out = x0_pred

        assert len(x0_pred.shape) == 4, \
            'Model output must be a tensor shaped as an image (with explicit axes for the spatial dimensions).'

        u = x0_pred[:, 0]                       # [batch, pixels, pixels]
        f = conditioning_f[:, 0]                # [batch, pixels, pixels]

        u_d00 = self.grads.stencil_gradients(u, mode='d_d00')
        u_d11 = self.grads.stencil_gradients(u, mode='d_d11')
        laplacian_u = u_d00 + u_d11

        eq_0 = laplacian_u - f
        eq_0 = generalized_image_to_b_xy_c(eq_0.unsqueeze(1))  # -> [batch, pixels*pixels, 1]

        output = {}
        output['residual'] = eq_0

        if return_model_out:
            output['model_out'] = model_out

        if reduce == 'full':
            return {k: (v.mean() if torch.is_tensor(v) else v) for k, v in output.items()}
        elif reduce == 'per-batch':
            # average over the spatial (pixel) dimension only, keep the batch and component dims
            # so downstream code (which expects shape [batch, n_components]) works the same way
            # it does for the other governing equations.
            return {k: (v.mean(dim=1) if torch.is_tensor(v) and k == 'residual' else v)
                    for k, v in output.items()}
        else:
            return output
