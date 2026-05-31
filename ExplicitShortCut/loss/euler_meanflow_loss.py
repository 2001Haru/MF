import torch

from utils.scheduler import LinearFlowScheduler


class EulerMeanFlowLoss:
    """JVP-free Euler MeanFlow loss for velocity or endpoint prediction.

    This repository uses t=0 for data and t=1 for noise. For a shortcut from
    t to r, where 0 <= r <= t, the endpoint-like parameterization is

        h_{t->r}(x_t) = x_t - t * u_{t->r}(x_t).

    The finite-difference EulerMF identities can be rearranged into stable
    bootstrap recurrences. Let s=max(r, t-dt), and advance the paired
    conditional path from x_t to x_s. Then:

        u_{t->r} = (t-s)/(t-r) * v_t + (s-r)/(t-r) * u_{s->r}

        h_{t->r} = r(t-s)/(s(t-r)) * x_data
                 + t(s-r)/(s(t-r)) * h_{s->r}.

    These forms are equivalent to the one-step finite-difference correction,
    but avoid JVP and avoid feeding the current prediction back into its own
    target. At r=t, the targets reduce to v_t and x_data respectively.
    """

    def __init__(
        self,
        path_type="linear",
        loss_type="l2",
        time_sampler="uniform",
        time_mu=-0.4,
        time_sigma=1.0,
        ratio_r_not_equal_t=0.75,
        adaptive_p=1.0,
        label_dropout_prob=0.1,
        prediction_type="velocity",
        loss_time_weight="none",
        euler_dt=0.01,
        endpoint_eps=1e-5,
    ):
        if path_type != "linear":
            raise NotImplementedError("EulerMeanFlowLoss currently supports only linear paths")
        if prediction_type not in {"velocity", "endpoint"}:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        if loss_time_weight not in {"none", "endpoint"}:
            raise ValueError(f"Unknown loss time weight: {loss_time_weight}")
        if euler_dt <= 0:
            raise ValueError("euler_dt must be positive")

        self.path_type = path_type
        self.loss_type = loss_type
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.adaptive_p = adaptive_p
        self.label_dropout_prob = label_dropout_prob
        self.prediction_type = prediction_type
        self.loss_time_weight = loss_time_weight
        self.euler_dt = euler_dt
        self.endpoint_eps = endpoint_eps
        self.flow_scheduler = LinearFlowScheduler()

    def sample_time_steps(self, batch_size, device):
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device)
            normal_samples = normal_samples * self.time_sigma + self.time_mu
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")

        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        equal_mask = torch.rand(batch_size, device=device) >= self.ratio_r_not_equal_t
        r = torch.where(equal_mask, t, r)
        return r, t

    def interpolant(self, t):
        return (
            self.flow_scheduler.alpha(t),
            self.flow_scheduler.sigma(t),
            self.flow_scheduler.d_alpha(t),
            self.flow_scheduler.d_sigma(t),
        )

    def _prepare_model_kwargs(self, model_tgt, kwargs, batch_size, device):
        model_kwargs = {}
        if kwargs.get("y") is not None:
            y = kwargs["y"].clone()
            model_kwargs["y"] = y
        if model_kwargs.get("y") is not None and self.label_dropout_prob > 0:
            dropout_mask = torch.rand(batch_size, device=device) < self.label_dropout_prob
            y[dropout_mask] = model_tgt.num_classes
            model_kwargs["y"] = y
        return model_kwargs

    def _bootstrap_target(self, model_tgt, images, z_t, v_t, r, t, model_kwargs):
        span = t - r
        is_anchor = span <= self.endpoint_eps
        dt = torch.minimum(torch.full_like(span, self.euler_dt), span)
        s = t - dt

        dt_view = dt.view(-1, 1, 1, 1)
        z_s = z_t - dt_view * v_t

        with torch.no_grad():
            next_prediction = model_tgt(z_s, r, s, **model_kwargs)

        safe_span = span.clamp_min(self.endpoint_eps)
        if self.prediction_type == "velocity":
            anchor_weight = dt / safe_span
            next_weight = (s - r) / safe_span
            target = (
                anchor_weight.view(-1, 1, 1, 1) * v_t
                + next_weight.view(-1, 1, 1, 1) * next_prediction
            )
            return torch.where(is_anchor.view(-1, 1, 1, 1), v_t, target)

        safe_s = s.clamp_min(self.endpoint_eps)
        anchor_weight = r * dt / (safe_s * safe_span)
        next_weight = t * (s - r) / (safe_s * safe_span)
        target = (
            anchor_weight.view(-1, 1, 1, 1) * images
            + next_weight.view(-1, 1, 1, 1) * next_prediction
        )
        endpoint_anchor = is_anchor | (s <= self.endpoint_eps)
        return torch.where(endpoint_anchor.view(-1, 1, 1, 1), images, target)

    def _loss(self, prediction, target, t):
        error = prediction - target.detach()
        squared_error = (error**2).reshape(error.shape[0], -1)
        mse_per_sample = torch.mean(squared_error, dim=-1)

        if self.loss_type == "adaptive":
            loss_mid = torch.sum(squared_error, dim=-1)
            weights = 1.0 / (loss_mid.detach() + 1e-3).pow(self.adaptive_p)
            loss = weights * loss_mid
        elif self.loss_type == "l2":
            loss = mse_per_sample
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.loss_time_weight == "endpoint":
            loss = loss * t.square()
        return loss, torch.mean(error**2)

    def __call__(self, model, model_tgt, images, kwargs=None):
        kwargs = {} if kwargs is None else kwargs.copy()
        batch_size = images.shape[0]
        device = images.device
        model_kwargs = self._prepare_model_kwargs(model_tgt, kwargs, batch_size, device)

        r, t = self.sample_time_steps(batch_size, device)
        noises = torch.randn_like(images)
        t_view = t.view(-1, 1, 1, 1)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_view)
        z_t = alpha_t * images + sigma_t * noises
        v_t = d_alpha_t * images + d_sigma_t * noises

        prediction = model(z_t, r, t, **model_kwargs)
        target = self._bootstrap_target(model_tgt, images, z_t, v_t, r, t, model_kwargs)
        return self._loss(prediction, target, t)
