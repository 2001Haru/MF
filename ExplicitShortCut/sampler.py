import torch
from utils import append_dims
from utils.solver import ddim_solver_condv, ddim_solver_condx0

@torch.no_grad()
def cfg_sampler(
    model,
    latents,
    y=None,
    cfg_scale=1.0,
    num_steps=1,
    scheduler=None,
    prediction_type="velocity",
):
    """
    MeanFlow sampler supporting both single-step and multi-step generation
    
    Based on Eq.(12): z_r = z_t - (t-r)u(z_t, r, t)
    For single-step: z_0 = z_1 - u(z_1, 0, 1)
    For multi-step: iteratively apply the Eq.(12) with intermediate steps
    """
    if prediction_type not in {"velocity", "endpoint"}:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    batch_size = latents.shape[0]
    device = latents.device
    sigma_data = scheduler.sigma_0
    latents = latents * sigma_data
    # Prepare for CFG if needed
    do_cfg = y is not None and cfg_scale > 1.0
    if do_cfg:
        if hasattr(model, 'module'):  # DDP
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes
        null_y = torch.full_like(y, num_classes)
    
    if num_steps == 1:
        r = torch.zeros(batch_size, device=device)
        t = torch.ones(batch_size, device=device)
        c_in, c_out = append_dims(scheduler.c_in(t), latents.ndim), append_dims(scheduler.c_out(t), latents.ndim)
        
        if do_cfg:
            z_combined = torch.cat([latents, latents], dim=0)
            r_combined = torch.cat([r, r], dim=0)
            t_combined = torch.cat([t, t], dim=0)
            y_combined = torch.cat([y, null_y], dim=0)
            
            prediction_combined = model(z_combined * c_in, r_combined, t_combined, y=y_combined)
            if prediction_type == "velocity":
                prediction_combined = prediction_combined * c_out
            prediction_cond, prediction_uncond = prediction_combined.chunk(2, dim=0)
            
            prediction = prediction_uncond + cfg_scale * (prediction_cond - prediction_uncond)
        else:
            prediction = model(latents * c_in, r, t, y=y)
            if prediction_type == "velocity":
                prediction = prediction * c_out
        
        if prediction_type == "velocity":
            x0 = ddim_solver_condv(latents, prediction, t, r, scheduler.alpha_bar, scheduler.beta_bar)
        else:
            x0 = ddim_solver_condx0(latents, prediction, t, r, scheduler.alpha_hat, scheduler.beta_hat)
        
    else:
        z = latents
        
        time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t_cur = time_steps[i]
            t_next = time_steps[i + 1]
            
            t = torch.full((batch_size,), t_cur, device=device)
            r = torch.full((batch_size,), t_next, device=device)
            c_in, c_out = append_dims(scheduler.c_in(t), latents.ndim), append_dims(scheduler.c_out(t), latents.ndim)
            
            if do_cfg:
                z_combined = torch.cat([z, z], dim=0)
                r_combined = torch.cat([r, r], dim=0)
                t_combined = torch.cat([t, t], dim=0)
                y_combined = torch.cat([y, null_y], dim=0)
                
                prediction_combined = model(z_combined * c_in, r_combined, t_combined, y=y_combined)
                if prediction_type == "velocity":
                    prediction_combined = prediction_combined * c_out
                prediction_cond, prediction_uncond = prediction_combined.chunk(2, dim=0)
                
                # Apply CFG
                prediction = prediction_uncond + cfg_scale * (prediction_cond - prediction_uncond)
            else:
                prediction = model(z * c_in, r, t, y=y)
                if prediction_type == "velocity":
                    prediction = prediction * c_out
            
            if prediction_type == "velocity":
                # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
                z = ddim_solver_condv(z, prediction, t, r, scheduler.alpha_bar, scheduler.beta_bar)
            else:
                z = ddim_solver_condx0(z, prediction, t, r, scheduler.alpha_hat, scheduler.beta_hat)
        
        x0 = z
    
    return x0
