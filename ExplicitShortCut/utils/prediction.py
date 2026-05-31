from .utils import append_dims


def predict_velocity(model, x, r, t, y=None, prediction_type="velocity", eps=1e-5):
    prediction = model(x, r, t, y=y)
    if prediction_type == "velocity":
        return prediction
    if prediction_type == "endpoint":
        return (x - prediction) / append_dims(t, x.ndim).clamp_min(eps)
    raise ValueError(f"Unknown prediction type: {prediction_type}")


def predict_endpoint(model, x, r, t, y=None, prediction_type="velocity"):
    prediction = model(x, r, t, y=y)
    if prediction_type == "endpoint":
        return prediction
    if prediction_type == "velocity":
        return x - append_dims(t, x.ndim) * prediction
    raise ValueError(f"Unknown prediction type: {prediction_type}")
