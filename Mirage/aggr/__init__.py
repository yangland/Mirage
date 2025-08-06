# aggr/__init__.py
from aggr.fedavg_median import fedavg, coordinate_wise_median, trimmed_mean
from aggr.flame import flame_aggr
from aggr.krum import krum_aggr
from aggr.geometric_median import geo_med_aggr
from aggr.normbound import normbound_aggr

SUPPORTED_AGG_METHODS = {
    "fedavg": fedavg,
    "median": coordinate_wise_median,
    "trimmed_mean": trimmed_mean,
    "flame": flame_aggr,
    "krum": krum_aggr,
    "geo": geo_med_aggr,
    "normbound": normbound_aggr,
}

def aggregate_global_model(
    agg_method,
    server_model,
    client_grad_dict,
    **kwargs
):
    """
    Dispatch aggregation rule.
    """
    agg_method = agg_method.lower()

    if agg_method not in SUPPORTED_AGG_METHODS:
        raise ValueError(f"Unsupported aggregation method: {agg_method}")

    # ✅ Extract required kwarg(s)
    device = kwargs.get("device")

    return SUPPORTED_AGG_METHODS[agg_method](
        server_model.state_dict(),
        client_grad_dict,
        # device=device,  # ✅ Pass it explicitly
        **kwargs,       # In case the method needs other extras (e.g., f, m)
    )
