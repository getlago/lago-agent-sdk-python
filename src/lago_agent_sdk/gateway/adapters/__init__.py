from .cloudflare_gateway import extract_cloudflare_log, resolve_subscription
from .databricks_gateway import extract_databricks_log, resolve_databricks_subscription

# `resolve_subscription` predates the second gateway and reads Cloudflare's
# `cf-aig-metadata`. Exported under an explicit name too, so the two gateways read
# symmetrically at the call site and neither is the implicit default.
resolve_cloudflare_subscription = resolve_subscription

__all__ = [
    "extract_cloudflare_log",
    "extract_databricks_log",
    "resolve_cloudflare_subscription",
    "resolve_databricks_subscription",
    "resolve_subscription",
]
