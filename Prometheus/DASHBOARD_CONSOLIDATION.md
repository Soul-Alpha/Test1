# Dashboard Consolidation Contract

The normal deployment surface is three Streamlit applications. Existing
dashboard scripts remain compatibility pages and are executed lazily by
`st.navigation`; they are not launched as independent processes.

| Command centre | Port | Compatibility pages |
|---|---:|---|
| Prometheus Trading Command Center | 8501 | Market Analysis, Evolution, Execution Academy, Academy Health |
| Hermes Execution and Learning Center | 8503 | Execution Overview, Pattern Context, Return Intelligence, Academy |
| Olympus Governance and Research Center | 8511 | Knowledge Growth, Zeus Validation, Olympus Observability |

## Invariants

- Prometheus remains the primary trading interface and default page on 8501.
- Hermes and Prometheus data/model identities remain separate.
- Zeus validation is visible within Olympus but cannot create operator approval.
- Compatibility pages are retained until metric parity is explicitly accepted.
- Dashboard services are read-only with respect to trading controls and orders.
- A missing or invalid artifact is not represented as empty historical data.
- Expensive builders run only for the selected page; command-centre routers do
  not construct IDIP, evolution, academy, or Hermes analytics.

## Retired normal-process ports

Ports 8502 and 8504–8510 are no longer launched by `start_all.ps1`. They remain
available for manual compatibility testing during the parity period.
