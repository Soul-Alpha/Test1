# Operational Scripts

This directory contains governed operational scripts for Prometheus.

## Governance rules

1. **Every script here must have an entry in this README** describing its
   purpose, safe invocation, and idempotency status.
2. **No script may be run in a paper-trading or production environment without
   operator review and explicit approval.**
3. **Root-level hotfix scripts are prohibited.**  If an emergency fix requires
   a one-off script, create it here, document it, open a PR, and archive it
   under `scripts/archive/` once confirmed applied.

## Active scripts

*(none yet — add entries here as governed scripts are introduced)*

## Archived scripts

Legacy patch scripts from the repository root have been moved to
[scripts/archive/](archive/README.md) with a full provenance inventory.
Review and resolve those entries before introducing new scripts that touch
the same code paths.
