---
title: Selena Insights
emoji: \U0001F4B8
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

Personal transaction anomaly detection.

API: `POST /score` with `{"transactions": [{"date": "2026-01-01", "amount": 20000}]}`.
Dashboard: `/`. The demo runs on synthetic rows only.

## Status: blocked on Hugging Face pricing (2026-10-27)

Creating this Space returns **HTTP 402 Payment Required**:

> Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic
> requires a PRO subscription.

The files here are still correct and will build as-is on a PRO account or on any Docker host.
For a free public URL use the Render blueprint at `../../render.yaml`, which builds this same
Dockerfile with `dockerContext: .`.
