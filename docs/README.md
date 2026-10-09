# Strata in POM

This plugin lets a machine serve a whole **Qwen3.8-Flash-Next** model
(GGUF architecture `qwen4exp`, including its Coder and Unsloth files) with the
[Strata](https://github.com/Niko1221/Strata) engine instead of llama.cpp.
Strata keeps the dense part of the model and the most used experts on the GPU
and the rest of the experts in system RAM, so a model of about 80 GB runs on a
machine with one or more consumer cards and enough RAM.

## How a deployment uses it

1. Install the plugin on every machine that should be able to run Strata. The
   machine then shows the **Strata** inference service on the dashboard.
2. Open **Deploy**, choose the model, and in **Inference service** pick
   **Strata**. Only `qwen4exp` models and only machines with this plugin are
   offered. The deployment is a whole model on one machine, or the two layer
   slices of a model on two machines (see "A model split across two
   machines").
3. The settings of the deployment apply: the context size, the GPUs of the
   machine (one or more: Strata splits the layers between them), the
   concurrent requests and the KV cache type.
4. Deploy. The POM does not start llama.cpp on that machine: it asks this
   plugin to load the model. The first time a model is used the plugin
   prepares it for Strata (a few seconds, about 1.5 GB written in the plugin
   directory). Then Strata reads the weights, which takes a minute or two.
5. The model is served like any other: the Chat, `/v1/chat/completions`,
   `/v1/messages`, the request logs and retrieval all work, because the POM
   gateway renders the prompt as it does for a llama.cpp pipeline and sends it
   to Strata, which streams the tokens back.
6. **Undeploy** stops Strata and gives the GPU and the RAM back.

The plugin screens show the engine state and Strata's log (**Strata, Status**)
and Strata's own web app with its live GPU, speed and cache monitor
(**Strata, Monitor**). Both are for the administrator.

## Speculative decoding (MTP)

Strata writes about twice as fast with its MTP draft layer. The plugin uses
one when it finds a prepared draft layer (`mtp/rt/experts.bin`, the layout
Strata's own setup writes) in the plugin directory or up to three folders
above the model file. Without one Strata runs without speculation.

## A model split across two machines

Choose **Slices** with the Strata service to run a model that one machine
cannot hold on two: each machine loads its POM slice in Strata. The first
slice (from layer 0) runs as a stage server: for every prompt piece and every
draft window it answers with the hidden state of its last layer instead of
tokens. The second slice (to the last layer) serves the requests, with the
head, the sampling and the MTP draft layer, and asks the first machine for
those hidden states. The POM carries them between the machines, signed, over
the route it already uses between its nodes: neither Strata listens beyond
loopback. Every request starts from position 0 on both machines (no prompt
cache in this mode). Slices made by the POM's slicer load as they are.

## Where things live on a node

- `<POM database dir>/strata/packs/<model>` - the model prepared for Strata.
- `<POM database dir>/strata/run/strata-server.log` - Strata's server log.
- `<POM database dir>/strata/runtime/` - the downloaded Strata runtime of a
  release build.

## Limits

- A model runs on one machine, or split in two slices on two machines (one
  GPU each); three or more stages are not supported.
- Strata runs on NVIDIA cards of compute capability 7.5 or newer; this build
  carries code for sm_86 and sm_120 (the POM default).
- Images are not sent to Strata.
