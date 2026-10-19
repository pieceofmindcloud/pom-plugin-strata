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

The **Strata** screen, for the administrator, shows the engine state and
Strata's log. While a model is served it has two tabs: **Strata**, Strata's
own web app with its live GPU, speed and cache monitor, and **Status**.

## Speculative decoding (MTP)

Strata writes about twice as fast with its MTP draft layer. The plugin uses
one when it finds a prepared draft layer (`mtp/rt/experts.bin`, the layout
Strata's own setup writes) in the plugin directory or up to three folders
above the model file. Without one Strata runs without speculation.

## A model split across machines

Choose **Slices** with the Strata service to run a model that one machine
cannot hold on several: each machine loads its POM slice in Strata. Every
slice before the last runs as a stage server: for every prompt piece and every
draft window it answers with the hidden state of its last layer instead of
tokens; a middle slice first asks the slice before it for its own. The last
slice (to the last layer) serves the requests, with the head, the sampling and
the MTP draft layer, and asks the slice before it for those hidden states.

A machine with several GPUs runs its slice on all of them: Strata places the
slice's layers across the cards (`--layer-split auto`) and the first and last
card meet the machines before and after it.

A node whose GPUs are AMD runs the HIP runtime (ROCm, its libraries carried
in the runtime); a node with NVIDIA cards runs the CUDA one. One Strata
process runs one backend, so a machine with both kinds takes two POM nodes
(one per kind) and two slices, and the hidden state crosses between them like
between machines. The AMD node's container needs `/dev/kfd` and `/dev/dri`. The POM carries them between the machines, signed, over
the route it already uses between its nodes: neither Strata listens beyond
loopback. Slices made by the POM's slicer load as they are.

The POM's own settings for a split model apply here as for llama.cpp, in the
same section of the deploy inspector:

- **Prompt pieces in flight** (`prefill_max_inflight_chunks`): the second
  machine asks the first for that many prompt pieces ahead, so both read the
  prompt at once.
- **Hidden-state precision**: the hidden states cross between the machines in
  that precision (f16 by default; f32, f8, int4) and reach Strata as f32.

The prompt cache works across the two machines: the first keeps its own
checkpoints where the second keeps them, and a request continues on both from
the same tokens. The `reused` count of a request is then real; each machine's
Strata log says where it continued from ("continues after N tokens"). When
the first machine holds no state for those tokens, the request reads the
prompt from the start on both.

## Prefill on one machine, decode on another

Deploy the whole model on two machines and set **Inference role** in the POM
settings card of each: **Prefill only** on the machine that reads prompts,
**Decode only** (or both) on the one that answers. The POM never sends a
request to the prefill machine to answer. Before a conversation's turn runs on
the decode machine, the POM has the prefill machine read the prompt into a
batch slot and keep the conversation's session; the decode machine fetches
that session, restores it and reads only the rest, then writes the answer.

- A role keeps the sessions on (as **Save the KV cache** does) and the batch
  slots on (at least 2 concurrent requests), whatever those settings say.
- Every turn goes through the prefill machine. `POM_STRATA_PREFILL_MIN_TOKENS`
  in the prefill node's environment leaves to the decode machine the turns
  whose new part is shorter than that many tokens (sending the session costs
  about 118 MB whatever the prompt).
- What crosses is one session file per turn: about 118 MB of recurrent state
  plus the K/V of the attention layers (about 7.5 KB per token in q4_0). Both
  machines need the same model and KV type; anything that fails leaves the
  decode machine to read the prompt itself.
- Turns with images or a typed decision skip the split.

## Where things live on a node

- `<POM database dir>/strata/packs/<model>` - the model prepared for Strata.
- `<POM database dir>/strata/run/strata-server.log` - Strata's server log.
- `<POM database dir>/strata/runtime/` - the downloaded Strata runtime of a
  release build.

## Limits

- One Strata process runs one GPU backend: CUDA for NVIDIA cards of compute
  capability 7.5 or newer, HIP for the wave32 AMD families (RDNA2 to RDNA4,
  Strix Halo). A release carries both runtimes for those families; a
  development build only the architectures of `CUDA_ARCHITECTURES` /
  `HIP_ARCHITECTURES` (default sm_86 and sm_120, and gfx1101).
- Images are not sent to Strata.
