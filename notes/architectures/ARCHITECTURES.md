# Sequence architectures for CS2 round win probability: why each one, and what it's good and bad at

Status as of 6 Nov 2026. Every log-loss number is validation, final step, on 1,168 matches (1,515,228 seconds). The GDT grader has not yet been run on the GDT-trained models, so no GDT ranking exists yet.

Sources: the trainer `scripts/model/train_phase1_pos24.py` (line numbers cited), the Meridian pins (IDs cited), and the metrics files on the network volume. Parameter counts are sequence-module only, at the pos24 input width of 66 (the position-feature setting). Where a pin or docstring gives a different width, that's noted.

## 1. What every arm sees

- **Per-second state:** 10 shared values (clock and bomb state, map one-hot), 6 state values per player slot, and 12 loadout values per slot (pin 6fb5578a).
- **Position features:** 24 train-only features (`--dist`, `N_DIST = 24`, line 53). They were fitted on training matches only, so they don't leak validation or test information.
- **Lookback:** 20 seconds. Each prediction reads the 20 seconds ending at the current one.
- **Head:** a small MLP, d→64→1, with a sigmoid, trained on binary cross-entropy with logits.
- **Training:** Adam at lr 1e-3, no weight decay, no dropout, batch 2048, 30,000 steps. Validation every 1,000 steps. The reported number is the final step, with no checkpoint selection (pins 6fb5578a, 66bca815, 594b4126).
- **Seeds:** `torch.manual_seed(seed)` plus a seeded NumPy permutation for batch order.
- **Objective:** log loss for the original runs. The 6 Nov run set used the GDT-equivalent objective, pin 064d46c76 and amendment 079f7189.

Why a shared setup: pin 66bca815 requires "the same inputs and budget," so the comparison is about the architecture, not the inputs.

## 2. How the arm list was built

| Date (pin) | What was added | Why |
|---|---|---|
| 28 Oct (6fb5578a) | GRU control, GRU with lookback 1, LSTM, TCN, causal transformer | Does any sequence architecture beat the GRU? Is the GRU's gain from 20 seconds of memory, or from its per-player features? |
| 30 Oct (594b4126) | Same families at width 128 | Are the results sensitive to size? |
| 4 Nov (66bca815) | Seven-arm screen with position inputs, about 25k parameters | Rank compact recurrent and state-space designs at matched budget |
| 6 Nov (7fe1d12b) | minGRU, minLSTM, LRU; approval of xLSTM, RWKV-7, Griffin, Mamba-2 | The user approved a wider screen: "find the best black box model" |
| 6 Nov (06cabe74) | xLSTM 7B-style block, sigmoid-gate xLSTM, gdn2 | Newer variants from the literature survey |
| 6 Nov (9d24499e) | Gated DeltaNet, two-head xLSTM | Delta-rule memory; multi-head form |
| 6 Nov (14a13a0e) | 2-layer GRU and 2-layer transformer at about 100k parameters | Capacity track: do larger models help? |
| 6 Nov (ea145e3b, 31f6825b) | Round-2 rule: the top three by seed-0 loss go to five seeds | Seed-0 rankings are too noisy to pick winners on their own |

## 3. Each architecture

### 3.1 GRU (control)
- **What it is:** a gated recurrent unit with two gates (reset and update), hidden size 64, last-step output (`GRUSeq`, lines 75–83).
- **Why it's in:** it's the reference arm for every comparison (6fb5578a, 66bca815). Its status as the control is why every other arm is measured against it.
- **Parameters:** 25,344 with position inputs. 20,736 at width 42 (pin 594b4126).
- **Advantages:** simple, stable, and cheap. It beat the 57-input XGBoost on validation on every seed (pin 921fa3dd).
- **Limitations:** it's sequential over time, so training can't be parallelised across the window. Its hidden size of 64 is a choice pinned only as "parameter-matched," not tuned.
- **Results:** five seeds, log loss 0.451061. GDT-trained, 0.450392.

### 3.2 GRU with lookback 1 (history control)
- **What it is:** the same GRU, reading only the current second (`gru_lb1`, dispatch at line 506).
- **Why it's in:** it asks whether the GRU's gain comes from memory of the last 20 seconds, or from its per-player features alone (6fb5578a, docstring line 7).
- **Advantages:** isolates the value of history.
- **Limitations:** a control, not a candidate. No five-seed result was used for selection.

### 3.3 LSTM
- **What it is:** a long short-term memory cell with input, forget and output gates plus a cell state, hidden size 53 (`LSTMSeq`, lines 86–94).
- **Why it's in:** arm A2 of phase 1 (6fb5578a) and arm B of the screen (66bca815). The docstring says it's "parameter-matched to the GRU within about 1 percent."
- **Parameters:** 25,652 at width 66, which is 1.2% above the GRU. 20,564 at width 42.
- **Advantages:** a separate cell state can hold information longer than a single gated hidden state. Standard and well understood.
- **Limitations:** more gates than a GRU for about the same parameter budget. Sequential over time.
- **Results:** seed 0 only, log loss 0.451504. GDT-trained, 0.450286.

### 3.4 TCN (temporal convolutional network)
- **What it is:** four residual causal blocks with kernel 3 and dilations 1, 2, 4 and 8, after a projection to 40 values (`TCNSeq`, lines 107–116, with `CausalBlock` at lines 97–104). Last-step readout.
- **Why it's in:** arm A3 of phase 1 (6fb5578a). The receptive field is 31 seconds, which covers the 20-second window (docstring line 9).
- **Parameters:** 22,040 measured in the metrics. A pin estimate was about 21k.
- **Advantages:** convolutions parallelise over time during training, and the receptive field is fixed and known.
- **Limitations:** the receptive field is fixed by the dilation schedule. It doesn't adapt to the input the way attention or gating does.
- **Results:** five seeds, log loss 0.450735, the weakest of the five-seed arms on log loss. GDT-trained, 0.450480. Seed-0 log loss ranged 0.44996 to 0.45128.

### 3.5 Causal transformer (`tf`)
- **What it is:** a pre-norm causal transformer with learned positions, 2 layers, 4 heads, FFN width 64, and a causal mask with padding (`TFSeq`, lines 137–156; `TFBlock`, lines 119–134).
- **Why it's in:** arm A4 of phase 1 (6fb5578a). Pin 594b4126 notes it "may be under-tuned" at lr 1e-3.
- **Warm-up:** 500 steps, applied only when the arm is `tf` (line 707).
- **Parameters:** 19,936.
- **Advantages:** attention can weigh any second in the window directly, and training parallelises across the window.
- **Limitations:** the pins describe it as possibly under-tuned. Attention cost grows with window length, which is fine at 20 but would matter for longer lookbacks. Usually needs more data or careful tuning than the recurrent arms.
- **Results:** seed 0 only. Log loss 0.450372 (log-loss-trained), 0.450274 (GDT-trained).

### 3.6 Transformer, capacity track (`tf2`)
- **What it is:** the same transformer at width 76, with 2 layers, 4 heads and FFN 152 (line 535).
- **Why it's in:** capacity track, pin 14a13a0e: "Do deeper, larger sequence models beat the 25k-parameter arms on validation log loss when the inputs and training protocol are unchanged?"
- **Parameters:** 100,852.
- **Limitations:** **no warm-up.** Pin 14a13a0e says 500 steps, but the code applies warm-up only to arm `tf` (line 707). This is a documented deviation from the pin. See section 5.
- **Results:** seed 0 only. Log loss 0.450676 (log-loss-trained), 0.450391 (GDT-trained). It didn't beat the 25k transformer.

### 3.7 Two-layer GRU (`gru2`)
- **What it is:** two stacked GRU layers, width 94, last-step output (`GRUStackSeq`, lines 487–497).
- **Why it's in:** capacity track, pin 14a13a0e.
- **Parameters:** 99,264.
- **Advantages:** more capacity with the same recurrent structure as the control.
- **Limitations:** the larger model didn't help. Log loss 0.453934, the worst of the seed-0 arms, and 0.451261 GDT-trained.

### 3.8 minGRU
- **What it is:** a GRU simplified so the update and candidate depend only on the input, not on the previous hidden state (`MinGRUSeq`, lines 159–175, hidden 189). Update gate z and candidate g(h̃), with h = (1−z)h + z·g.
- **Why it's in:** arm C of the screen (66bca815, 7fe1d12b). The code comment says "gates depend only on the input, so the recurrence is a simple linear update" (line 160).
- **Parameters:** 25,326 at width 66.
- **Advantages:** because the recurrence is linear in the hidden state, it can in principle be computed with a parallel scan instead of a sequential loop.
- **Limitations:** **our implementation runs a Python loop over the lookback**, so the parallel advantage isn't realised here. The speed benefit is theoretical for this code. Simplified gates can also lose expressiveness.
- **Results:** five seeds. Log loss 0.450138 (log-loss-trained), 0.450274 (GDT-trained). This is the second-best five-seed log-loss mean, behind the 7B block, and it's within seed noise of the others.

### 3.9 minLSTM
- **What it is:** an LSTM with input-only forget and input gates normalised to sum to one, and a linear candidate (`MinLSTMSeq`, lines 178–196, hidden 126).
- **Why it's in:** arm D of the screen (66bca815, 7fe1d12b).
- **Parameters:** 25,326 at width 66.
- **Advantages:** same parallel-scan motivation as minGRU.
- **Limitations:** same as minGRU: our loop removes the speed benefit. Seed 0 only.
- **Results:** seed 0 only. Log loss 0.450471 (log-loss-trained), 0.450843 (GDT-trained). The GDT-trained run was the weaker of the two.

### 3.10 LRU (linear recurrent unit)
- **What it is:** 80 complex diagonal states and 64 outputs, with magnitudes initialised between 0.9 and 0.999 (`LRUSeq`, lines 199–228). Stability comes from the exponential-decay parametrisation: the eigenvalue magnitudes are kept below one by construction.
- **Why it's in:** arm E of the screen (66bca815, 7fe1d12b), after Orvieto et al. (2023).
- **Parameters:** 25,184.
- **Advantages:** stable by construction, and a linear recurrence, so it can be computed in parallel with a scan.
- **Limitations:** a linear recurrence has less input-dependent gating than the gated models. Our implementation is sequential in the loop too.
- **Results:** seed 0 only. Log loss 0.450645 (log-loss-trained), 0.450301 (GDT-trained).

### 3.11 xLSTM, loop form (`xlstm`)
- **What it is:** an mLSTM with matrix memory C and normaliser n, exponential input gating, sigmoid forget gate in log space, and sigmoid output gate, with one head of width 94 (`MLSTMSeq`, lines 231–262). The matrix and normaliser update in a Python loop over the lookback (lines 252–258).
- **Why it's in:** arm H of the screen, pin 6a674e1e, extending 7fe1d12b and 66bca815. The user approved the wider screen in chat.
- **Parameters:** about 25.3k (docstring line 234). Shapes give 25,326.
- **Advantages:** matrix memory can store more associations per parameter than a vector gate. Exponential gating with a normaliser is designed for stable long-range storage.
- **Limitations:** a single head here. The docstring and pin 6a674e1e both note "the paper uses several heads." The loop form is the slowest xLSTM implementation.
- **Results:** seed 0 only. Log loss 0.449463, the best seed-0 result in the screen (c734a903). GDT-trained, 0.450955.

### 3.12 xLSTM, closed form (`xlstm_fast`)
- **What it is:** the same recurrence and weights as `xlstm`, evaluated in closed form at the last step with no time loop (`MLSTMFastSeq`, lines 349–385, docstring 349–353). The closed form weights each past step by a function of the forget and input gates.
- **Why it's in:** pin ea145e3b uses it for seeds 1–4. The closed form was checked against the loop form (3e-7 difference on CPU, one head).
- **Parameters:** 25,326, same shapes as `xlstm`.
- **Advantages:** no time loop, so it's faster to run. Equivalent to the loop form in the CPU check.
- **Limitations:** the required GPU reproduction of seed 0 (within 1e-4 of 0.449463) is reported as passed in the working record, but no pin records it. The seed-0 GPU value is 0.449532, which is 7e-5 from the loop run.
- **Results:** five seeds. Log loss 0.450123, GDT-trained 0.450441.

### 3.13 xLSTM, multi-head (`xlstm_mh`)
- **What it is:** two heads of width 47, each with its own memory and gates (`MLSTMMultiHeadSeq`, lines 388–391, a subclass of `MLSTMFastSeq`).
- **Why it's in:** arm M, pin 9d24499e, the paper's multi-head form.
- **Parameters:** 25,460.
- **Advantages:** separate heads can learn different memory patterns.
- **Limitations:** the two heads are narrower, so each sees less. Seed 0 only.
- **Results:** seed 0 only. Log loss 0.450916 (log-loss-trained), 0.450451 (GDT-trained). The per-head loop check gave a 5e-7 difference.

### 3.14 xLSTM 7B-style block (`xlstm_blk`)
- **What it is:** an mLSTM mixer of width 46 with one head, then RMS norm with a gain, then a SwiGLU MLP with hidden 92 and a residual connection. Read at the last lookback step (`XLSTMBlockSeq`, lines 432–448). Based on arXiv 2503.13427.
- **Why it's in:** arm P, pin 06cabe74, added after the first seed-0 results. It entered five seeds under the round-2 rule, which required a seed-0 loss below 0.450298.
- **Parameters:** 25,434.
- **Advantages:** the residual block and gated MLP are the standard recipe for deep networks of this kind, and the extra MLP adds nonlinearity beyond the mixer.
- **Limitations:** one block, read at the last step, with no stacking (the docstring and the pin both say so). A deeper stack wasn't tested.
- **Results:** five seeds. Log loss 0.449940, the best five-seed log-loss mean. GDT-trained, 0.450746, the weakest of the GDT-trained five-seed means on log loss. On the log-loss paired check against the GRU control, seed 1 was no better than the control.

### 3.15 xLSTM sigmoid-gate (`xlstm_sig`)
- **What it is:** the same closed-form mLSTM as `xlstm_fast`, but with a sigmoid input gate, so its log is `logsigmoid` (`MLSTMFastSeq` with `input_gate="sigmoid"`, lines 371–373). Based on arXiv 2503.14376, a TFLA variant.
- **Why it's in:** arm Q, pin 06cabe74. The pin describes it as "the same as xLSTM arm H except that the input gate is sigmoid."
- **Parameters:** 25,326.
- **Advantages:** a sigmoid gate is bounded, so it avoids the exponential gate's overflow risk.
- **Limitations:** a bounded input gate can't weight a single step very strongly, which may matter for sharp events.
- **Results:** five seeds. Log loss 0.450337 (log-loss-trained), 0.450474 (GDT-trained). Pin 0d97f8f6 names it the single final candidate, selected on GDT validation under c807884d (pooled GDT difference +0.00131 against the GRU control, with a positive situation-weighted estimate). The holdout read is pre-registered and has not run.

### 3.16 RWKV-7
- **What it is:** a one-head WKV delta-rule state with decay between 0.545 and 1, and a LayerNorm readout (`RWKV7Seq`, lines 265–288, width 63).
- **Why it's in:** arm I, pin 6580d4fc, added in the 6 Nov extension (7fe1d12b).
- **Parameters:** 25,452.
- **Advantages:** a delta-rule update that can overwrite stored associations, rather than only accumulating them.
- **Limitations:** the pin says the implementation is simplified: **no token shift, no value residual, no bonus term, one head.** It's not a full RWKV-7.
- **Results:** five seeds. Log loss 0.450168 (log-loss-trained), 0.450730 (GDT-trained). Seed 0 was 0.450108 on log loss, behind the xLSTM variants.

### 3.17 Griffin
- **What it is:** an RG-LRU gated linear recurrence with a GeLU branch, width 83 (`GriffinSeq`, lines 291–317).
- **Why it's in:** arm J, pin 4025d97d, from the 6 Nov extension.
- **Parameters:** 25,149.
- **Advantages:** gated linear recurrence is cheap per step.
- **Limitations:** the pin notes dense gates where the paper uses block-diagonal ones, and no temporal convolution. The paper's local-attention component isn't in this version.
- **Results:** seed 0 only. Log loss 0.452294 (log-loss-trained), the weakest seed-0 result among the screen's recurrent arms. GDT-trained, 0.450490.

### 3.18 Mamba-2
- **What it is:** a selective state-space block with input-dependent scalar decay, state size 16 and a width-4 convolution. Implemented as a pure-PyTorch sequential scan (`Mamba2Seq`, lines 320–346, width 97).
- **Why it's in:** arm K, pin d47a24f1, from the 6 Nov extension.
- **Parameters:** 25,203.
- **Advantages:** input-dependent decay lets the state forget selectively. It runs on the 3080 and 4090 without a custom kernel build.
- **Limitations:** no chunked SSD kernel (lines 321–322), so the sequential scan is slow. Speed was not measured against the chunked version.
- **Results:** seed 0 only. Log loss 0.450642 (log-loss-trained), 0.450182 (GDT-trained).

### 3.19 Gated DeltaNet (`gdn`)
- **What it is:** one head with scalar decay and write gates, L2-normalised queries and keys, a delta-rule state update, RMS norm and a SiLU gate (`GatedDeltaNetSeq`, lines 394–429, width 73).
- **Why it's in:** arm L, pin 9d24499e, a 25k-track comparison against the GRU control (25,344).
- **Parameters:** 25,175.
- **Advantages:** delta-rule memory can overwrite stored associations, and gated decay adds forgetting.
- **Limitations:** scalar gates per head, so less per-channel control than the channel-gated variant.
- **Results:** seed 0 only. Log loss 0.450431 (log-loss-trained), 0.450475 (GDT-trained).

### 3.20 Channel-gated Gated DeltaNet (`gdn2`)
- **What it is:** per-channel decay and write gates instead of scalar ones (`ChannelGatedDeltaSeq`, lines 451–484, width 55).
- **Why it's in:** arm R, pin 06cabe74, from the survey of abstracts.
- **Parameters:** about 25.2k per the code comment; pin 9d24499e gives 25,245.
- **Advantages:** finer control over which channels forget and which store.
- **Limitations:** the code comment says the exact equations were not retrieved from the paper body. The arm's design is taken from the survey abstracts, so it's not a faithful reproduction. Pin 31f6825b records it as not qualifying for five seeds.
- **Results:** seed 0 only. Log loss 0.450955 (log-loss-trained), 0.450035 (GDT-trained), the best GDT-trained seed-0 result among the delta-rule arms.

## 4. Results at a glance

Validation final-step log loss. Five-seed means for the seven arms that got five seeds. Seed-0 values for the rest. GDT-trained is the 6 Nov run set. Log-loss-trained is the earlier set.

| Arm | Type | Params | Seeds | Log-loss-trained | GDT-trained |
|---|---|---|---|---|---|
| xLSTM 7B block | xLSTM | 25,434 | 5 | 0.449940 | 0.450746 |
| xLSTM closed form | xLSTM | 25,326 | 5 | 0.450123 | 0.450441 |
| minGRU | gated | 25,326 | 5 | 0.450138 | 0.450274 |
| RWKV-7 | delta/WKV | 25,452 | 5 | 0.450168 | 0.450730 |
| xLSTM sigmoid-gate | xLSTM | 25,326 | 5 | 0.450337 | 0.450474 |
| TCN | conv | 22,040 | 5 | 0.450735 | 0.450480 |
| GRU control | gated | 25,344 | 5 | 0.451061 | 0.450392 |
| xLSTM loop | xLSTM | 25,326 | 1 | 0.449463 | 0.450955 |
| Transformer | attention | 19,936 | 1 | 0.450372 | 0.450274 |
| Transformer (capacity) | attention | 100,852 | 1 | 0.450676 | 0.450391 |
| minLSTM | gated | 25,326 | 1 | 0.450471 | 0.450843 |
| LRU | linear | 25,184 | 1 | 0.450645 | 0.450301 |
| GatedDeltaNet | delta | 25,175 | 1 | 0.450431 | 0.450475 |
| Channel-gated GDN | delta | 25,245 | 1 | 0.450955 | 0.450035 |
| Multi-head xLSTM | xLSTM | 25,460 | 1 | 0.450916 | 0.450451 |
| Mamba-2 | SSM | 25,203 | 1 | 0.450642 | 0.450182 |
| LSTM | gated | 25,652 | 1 | 0.451504 | 0.450286 |
| Griffin | gated linear | 25,149 | 1 | 0.452294 | 0.450490 |
| 2-layer GRU | gated | 99,264 | 1 | 0.453934 | 0.451261 |

**Reading this table:** across the five-seed arms, the spread on log loss is about 0.0005 for the GDT-trained runs and about 0.0011 for the log-loss-trained runs. Seed noise is about 0.0005. So no arm is separated from the others on log loss. The GDT grader is what decides the ranking, and it hasn't been run yet.

## 5. Discrepancies and gaps

1. **Transformer warm-up.** Pin 14a13a0e says the capacity transformer uses 500 warm-up steps, as the 25k transformer does. The code applies warm-up only to arm `tf`, so `tf2` ran with none. The result is recorded as run, and the deviation is not yet noted in a pin.
2. **Trainer header.** The docstring (lines 21–24) describes four distance features, a 14-value state and a 46-wide encoder output. The code uses `N_DIST = 24`. The header is stale.
3. **Defaults versus runs.** The code's defaults are 12,000 steps and log-loss training. Every run used 30,000 steps through command-line flags. The 6 Nov run set also set the GDT objective flag.
4. **Parameter counts depend on input width.** Pins quote counts at width 42 (6fb5578a) and at width 66 (the pos24 setting). Use the metrics-file counts for the real models.
5. **Reproduction check.** Pin ea145e3b requires the closed-form xLSTM to reproduce the loop run's seed-0 value (0.449463) within 1e-4 on the GPU. The working record says the check passed (0.449532). No pin records it.
6. **Membership changed as arms were added.** At 06:16 on 6 Nov, the seed-0 ranking put minGRU third (pin ea145e3b). xLSTM 7B block and sigmoid-gate xLSTM were then added later under the same rule, with the threshold 0.450298 set to minGRU's value. The round-2 set ended as xLSTM closed form, 7B block and sigmoid-gate. RWKV-7 was ahead of minGRU on seed 0 (0.450108 against 0.450298) but was not promoted.
7. **Training objective.** The log-loss-trained results and the GDT-trained results come from different objectives. The 6 Nov amendment (079f7189) makes the GDT-equivalent objective the training target, per the user's report of Dr Freeman's instruction. Written confirmation is pending.
8. **Unverified implementations.** The gdn2 equations were not retrieved from the paper body (code comment). The Griffin version is missing the local-attention component and uses dense gates. The RWKV-7 version is simplified. Mamba-2 is a sequential pure-PyTorch scan, not the chunked kernel.
9. **Speed claims are untested.** The minGRU and minLSTM parallel-scan advantage is not realised by our Python loop. The closed-form speedup for xLSTM was not timed. No speed comparison is pinned.
10. **Single seeds.** Thirteen arms have one seed. Comparisons between single-seed arms are not reliable on their own, and the round-2 rule exists for this reason.
11. **Holdout use.** The 83-input XGBoost holdout read (pin 56316f69) happened once, before the GDT decision. The 57-input model's test-split log loss informed its own selection (pin 64b9d52d item 5b, disclosed).

## 6. Where the evidence lives

- Pins: 6fb5578a, 594b4126, 66bca815, 7fe1d12b, 6a674e1e, 6580d4fc, 4025d97d, d47a24f1, 9d24499e, 06cabe74, ea145e3b, 14a13a0e, 31f6825b, c734a903, b8a83b32, c807884d, 64d46c76, 079f7189.
- Log-loss-trained metrics and predictions: laptop `data/diagnostics/pos24_*`, and the network volume under `results/pos24_*` and `results/local_laptop/`.
- GDT-trained metrics and predictions: the network volume under `results/pos24_gdt_p1` to `pos24_gdt_p5` and `pos24_gdt_p4b`.
- GDT grader and the selection outputs: `data/diagnostics/gdt_seq/`.
