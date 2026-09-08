---
language:
  - pt
  - es
  - hi
license: apache-2.0
library_name: pytorch
pipeline_tag: text-generation
tags:
  - sft
  - instruction-tuning
  - multilingual
---

# pt-es-hi-sft

A 138M-parameter (205M including embeddings) decoder-only GPT for
Portuguese, Spanish and Hindi, instruction-tuned from a pretrained
checkpoint. Trained from scratch as a teaching/research exercise — it is
small, and the limitations below are the interesting part of this card.

**This is not a `transformers` model.** The architecture (learned
positional embeddings *and* qk-norm) matches no stock HF class, so
`AutoModel.from_pretrained` will not load it. Load it against the
[training repo](https://github.com/frankreyesgarcia/LLM-Assignment):

```python
import json, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from src.model.gpt import GPT, GPTConfig          # from the training repo
from transformers import AutoTokenizer

repo = "andre15silva/pt-es-hi-sft"
cfg = json.load(open(hf_hub_download(repo, "config.json")))
model = GPT(GPTConfig(**{k: cfg[k] for k in
    ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout", "bias", "qk_norm")}))
model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")), strict=False)  # lm_head is tied
model.eval()

tok = AutoTokenizer.from_pretrained(repo)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Olá! Como estás?"}], tokenize=False, add_generation_prompt=True)
ids = torch.tensor([tok(prompt, add_special_tokens=False)["input_ids"]])
print(tok.decode(model.generate(ids, max_new_tokens=80, temperature=0.7, top_k=40)[0]))
```

`strict=False` is required: `lm_head.weight` is tied to
`transformer.wte.weight` and is therefore not stored separately.
`GPT.__init__` re-ties them.

## Prompt format

Always use the chat template. The model was fine-tuned on it exclusively
and degrades on bare text, leaking `<|eot_id|><|start_header_id|>` into
its output. Turns end with `<|eot_id|>`, not the tokenizer's
`<|end_of_text|>` — stop generation on both.

## Training

| | |
|---|---|
| Base checkpoint | pretrained on pt/es/hi, 2.2e18 FLOPs, iteration 240,204 |
| Architecture | 11 layers, 16 heads, n_embd 1024, block_size 1024, qk-norm, tied embeddings |
| Vocabulary | 32,000 (SentencePiece-style BPE trained on the same pt/es/hi corpus) |
| SFT data | 158,834 conversations / 90.9M tokens |
| Schedule | 2 epochs (19,854 steps), batch 16, lr 5e-5 → 5e-6, warmup 397, bf16 |
| Hardware | 1× A100-40GB, 3h24m |
| Loss | val 2.723 → **1.606** (perplexity 4.98), still falling at the end |

Loss is computed on assistant tokens only; every other position is masked
to `-1`. One conversation per batch row, right-padded to the longest in
the batch, with length-grouped batching (98.4% of the padded tensor was
real tokens).

### Data mix

| source | tokens | share |
|---|---|---|
| `aya_hi` (templated headline + news) | 30.3M | 33.3% |
| `utter-project/EuroBlocks-SFT-2512` (es) | 17.8M | 19.6% |
| `utter-project/EuroBlocks-SFT-2512` (pt) | 14.1M | 15.5% |
| `duarteocarmo/smoltalk2PT` (4 configs) | 26.7M | 29.4% |
| `ai4bharat/indic-instruct-data-v0.1` (hh-rlhf/hi) | 1.8M | 2.0% |

By language: pt 44.9%, hi 35.3%, es 19.6%.

**48.7% of rendered conversations (154,049 of 316,124) were dropped** for
exceeding the 1,024-token context window — a hard ceiling from the base
model's learned positional embeddings. The filter is not uniform:
`magpie_ultra`, the multi-turn general-instruction source, lost 88% of
its rows; `smol_rewrite` lost 2.

## Evaluation

Base checkpoint vs this model, n=500 per task, zero-shot. "chat" is the
chat template; "raw" is plain ICL text.

| benchmark | metric | base·raw | SFT·raw | base·chat | **SFT·chat** |
|---|---|---|---|---|---|
| PT-Culture | token_f1 | 0.177 | 0.169 | 0.193 | **0.292** |
| ChatRAG-Hi | token_f1 | 0.285 | 0.304 | 0.262 | 0.233 |
| CALAME-PT | accuracy | 0.408 | **0.422** | — | — |
| Portugal Basic QA (n=50) | accuracy | 0.560 | 0.520 | 0.480 | 0.420 |
| Belebele-es | acc | 0.260 | 0.222 | 0.240 | 0.230 |
| COPA-es | acc | 0.556 | 0.526 | 0.526 | 0.522 |
| OpenBookQA-es | acc | 0.200 | 0.190 | 0.182 | 0.172 |
| XStoryCloze-es | acc | 0.532 | 0.516 | 0.514 | 0.514 |

Reading these honestly:

- **PT-Culture (+51% over base in chat mode) is the real gain**, and it
  beats the control: applying the chat template to the *base* model only
  reached 0.193, so most of the gain is the tuning, not the format.
- **CALAME-PT went up** (0.408 → 0.422), so fine-tuning did not damage raw
  language modelling — no catastrophic forgetting.
- **The multiple-choice tasks are flat by design.** They score by ranking
  fixed continuations by log-probability; SFT teaches format and stopping,
  not knowledge. Six of the eight tasks above are of this kind.
- **EsCoLA is excluded**: all four runs are degenerate (MCC ≈ 0 against
  accuracies swinging 0.32–0.69 on a ~2/3-imbalanced label, i.e. a
  near-constant predicted class).
- **ALBA, the open-ended instruction-following benchmark, was not scored**
  (no judge API key). It is the measurement most relevant to an SFT model
  and it is missing.

## Limitations

**It behaves like an assistant. It does not know very much.**

```
"Olá! Como estás?"            → "Olá! Estou bem, obrigado por perguntar.
                                 E você, como está? 😊"      ✓ replies, stops
"Qual é a capital de França?" → "A capital de França é Paris."  ✓
"Qual é a capital de Portugal?" → "A capital de Portugal é Porto de Ave."  ✗
```

Lisboa never appeared across five seeds. The base model *can* complete
`"Lisboa é a capital de …"` → `"Portugal"` correctly, but neither
checkpoint answers the question form — a reversal-curse failure inherited
from pretraining, not caused by SFT. Fine-tuning made confident
fabrication *more* fluent: wrong answers arrive in the same assured,
markdown-formatted register as correct ones.

Other known limitations:

- **Hindi is weak.** 35.3% of training tokens, but 94% of that is two
  templated generation tasks (headline→article, article→summary), not
  instruction following. Hindi chat replies are frequently incoherent
  where pt/es are clean, and ChatRAG-Hi is the one generative benchmark
  that got *worse*.
- **No English.** Neither pretraining nor SFT included it. English prompts
  are usually answered in Spanish or Portuguese.
- **1,024-token context**, hard-capped by learned positional embeddings.
- **No safety tuning of any kind.** No RLHF, no DPO, no refusal training,
  no red-teaming. Do not deploy this.
- **Untuned hyperparameters.** lr 5e-5 / 5e-6 is the conventional
  ~10×-below-pretraining starting point, not a swept optimum. Validation
  loss was still falling at 2 epochs, so the model is undertrained.

## Reproducing

```bash
uv run scripts/prepare_sft_data.py --tokenizer-dir <tokenizer> \
    --out-dir data/sft --max-example-tokens 1024
uv run scripts/train_sft.py --init-from <pretrain ckpt> --data-dir data/sft \
    --out-dir runs/sft --block-size 1024 --batch-size 16 --max-iters 19854 \
    --lr 5e-5 --min-lr 5e-6 --wandb
```

See `scripts/slurm/13_prepare_sft_data.sh` and `14_train_sft.sh` in the
training repo for the cluster versions with the derived iteration counts.
