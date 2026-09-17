# Study 004 — give the context a larger unit of language

Protocol written before measuring the new subword model. This is an engineering
and learning diagnostic, not a claim of a novel tokenizer or interpretability method.

The previous character models saw only 51,200 training characters and had a
128-character context. Fragmented samples do not establish a failure of character
modeling. We now change tokenization and substantially increase training together;
any improvement cannot be attributed to tokenization alone.

## Data and vocabulary

Fit deterministic word-boundary byte-pair encoding on the same 2,097 training
stories only. Retain the exact 32 calibration and 152 evaluation stories. Use 96
ASCII base characters, explicit beginning/end-of-story tokens, and 3,998 learned
merges for a vocabulary of 4,096. Preserve original story boundaries, including
internal blank lines. Reject unsupported prompt characters explicitly.

Record tokenizer identity, merge table, source hashes, exact story offsets and
the binary corpus hash. Verify decoding every story reproduces its source and
rebuilding the vocabulary from training text reproduces the published vocabulary.
The procedure builds on the word-boundary BPE explored in Jaxverse. BPE itself is
established; see [the Hugging Face tokenizer documentation](https://huggingface.co/docs/tokenizers/api/trainers).

## First specimen

Train a decoder with four layers, width 128, four attention heads, 512 ReLU MLP
channels per layer, 128 subword positions, batch size four, Adam, and seed 42.
Its 2,048 plotted points are actual MLP channels. The initial target is 4,096
updates, with measurements at 0, 256, 1,024, and 4,096 updates. Count supervised
nonpadding targets exactly; report compute slots separately where needed. If
browser performance prevents the full budget in this session, retain the exact
achieved step, checkpoints and reason without calling the run complete.

Train within stories, include EOS targets, and mask right padding from the loss.
Use fixed held-out windows and a train-only smoothed unigram baseline on precisely
the same scored targets. Loss is nats per subword token; it is not directly
comparable with nats per character in Study 003. No repeated-seed or scaling claim
is licensed by this single run.

Sample at the declared capture steps from `Once upon a time`, using seed 71,
temperature 0.8, top-k 40 and a maximum of 128 generated tokens. Preserve every
sample, including failures and EOS termination. Sampling and probing must not
change parameters, optimizer state or the training random stream.

## Measurement and inspection

Capture every channel on eight fixed calibration windows at 16 token positions
each. Record exact token IDs as well as decoded text. Reuse the existing centered,
normalized fingerprint PCA and fixed 128-focal, six-neighbor retention audit.
Display projection distortion alongside the map. Semantic interpretation requires
additional evidence; proximity alone is an activation correlation measurement.

Let the user inspect a prompt token by token, select a layer/channel, read its
activation trace, inspect the 4,096-way next-token distribution, and compare an
all-positions single-channel lesion against the intact model. Record both raw
distributions. Treat effects near floating-point variation as unresolved.

Retain complete resumable parameters, Adam moments, training RNG state, tokenizer,
metrics, maps, samples and observations in a distinct versioned binary archive.
Keep the character lab and its historical records available. Publish evidence and
limitations in the lab's research journal.
