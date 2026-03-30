"""
Seq2Seq EN → RU  –  v3: Bahdanau Attention
===========================================
What changed vs homework-6:
  • Encoder  – now BIDIRECTIONAL; final hidden/cell merged with a linear layer
               so the decoder (which is unidirectional) still receives an
               (n_layers, B, hidden_dim) state.
  • Attention – Bahdanau additive attention:
                  e_t = v · tanh(W_enc·h_enc + W_dec·s_t)
                  α_t = softmax(e_t)
                  c_t = Σ α_t · h_enc
  • Decoder  – at every step concatenates the context vector c_t with the
               token embedding before feeding into the LSTM, and also
               concatenates c_t with the LSTM output before the final
               projection (gives the projection access to "what was attended").
  • Everything else (BLEU, beam search, training loop, CLI) is unchanged.

Usage:
    # Train
    python homework-7.py --data rus.txt --epochs 20

    # Evaluate BLEU + run beam-search comparison (after training)
    python homework-7.py --data rus.txt --eval --save best_model_v3.pt

    # Translate one sentence
    python homework-7.py --data rus.txt --translate "wake up"

    # Low-resource run (RTX 3050 6 GB)
    python homework-7.py --data rus.txt --epochs 15 --batch 64 --hidden 512
"""

import argparse
import math
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────
# 1.  CONSTANTS
# ─────────────────────────────────────────────
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

EMBED_DIM   = 256
HIDDEN_DIM  = 512          # per-direction in encoder; decoder uses same dim
ATTN_DIM    = 256          # attention projection dimension
N_LAYERS    = 2
DROPOUT     = 0.3
BATCH_SIZE  = 128
LR          = 0.001
CLIP        = 1.0
TF_RATIO    = 0.5
MAX_LEN     = 50
MAX_GEN     = 60


# ─────────────────────────────────────────────
# 2.  BLEU  (unchanged from homework-6)
# ─────────────────────────────────────────────

def get_ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def clipped_precision(hypothesis: list, reference: list, n: int) -> float:
    hyp_ngrams = get_ngrams(hypothesis, n)
    ref_ngrams = get_ngrams(reference, n)
    hyp_count = sum(hyp_ngrams.values())
    if hyp_count == 0:
        return 0.0
    clipped_matches = sum(min(c, ref_ngrams.get(ng, 0))
                          for ng, c in hyp_ngrams.items())
    return clipped_matches / hyp_count


def brevity_penalty(hypothesis: list, reference: list) -> float:
    c, r = len(hypothesis), len(reference)
    if c == 0:
        return 0.0
    return 1.0 if c >= r else math.exp(1 - r / c)


def bleu_score(hypothesis: list, reference: list, max_n: int = 4) -> float:
    bp = brevity_penalty(hypothesis, reference)
    if bp == 0.0:
        return 0.0
    log_avg = 0.0
    for n in range(1, max_n + 1):
        p = clipped_precision(hypothesis, reference, n)
        if p == 0.0:
            return 0.0
        log_avg += (1.0 / max_n) * math.log(p)
    return bp * math.exp(log_avg)


def corpus_bleu(hypotheses: list, references: list, max_n: int = 4) -> float:
    clipped_total = [0] * max_n
    hyp_total     = [0] * max_n
    for hyp, ref in zip(hypotheses, references):
        for n in range(1, max_n + 1):
            hyp_ngrams = get_ngrams(hyp, n)
            ref_ngrams = get_ngrams(ref, n)
            hyp_total[n - 1] += sum(hyp_ngrams.values())
            for ng, c in hyp_ngrams.items():
                clipped_total[n - 1] += min(c, ref_ngrams.get(ng, 0))
    c = sum(len(h) for h in hypotheses)
    r = sum(len(ref) for ref in references)
    if c == 0:
        return 0.0
    bp = 1.0 if c >= r else math.exp(1 - r / c)
    log_avg = 0.0
    for n in range(1, max_n + 1):
        if hyp_total[n - 1] == 0 or clipped_total[n - 1] == 0:
            return 0.0
        log_avg += (1.0 / max_n) * math.log(clipped_total[n - 1] / hyp_total[n - 1])
    return bp * math.exp(log_avg)


# ─────────────────────────────────────────────
# 3.  TEXT PREPROCESSING  (unchanged)
# ─────────────────────────────────────────────

def unicode_to_ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def normalize_en(s: str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_ru(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^а-яёА-ЯЁ\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_pairs(path: str, max_len: int = MAX_LEN):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            en = normalize_en(parts[0])
            ru = normalize_ru(parts[1])
            if not en or not ru:
                continue
            en_tok, ru_tok = en.split(), ru.split()
            if not en_tok or not ru_tok:
                continue
            if len(en_tok) > max_len or len(ru_tok) > max_len:
                continue
            pairs.append((en_tok, ru_tok))
    return pairs


# ─────────────────────────────────────────────
# 4.  VOCABULARY  (unchanged)
# ─────────────────────────────────────────────

class Vocabulary:
    def __init__(self, name: str):
        self.name = name
        self.word2idx = {PAD_TOKEN: PAD_IDX, SOS_TOKEN: SOS_IDX,
                         EOS_TOKEN: EOS_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def add_token(self, token: str):
        if token not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build(self, sentences):
        for sent in sentences:
            for tok in sent:
                self.add_token(tok)

    def encode(self, tokens):
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices):
        return [self.idx2word.get(i, UNK_TOKEN) for i in indices]

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────
# 5.  DATASET & DATALOADER  (unchanged)
# ─────────────────────────────────────────────

class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab: Vocabulary, tgt_vocab: Vocabulary):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.pairs[idx]
        src_ids = torch.tensor(self.src_vocab.encode(src_tokens), dtype=torch.long)
        tgt_ids = torch.tensor(
            [SOS_IDX] + self.tgt_vocab.encode(tgt_tokens) + [EOS_IDX],
            dtype=torch.long,
        )
        return src_ids, tgt_ids


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=False, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=False, padding_value=PAD_IDX)
    return src_padded, tgt_padded   # (src_len, B), (tgt_len, B)


# ─────────────────────────────────────────────
# 6.  MODEL  ← NEW: attention encoder-decoder
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Bidirectional LSTM encoder.

    Outputs:
      enc_out  : (src_len, B, hidden_dim * 2)  — all time-step hidden states,
                 used by attention at every decoder step.
      hidden   : (n_layers, B, hidden_dim)      — initial decoder hidden state
      cell     : (n_layers, B, hidden_dim)      — initial decoder cell state

    The bidirectional final states are merged with a learned linear projection:
        h_dec = tanh( W_h · [h_fwd ; h_bwd] )
    one per layer, so the decoder (unidirectional, hidden_dim) receives a
    correctly-shaped initial state regardless of n_layers.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Merge forward + backward final states → decoder initial state
        # Applied independently to each layer
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell   = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        # src: (src_len, B)
        embedded = self.dropout(self.embedding(src))            # (src_len, B, E)
        enc_out, (hidden, cell) = self.lstm(embedded)
        # enc_out : (src_len, B, hidden_dim * 2)
        # hidden  : (n_layers * 2, B, hidden_dim)  — bidirectional
        # cell    : (n_layers * 2, B, hidden_dim)

        # Reshape to (n_layers, 2, B, H) then concat fwd+bwd along last dim
        # hidden[-2] = last fwd layer, hidden[-1] = last bwd layer (for each layer)
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        cell   = cell.view(self.n_layers, 2, -1, self.hidden_dim)

        # Concatenate forward and backward: (n_layers, B, hidden_dim * 2)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        cell   = torch.cat([cell[:, 0],   cell[:, 1]],   dim=-1)

        # Project down to hidden_dim: (n_layers, B, hidden_dim)
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell   = torch.tanh(self.fc_cell(cell))

        return enc_out, hidden, cell
        # enc_out is kept for attention; hidden/cell initialise the decoder


class Attention(nn.Module):
    """
    Bahdanau (additive) attention.

    At decoder step t, given:
      dec_hidden : (B, hidden_dim)           — current decoder hidden state s_t
      enc_out    : (src_len, B, hidden_dim*2) — all encoder outputs

    Computes:
      energy = v · tanh( W_enc · enc_out + W_dec · s_t )
      α      = softmax(energy)                 shape: (B, src_len)
      context = Σ_i α_i · enc_out_i            shape: (B, hidden_dim*2)

    Parameters
    ----------
    hidden_dim : decoder hidden size (= encoder per-direction hidden size)
    attn_dim   : internal projection size (hyperparameter)
    """

    def __init__(self, hidden_dim: int, attn_dim: int):
        super().__init__()
        # enc_out has 2*hidden_dim channels (bidirectional)
        self.W_enc = nn.Linear(hidden_dim * 2, attn_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim,     attn_dim, bias=False)
        self.v     = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_out):
        """
        dec_hidden : (B, hidden_dim)           top-layer decoder hidden state
        enc_out    : (src_len, B, hidden_dim*2)

        Returns
        -------
        context    : (B, hidden_dim*2)
        alpha      : (B, src_len)              attention weights (for inspection)
        """
        src_len = enc_out.size(0)

        # Expand dec_hidden to match every source position
        # (B, hidden_dim) → (B, 1, hidden_dim) → (B, src_len, hidden_dim)
        dec_hidden_exp = dec_hidden.unsqueeze(1).expand(-1, src_len, -1)

        # enc_out: (src_len, B, 2H) → (B, src_len, 2H)
        enc_out_t = enc_out.permute(1, 0, 2)

        # Additive energy: (B, src_len, attn_dim)
        energy = torch.tanh(
            self.W_enc(enc_out_t) + self.W_dec(dec_hidden_exp)
        )

        # Score: (B, src_len, 1) → (B, src_len)
        score = self.v(energy).squeeze(-1)

        alpha = F.softmax(score, dim=-1)                       # (B, src_len)

        # Weighted sum of encoder outputs
        # alpha: (B, 1, src_len)  ×  enc_out_t: (B, src_len, 2H)
        context = torch.bmm(alpha.unsqueeze(1), enc_out_t)     # (B, 1, 2H)
        context = context.squeeze(1)                           # (B, 2H)

        return context, alpha


class Decoder(nn.Module):
    """
    Attention decoder.

    At every step:
      1. Look up token embedding: (B, embed_dim)
      2. Compute context vector via attention: (B, hidden_dim*2)
      3. Concatenate embedding + context → LSTM input: (B, embed_dim + hidden_dim*2)
      4. Run one LSTM step → output: (B, hidden_dim)
      5. Project [output ; context] → logits: (B, tgt_vocab_size)
         (giving the projection both "what the LSTM decided" and
          "what it was looking at" improves accuracy)
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 attn_dim, n_layers, dropout):
        super().__init__()
        self.attention = Attention(hidden_dim, attn_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        # Input to LSTM = embedding + context vector
        self.lstm = nn.LSTM(
            embed_dim + hidden_dim * 2,   # ← concatenated input
            hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
        )
        # Output projection: hidden + context → vocab
        self.fc_out = nn.Linear(hidden_dim + hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden, cell, enc_out):
        """
        token   : (B,)
        hidden  : (n_layers, B, hidden_dim)
        cell    : (n_layers, B, hidden_dim)
        enc_out : (src_len, B, hidden_dim*2)

        Returns
        -------
        logits  : (B, tgt_vocab_size)
        hidden  : (n_layers, B, hidden_dim)
        cell    : (n_layers, B, hidden_dim)
        alpha   : (B, src_len)               attention weights
        """
        # ── embedding ───────────────────────────────────────────────────────
        token    = token.unsqueeze(0)                          # (1, B)
        embedded = self.dropout(self.embedding(token))         # (1, B, E)

        # ── attention ───────────────────────────────────────────────────────
        # Use the top LSTM layer's hidden state as the query
        top_hidden      = hidden[-1]                           # (B, hidden_dim)
        context, alpha  = self.attention(top_hidden, enc_out)  # (B, 2H), (B, src)

        # ── LSTM step ───────────────────────────────────────────────────────
        # Concatenate embedding with context; both need a time-step dim
        rnn_input = torch.cat(
            [embedded, context.unsqueeze(0)], dim=-1
        )                                                      # (1, B, E + 2H)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        # output: (1, B, hidden_dim) → (B, hidden_dim)
        output = output.squeeze(0)

        # ── projection ──────────────────────────────────────────────────────
        logits = self.fc_out(
            torch.cat([output, context], dim=-1)
        )                                                      # (B, vocab_size)

        return logits, hidden, cell, alpha


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def forward(self, src, tgt, teacher_forcing_ratio=TF_RATIO):
        """
        src : (src_len, B)
        tgt : (tgt_len, B)  — includes <sos> at position 0
        """
        tgt_len, batch_size = tgt.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(tgt_len, batch_size, vocab_size, device=self.device)

        enc_out, hidden, cell = self.encoder(src)
        dec_input = tgt[0]                                     # <sos> tokens

        for t in range(1, tgt_len):
            logits, hidden, cell, _ = self.decoder(
                dec_input, hidden, cell, enc_out
            )
            outputs[t] = logits
            top1 = logits.argmax(dim=1)
            dec_input = tgt[t] if random.random() < teacher_forcing_ratio else top1

        return outputs

    # ── Greedy decoding ──────────────────────────────────────────────────────
    @torch.no_grad()
    def translate(self, src_tensor, max_len=MAX_GEN):
        """
        src_tensor : (src_len,)  — single sentence, no batch dim.
        Returns list of token indices (excluding special tokens).
        """
        self.eval()
        src = src_tensor.unsqueeze(1).to(self.device)          # (src_len, 1)
        enc_out, hidden, cell = self.encoder(src)

        dec_input = torch.tensor([SOS_IDX], device=self.device)
        generated = []
        for _ in range(max_len):
            logits, hidden, cell, _ = self.decoder(
                dec_input, hidden, cell, enc_out
            )
            pred = logits.argmax(dim=1)
            if pred.item() == EOS_IDX:
                break
            generated.append(pred.item())
            dec_input = pred
        return generated

    # ── Beam-search decoding ─────────────────────────────────────────────────
    @torch.no_grad()
    def beam_search(self, src_tensor, beam_width: int = 5, max_len: int = MAX_GEN):
        """
        Beam search with attention decoder.

        Each beam candidate carries:
            (cumulative_log_prob, token_sequence, hidden, cell)

        enc_out is shared across all beams (same source sentence).
        """
        self.eval()
        src = src_tensor.unsqueeze(1).to(self.device)          # (src_len, 1)
        enc_out, hidden, cell = self.encoder(src)

        beams     = [(0.0, [], hidden, cell)]
        completed = []

        for _ in range(max_len):
            if not beams:
                break

            all_candidates = []
            for log_prob, seq, h, c in beams:
                last_tok = torch.tensor(
                    [seq[-1] if seq else SOS_IDX], device=self.device
                )
                logits, h_new, c_new, _ = self.decoder(last_tok, h, c, enc_out)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)

                topk_log_probs, topk_ids = log_probs.topk(beam_width)
                for lp, tok_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    new_log_prob = log_prob + lp
                    new_seq      = seq + [tok_id]
                    if tok_id == EOS_IDX:
                        norm_score = new_log_prob / len(new_seq)
                        completed.append((norm_score, seq))
                    else:
                        all_candidates.append(
                            (new_log_prob, new_seq, h_new, c_new)
                        )

            all_candidates.sort(
                key=lambda x: x[0] / max(len(x[1]), 1), reverse=True
            )
            beams = all_candidates[:beam_width]

        if completed:
            completed.sort(key=lambda x: x[0], reverse=True)
            return completed[0][1]
        if beams:
            beams.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
            return beams[0][1]
        return []

    # ── Attention visualisation helper ───────────────────────────────────────
    @torch.no_grad()
    def attention_weights(self, src_tensor, max_len=MAX_GEN):
        """
        Return (generated_ids, alphas) where alphas is a list of
        (src_len,) numpy arrays — one per generated token — showing
        where the model looked at each step.
        """
        self.eval()
        src = src_tensor.unsqueeze(1).to(self.device)
        enc_out, hidden, cell = self.encoder(src)

        dec_input = torch.tensor([SOS_IDX], device=self.device)
        generated, all_alphas = [], []
        for _ in range(max_len):
            logits, hidden, cell, alpha = self.decoder(
                dec_input, hidden, cell, enc_out
            )
            pred = logits.argmax(dim=1)
            if pred.item() == EOS_IDX:
                break
            generated.append(pred.item())
            all_alphas.append(alpha.squeeze(0).cpu().numpy())  # (src_len,)
            dec_input = pred
        return generated, all_alphas


# ─────────────────────────────────────────────
# 7.  TRAINING HELPERS  (unchanged)
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        output_flat = output[1:].reshape(-1, output.shape[-1])
        tgt_flat    = tgt[1:].reshape(-1)
        loss = criterion(output_flat, tgt_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt, teacher_forcing_ratio=0.0)
        output_flat = output[1:].reshape(-1, output.shape[-1])
        tgt_flat    = tgt[1:].reshape(-1)
        epoch_loss += criterion(output_flat, tgt_flat).item()
    return epoch_loss / len(loader)


def epoch_time(start, end):
    e = end - start
    return int(e // 60), int(e % 60)


# ─────────────────────────────────────────────
# 8.  TRANSLATE HELPERS  (unchanged API, attention-aware internally)
# ─────────────────────────────────────────────

def tokens_from_sentence(sentence: str, src_vocab: Vocabulary):
    toks = normalize_en(sentence).split()
    return torch.tensor(src_vocab.encode(toks), dtype=torch.long) if toks else None


def ids_to_string(ids: list, tgt_vocab: Vocabulary) -> str:
    return " ".join(tgt_vocab.decode(ids))


def translate_greedy(sentence: str, model, src_vocab, tgt_vocab, device) -> str:
    src = tokens_from_sentence(sentence, src_vocab)
    if src is None:
        return ""
    return ids_to_string(model.translate(src), tgt_vocab)


def translate_beam(sentence: str, model, src_vocab, tgt_vocab, device,
                   beam_width: int = 5) -> str:
    src = tokens_from_sentence(sentence, src_vocab)
    if src is None:
        return ""
    return ids_to_string(model.beam_search(src, beam_width=beam_width), tgt_vocab)


# ─────────────────────────────────────────────
# 9.  BLEU VERIFICATION  (unchanged)
# ─────────────────────────────────────────────

def verify_bleu():
    hyp = "the match was postponed because of the snow".split()
    ref = "the match was postponed because it was snowing".split()
    score = bleu_score(hyp, ref)
    status = "✓ PASS" if abs(score - 0.516) < 0.005 else "✗ FAIL"
    print(f"\nBLEU verification  {status}")
    print(f"  hyp : {' '.join(hyp)}")
    print(f"  ref : {' '.join(ref)}")
    print(f"  BLEU: {score:.4f}  (expected ≈ 0.516)")
    for n in range(1, 5):
        print(f"  p_{n} = {clipped_precision(hyp, ref, n):.4f}")
    print(f"  BP  = {brevity_penalty(hyp, ref):.4f}")


# ─────────────────────────────────────────────
# 10. EVALUATION EXPERIMENT  (unchanged)
# ─────────────────────────────────────────────

def run_evaluation(model, val_pairs, src_vocab, tgt_vocab, device,
                   n_sentences: int = 100, beam_widths=(3, 5, 10)):
    model.eval()
    subset    = val_pairs[:n_sentences]
    configs   = [("greedy", None)] + [(f"beam-{w}", w) for w in beam_widths]
    results   = {}
    references = [p[1] for p in subset]

    for name, bw in configs:
        hypotheses = []
        t0 = time.time()
        for src_toks, _ in subset:
            src_str = " ".join(src_toks)
            pred = (translate_greedy(src_str, model, src_vocab, tgt_vocab, device)
                    if bw is None else
                    translate_beam(src_str, model, src_vocab, tgt_vocab, device, bw))
            hypotheses.append(pred.split())
        elapsed = time.time() - t0

        b1 = corpus_bleu(hypotheses, references, max_n=1)
        b2 = corpus_bleu(hypotheses, references, max_n=2)
        b4 = corpus_bleu(hypotheses, references, max_n=4)
        ms = (elapsed / n_sentences) * 1000

        results[name] = {"bleu1": b1, "bleu2": b2, "bleu4": b4,
                         "ms": ms, "hyps": hypotheses}
        print(f"  [{name:>8}]  BLEU-1={b1:.4f}  BLEU-2={b2:.4f}  BLEU-4={b4:.4f}"
              f"   {ms:.1f} ms/sent")

    header = f"\n{'Config':<12} {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-4':>8} {'ms/sent':>10}"
    print(header)
    print("-" * len(header))
    for name, _ in configs:
        r = results[name]
        print(f"{name:<12} {r['bleu1']:>8.4f} {r['bleu2']:>8.4f} "
              f"{r['bleu4']:>8.4f} {r['ms']:>10.1f}")

    print("\n" + "=" * 72)
    print("5 TRANSLATION EXAMPLES  (greedy vs beam-5 vs reference)")
    print("=" * 72)

    greedy_hyps = results["greedy"]["hyps"]
    beam5_hyps  = results["beam-5"]["hyps"]

    scored = [(i,
               bleu_score(gh, ref),
               bleu_score(bh, ref),
               bleu_score(bh, ref) - bleu_score(gh, ref))
              for i, (ref, gh, bh)
              in enumerate(zip(references, greedy_hyps, beam5_hyps))]
    scored.sort(key=lambda x: -x[3])

    selected = [s[0] for s in scored[:4]]
    greedy_wins = sorted([s for s in scored if s[3] < 0], key=lambda x: x[3])
    selected.append(greedy_wins[0][0] if greedy_wins else scored[-1][0])

    col = 22
    for rank, idx in enumerate(selected, 1):
        src_str = " ".join(subset[idx][0])
        ref_str = " ".join(references[idx])
        g_str   = " ".join(greedy_hyps[idx])
        b_str   = " ".join(beam5_hyps[idx])
        g_b     = bleu_score(greedy_hyps[idx], references[idx])
        b_b     = bleu_score(beam5_hyps[idx],  references[idx])
        tag     = "beam wins ↑" if b_b >= g_b else "greedy wins ↑"
        print(f"\nExample {rank}  [{tag}]")
        print(f"  {'Source:':<{col}} {src_str}")
        print(f"  {'Reference:':<{col}} {ref_str}")
        print(f"  {'Greedy:':<{col}} {g_str}  (BLEU={g_b:.3f})")
        print(f"  {'Beam-5:':<{col}} {b_str}  (BLEU={b_b:.3f})")


# ─────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────

def build_model(src_vocab, tgt_vocab, args, device):
    encoder = Encoder(len(src_vocab), args.embed, args.hidden,
                      args.layers, args.dropout)
    decoder = Decoder(len(tgt_vocab), args.embed, args.hidden,
                      args.attn_dim, args.layers, args.dropout)
    return Seq2Seq(encoder, decoder, device).to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Seq2Seq EN→RU v3: Bahdanau Attention"
    )
    parser.add_argument("--data",      default="rus.txt")
    parser.add_argument("--epochs",    type=int,   default=15)
    parser.add_argument("--batch",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--embed",     type=int,   default=EMBED_DIM)
    parser.add_argument("--hidden",    type=int,   default=HIDDEN_DIM)
    parser.add_argument("--attn_dim",  type=int,   default=ATTN_DIM,
                        help="Attention projection dimension")
    parser.add_argument("--layers",    type=int,   default=N_LAYERS)
    parser.add_argument("--dropout",   type=float, default=DROPOUT)
    parser.add_argument("--lr",        type=float, default=LR)
    parser.add_argument("--clip",      type=float, default=CLIP)
    parser.add_argument("--tf",        type=float, default=TF_RATIO)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--save",      default="best_model_v3.pt")
    parser.add_argument("--translate", default=None,
                        help="Translate one sentence (greedy + beam)")
    parser.add_argument("--eval",      action="store_true",
                        help="Load checkpoint and run BLEU/beam comparison")
    parser.add_argument("--n_eval",    type=int,   default=100)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    verify_bleu()

    # ── data ────────────────────────────────────────────────────────────────
    print(f"\nLoading data from '{args.data}' …")
    pairs = load_pairs(args.data)
    print(f"  {len(pairs):,} pairs after filtering (max_len={MAX_LEN})")

    random.shuffle(pairs)
    split = int(0.9 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    print(f"  Train: {len(train_pairs):,}  |  Val: {len(val_pairs):,}")

    src_vocab = Vocabulary("en")
    tgt_vocab = Vocabulary("ru")
    src_vocab.build(p[0] for p in train_pairs)
    tgt_vocab.build(p[1] for p in train_pairs)
    print(f"  EN vocab: {len(src_vocab):,}  |  RU vocab: {len(tgt_vocab):,}")

    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds   = TranslationDataset(val_pairs,   src_vocab, tgt_vocab)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          collate_fn=collate_fn)

    model = build_model(src_vocab, tgt_vocab, args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    print(f"  Architecture: bidirectional encoder + Bahdanau attention decoder")
    print(f"  hidden={args.hidden}  attn_dim={args.attn_dim}  layers={args.layers}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── eval / translate mode ─────────────────────────────────────────────
    if args.eval or args.translate:
        ckpt = Path(args.save)
        if not ckpt.exists():
            print(f"No checkpoint at '{args.save}'. Train first.")
            return
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)

        # Restore architecture dims from checkpoint (prevents size mismatch)
        saved_args = checkpoint.get("args", {})
        args.embed    = saved_args.get("embed",    args.embed)
        args.hidden   = saved_args.get("hidden",   args.hidden)
        args.attn_dim = saved_args.get("attn_dim", args.attn_dim)
        args.layers   = saved_args.get("layers",   args.layers)
        args.dropout  = saved_args.get("dropout",  args.dropout)

        src_vocab = checkpoint["src_vocab"]
        tgt_vocab = checkpoint["tgt_vocab"]
        model     = build_model(src_vocab, tgt_vocab, args, device)
        model.load_state_dict(checkpoint["model"])

        if args.translate:
            g = translate_greedy(args.translate, model, src_vocab, tgt_vocab, device)
            b = translate_beam(args.translate, model, src_vocab, tgt_vocab, device, 5)
            print(f"\nEN:      {args.translate}")
            print(f"Greedy:  {g}")
            print(f"Beam-5:  {b}")

            # Also print attention weights for the greedy pass
            src_ids = tokens_from_sentence(args.translate, src_vocab)
            if src_ids is not None:
                src_toks = normalize_en(args.translate).split()
                gen_ids, alphas = model.attention_weights(src_ids)
                gen_toks = tgt_vocab.decode(gen_ids)
                print("\nAttention weights (rows = generated tokens, "
                      "cols = source tokens):")
                col_w = max(len(t) for t in src_toks) + 2
                header = "".join(f"{t:>{col_w}}" for t in src_toks)
                print(f"{'':>12}{header}")
                for tok, a in zip(gen_toks, alphas):
                    row = "".join(f"{v:>{col_w}.2f}" for v in a)
                    print(f"  {tok:>10}{row}")
            return

        print(f"\nRunning evaluation on {args.n_eval} validation sentences …")
        run_evaluation(model, val_pairs, src_vocab, tgt_vocab, device,
                       n_sentences=args.n_eval)
        return

    # ── training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    demo = ["go", "help me", "run", "i know", "who won"]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss  = train_epoch(model, train_dl, optimizer, criterion,
                               args.clip, device)
        val_loss = evaluate_loss(model, val_dl, criterion, device)
        mins, secs = epoch_time(t0, time.time())

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({
                "model":     model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "args":      vars(args),
            }, args.save)

        marker = "✓" if improved else " "
        print(f"Epoch {epoch:03d}/{args.epochs} [{mins}m{secs:02d}s] {marker} "
              f"Train: {tr_loss:.4f}  Val: {val_loss:.4f}")

        if epoch % 5 == 0:
            print("  Quick translations:")
            for s in demo:
                g = translate_greedy(s, model, src_vocab, tgt_vocab, device)
                print(f"    '{s}'  →  '{g}'")
            model.train()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to '{args.save}'")

    # ── final translations ─────────────────────────────────────────────────
    checkpoint = torch.load(args.save, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    test_sents = ["go", "help me", "run", "stop it", "i know",
                  "who won", "wake up", "be brave", "i am old", "i will try"]
    print("\nFinal translations (greedy  |  beam-5):")
    for s in test_sents:
        g = translate_greedy(s, model, src_vocab, tgt_vocab, device)
        b = translate_beam(s, model, src_vocab, tgt_vocab, device, 5)
        print(f"  {s:<25}  greedy: {g:<30}  beam-5: {b}")

    print("\nQuick BLEU on first 200 val sentences:")
    n = min(200, len(val_pairs))
    hyps_g = [translate_greedy(" ".join(p[0]), model, src_vocab, tgt_vocab,
                                device).split()
              for p in val_pairs[:n]]
    refs = [p[1] for p in val_pairs[:n]]
    for max_n in (1, 2, 4):
        sc = corpus_bleu(hyps_g, refs, max_n=max_n)
        print(f"  BLEU-{max_n} (greedy): {sc:.4f}")


if __name__ == "__main__":
    main()