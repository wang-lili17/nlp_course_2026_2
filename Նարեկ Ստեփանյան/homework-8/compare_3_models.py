"""
compare_models.py
=================
Loads checkpoints from homework-5, homework-6, and homework-7 and produces
a side-by-side BLEU comparison table (greedy + beam-5) across all three models.

Usage
-----
python compare_3_models.py --data rus.txt --ckpt5 homework-5.pt --ckpt6 homework-6.pt --ckpt7 homework-7.pt --n_eval 500

All three checkpoint files must be present; omit any --ckptN flag to skip
that model.
"""

import argparse
import math
import random
import re
import time
import unicodedata
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"; SOS_TOKEN = "<sos>"; EOS_TOKEN = "<eos>"; UNK_TOKEN = "<unk>"
PAD_IDX = 0; SOS_IDX = 1; EOS_IDX = 2; UNK_IDX = 3
MAX_LEN = 50; MAX_GEN = 60

# ─────────────────────────────────────────────────────────────────────────────
# 2.  BLEU
# ─────────────────────────────────────────────────────────────────────────────
def get_ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def clipped_precision(hyp, ref, n):
    hyp_ng = get_ngrams(hyp, n); ref_ng = get_ngrams(ref, n)
    total  = sum(hyp_ng.values())
    if total == 0: return 0.0
    return sum(min(c, ref_ng.get(ng, 0)) for ng, c in hyp_ng.items()) / total

def brevity_penalty(hyp, ref):
    c, r = len(hyp), len(ref)
    if c == 0: return 0.0
    return 1.0 if c >= r else math.exp(1 - r/c)

def bleu_score(hyp, ref, max_n=4):
    bp = brevity_penalty(hyp, ref)
    if bp == 0.0: return 0.0
    log_avg = 0.0
    for n in range(1, max_n+1):
        p = clipped_precision(hyp, ref, n)
        if p == 0.0: return 0.0
        log_avg += math.log(p) / max_n
    return bp * math.exp(log_avg)

def corpus_bleu(hyps, refs, max_n=4):
    clipped = [0]*max_n; total = [0]*max_n
    for hyp, ref in zip(hyps, refs):
        for n in range(1, max_n+1):
            hng = get_ngrams(hyp, n); rng = get_ngrams(ref, n)
            total[n-1]   += sum(hng.values())
            clipped[n-1] += sum(min(c, rng.get(ng,0)) for ng, c in hng.items())
    c = sum(len(h) for h in hyps); r = sum(len(ref) for ref in refs)
    if c == 0: return 0.0
    bp = 1.0 if c >= r else math.exp(1 - r/c)
    log_avg = 0.0
    for n in range(1, max_n+1):
        if total[n-1] == 0 or clipped[n-1] == 0: return 0.0
        log_avg += math.log(clipped[n-1]/total[n-1]) / max_n
    return bp * math.exp(log_avg)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def unicode_to_ascii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def normalize_en(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_ru(s):
    s = s.lower().strip()
    s = re.sub(r"[^а-яёА-ЯЁ\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def load_pairs(path, max_len=MAX_LEN):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2: continue
            en = normalize_en(parts[0]); ru = normalize_ru(parts[1])
            if not en or not ru: continue
            et, rt = en.split(), ru.split()
            if not et or not rt: continue
            if len(et) > max_len or len(rt) > max_len: continue
            pairs.append((et, rt))
    return pairs

# ─────────────────────────────────────────────────────────────────────────────
# 4.  VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2idx = {PAD_TOKEN:PAD_IDX, SOS_TOKEN:SOS_IDX,
                         EOS_TOKEN:EOS_IDX, UNK_TOKEN:UNK_IDX}
        self.idx2word = {v:k for k,v in self.word2idx.items()}
    def add_token(self, t):
        if t not in self.word2idx:
            i = len(self.word2idx); self.word2idx[t]=i; self.idx2word[i]=t
    def build(self, sentences):
        for s in sentences:
            for t in s: self.add_token(t)
    def encode(self, tokens):
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]
    def decode(self, ids):
        return [self.idx2word.get(i, UNK_TOKEN) for i in ids]
    def __len__(self): return len(self.word2idx)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL DEFINITIONS
#     v1 (hw5): vanilla unidirectional LSTM, no beam
#     v2 (hw6): vanilla LSTM + beam search
#     v3 (hw7): bidirectional encoder + Bahdanau attention + beam search
# ─────────────────────────────────────────────────────────────────────────────

# ── v1 / v2  (identical architecture) ────────────────────────────────────────
class EncoderV1(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class DecoderV1(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, token, hidden, cell):
        token    = token.unsqueeze(0)
        embedded = self.dropout(self.embedding(token))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        logits = self.fc_out(output.squeeze(0))
        return logits, hidden, cell

class Seq2SeqV1(nn.Module):
    """hw5: greedy only."""
    def __init__(self, encoder, decoder, device):
        super().__init__(); self.encoder=encoder; self.decoder=decoder; self.device=device
    @torch.no_grad()
    def translate(self, src_tensor, max_len=MAX_GEN):
        self.eval()
        src = src_tensor.unsqueeze(1).to(self.device)
        hidden, cell = self.encoder(src)
        dec_input = torch.tensor([SOS_IDX], device=self.device)
        generated = []
        for _ in range(max_len):
            logits, hidden, cell = self.decoder(dec_input, hidden, cell)
            pred = logits.argmax(dim=1)
            if pred.item() == EOS_IDX: break
            generated.append(pred.item()); dec_input = pred
        return generated

class Seq2SeqV2(nn.Module):
    """hw6: greedy + beam search."""
    def __init__(self, encoder, decoder, device):
        super().__init__(); self.encoder=encoder; self.decoder=decoder; self.device=device
    @torch.no_grad()
    def translate(self, src_tensor, max_len=MAX_GEN):
        self.eval()
        src = src_tensor.unsqueeze(1).to(self.device)
        hidden, cell = self.encoder(src)
        dec_input = torch.tensor([SOS_IDX], device=self.device)
        generated = []
        for _ in range(max_len):
            logits, hidden, cell = self.decoder(dec_input, hidden, cell)
            pred = logits.argmax(dim=1)
            if pred.item() == EOS_IDX: break
            generated.append(pred.item()); dec_input = pred
        return generated
    @torch.no_grad()
    def beam_search(self, src_tensor, beam_width=5, max_len=MAX_GEN):
        self.eval()
        src = src_tensor.unsqueeze(1).to(self.device)
        hidden, cell = self.encoder(src)
        beams = [(0.0, [], hidden, cell)]; completed = []
        for _ in range(max_len):
            if not beams: break
            all_cands = []
            for lp, seq, h, c in beams:
                tok = torch.tensor([seq[-1] if seq else SOS_IDX], device=self.device)
                logits, h2, c2 = self.decoder(tok, h, c)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                for tlp, tid in zip(*log_probs.topk(beam_width)):
                    nlp = lp + tlp.item(); nseq = seq + [tid.item()]
                    if tid.item() == EOS_IDX:
                        completed.append((nlp/len(nseq), seq))
                    else:
                        all_cands.append((nlp, nseq, h2, c2))
            all_cands.sort(key=lambda x: x[0]/max(len(x[1]),1), reverse=True)
            beams = all_cands[:beam_width]
        if completed:
            return max(completed, key=lambda x: x[0])[1]
        if beams:
            return max(beams, key=lambda x: x[0]/max(len(x[1]),1))[1]
        return []

# ── v3 (hw7): bidirectional encoder + Bahdanau attention ─────────────────────
class EncoderV3(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim=hidden_dim; self.n_layers=n_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_cell   = nn.Linear(hidden_dim*2, hidden_dim)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        enc_out, (hidden, cell) = self.lstm(embedded)
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        cell   = cell.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden = torch.tanh(self.fc_hidden(torch.cat([hidden[:,0], hidden[:,1]], dim=-1)))
        cell   = torch.tanh(self.fc_cell(  torch.cat([cell[:,0],   cell[:,1]],   dim=-1)))
        return enc_out, hidden, cell

class AttentionV3(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(hidden_dim*2, attn_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim,   attn_dim, bias=False)
        self.v     = nn.Linear(attn_dim, 1,            bias=False)
    def forward(self, dec_hidden, enc_out):
        src_len = enc_out.size(0)
        dec_exp = dec_hidden.unsqueeze(1).expand(-1, src_len, -1)
        enc_t   = enc_out.permute(1, 0, 2)
        energy  = torch.tanh(self.W_enc(enc_t) + self.W_dec(dec_exp))
        alpha   = F.softmax(self.v(energy).squeeze(-1), dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), enc_t).squeeze(1)
        return context, alpha

class DecoderV3(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, attn_dim, n_layers, dropout):
        super().__init__()
        self.attention = AttentionV3(hidden_dim, attn_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm   = nn.LSTM(embed_dim + hidden_dim*2, hidden_dim, n_layers,
                              dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim*2, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, token, hidden, cell, enc_out):
        embedded        = self.dropout(self.embedding(token.unsqueeze(0)))
        context, alpha  = self.attention(hidden[-1], enc_out)
        rnn_input       = torch.cat([embedded, context.unsqueeze(0)], dim=-1)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        logits = self.fc_out(torch.cat([output.squeeze(0), context], dim=-1))
        return logits, hidden, cell, alpha

class Seq2SeqV3(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__(); self.encoder=encoder; self.decoder=decoder; self.device=device
    @torch.no_grad()
    def translate(self, src_tensor, max_len=MAX_GEN):
        self.eval()
        src = src_tensor.unsqueeze(1).to(self.device)
        enc_out, hidden, cell = self.encoder(src)
        dec_input = torch.tensor([SOS_IDX], device=self.device)
        generated = []
        for _ in range(max_len):
            logits, hidden, cell, _ = self.decoder(dec_input, hidden, cell, enc_out)
            pred = logits.argmax(dim=1)
            if pred.item() == EOS_IDX: break
            generated.append(pred.item()); dec_input = pred
        return generated
    @torch.no_grad()
    def beam_search(self, src_tensor, beam_width=5, max_len=MAX_GEN):
        self.eval()
        src = src_tensor.unsqueeze(1).to(self.device)
        enc_out, hidden, cell = self.encoder(src)
        beams = [(0.0, [], hidden, cell)]; completed = []
        for _ in range(max_len):
            if not beams: break
            all_cands = []
            for lp, seq, h, c in beams:
                tok = torch.tensor([seq[-1] if seq else SOS_IDX], device=self.device)
                logits, h2, c2, _ = self.decoder(tok, h, c, enc_out)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                for tlp, tid in zip(*log_probs.topk(beam_width)):
                    nlp = lp + tlp.item(); nseq = seq + [tid.item()]
                    if tid.item() == EOS_IDX:
                        completed.append((nlp/len(nseq), seq))
                    else:
                        all_cands.append((nlp, nseq, h2, c2))
            all_cands.sort(key=lambda x: x[0]/max(len(x[1]),1), reverse=True)
            beams = all_cands[:beam_width]
        if completed:
            return max(completed, key=lambda x: x[0])[1]
        if beams:
            return max(beams, key=lambda x: x[0]/max(len(x[1]),1))[1]
        return []

# ─────────────────────────────────────────────────────────────────────────────
# 6.  CHECKPOINT LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_v1(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a    = ckpt.get("args", {})
    sv, tv = ckpt["src_vocab"], ckpt["tgt_vocab"]
    enc = EncoderV1(len(sv), a.get("embed",256), a.get("hidden",1024),
                    a.get("layers",2), a.get("dropout",0.3))
    dec = DecoderV1(len(tv), a.get("embed",256), a.get("hidden",1024),
                    a.get("layers",2), a.get("dropout",0.3))
    model = Seq2SeqV1(enc, dec, device).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, sv, tv

def load_v2(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a    = ckpt.get("args", {})
    sv, tv = ckpt["src_vocab"], ckpt["tgt_vocab"]
    enc = EncoderV1(len(sv), a.get("embed",256), a.get("hidden",1024),
                    a.get("layers",2), a.get("dropout",0.3))
    dec = DecoderV1(len(tv), a.get("embed",256), a.get("hidden",1024),
                    a.get("layers",2), a.get("dropout",0.3))
    model = Seq2SeqV2(enc, dec, device).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, sv, tv

def load_v3(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    a    = ckpt.get("args", {})
    sv, tv = ckpt["src_vocab"], ckpt["tgt_vocab"]
    enc = EncoderV3(len(sv), a.get("embed",256), a.get("hidden",512),
                    a.get("layers",2), a.get("dropout",0.3))
    dec = DecoderV3(len(tv), a.get("embed",256), a.get("hidden",512),
                    a.get("attn_dim",256), a.get("layers",2), a.get("dropout",0.3))
    model = Seq2SeqV3(enc, dec, device).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, sv, tv

# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def tokens_from_sentence(sentence, src_vocab):
    toks = normalize_en(sentence).split()
    if not toks: return None
    return torch.tensor(src_vocab.encode(toks), dtype=torch.long)

def ids_to_str(ids, tgt_vocab):
    return " ".join(tgt_vocab.decode(ids))

def evaluate_model(model, val_pairs, src_vocab, tgt_vocab, device,
                   n, has_beam, beam_width=5):
    """Returns dict with bleu1/2/4 for greedy (and beam if available)."""
    subset = val_pairs[:n]
    refs   = [p[1] for p in subset]

    # greedy
    greedy_hyps = []
    t0 = time.time()
    for src_toks, _ in subset:
        src = tokens_from_sentence(" ".join(src_toks), src_vocab)
        if src is None: greedy_hyps.append([]); continue
        greedy_hyps.append(ids_to_str(model.translate(src), tgt_vocab).split())
    greedy_ms = (time.time()-t0)/n*1000

    result = {
        "greedy": {
            "bleu1": corpus_bleu(greedy_hyps, refs, 1),
            "bleu2": corpus_bleu(greedy_hyps, refs, 2),
            "bleu4": corpus_bleu(greedy_hyps, refs, 4),
            "ms":    greedy_ms,
        }
    }

    if has_beam:
        beam_hyps = []
        t0 = time.time()
        for src_toks, _ in subset:
            src = tokens_from_sentence(" ".join(src_toks), src_vocab)
            if src is None: beam_hyps.append([]); continue
            beam_hyps.append(ids_to_str(
                model.beam_search(src, beam_width=beam_width), tgt_vocab).split())
        beam_ms = (time.time()-t0)/n*1000
        result["beam5"] = {
            "bleu1": corpus_bleu(beam_hyps, refs, 1),
            "bleu2": corpus_bleu(beam_hyps, refs, 2),
            "bleu4": corpus_bleu(beam_hyps, refs, 4),
            "ms":    beam_ms,
        }
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare HW5/HW6/HW7 BLEU")
    parser.add_argument("--data",   required=True, help="Path to rus.txt")
    parser.add_argument("--ckpt5",  default=None,  help="HW5 checkpoint (.pt)")
    parser.add_argument("--ckpt6",  default=None,  help="HW6 checkpoint (.pt)")
    parser.add_argument("--ckpt7",  default=None,  help="HW7 checkpoint (.pt)")
    parser.add_argument("--n_eval", type=int, default=500,
                        help="Number of val sentences to evaluate on")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--beam_width", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── load data (need val split to evaluate on same sentences) ─────────────
    print(f"Loading data from '{args.data}' …")
    pairs = load_pairs(args.data)
    random.shuffle(pairs)
    split = int(0.9 * len(pairs))
    val_pairs = pairs[split:]
    n = min(args.n_eval, len(val_pairs))
    print(f"Evaluating on {n} validation sentences.\n")

    # ── load each checkpoint ─────────────────────────────────────────────────
    models = {}  # name → (model, sv, tv, has_beam, label)
    loaders = [
        ("HW-5  (vanilla LSTM, greedy)",       args.ckpt5, load_v1, False),
        ("HW-6  (vanilla LSTM + beam)",         args.ckpt6, load_v2, True),
        ("HW-7  (bidir + attention + beam)",    args.ckpt7, load_v3, True),
    ]
    for label, path, loader_fn, has_beam in loaders:
        if path is None:
            print(f"  [{label}]  skipped (no checkpoint provided)")
            continue
        try:
            model, sv, tv = loader_fn(path, device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            models[label] = (model, sv, tv, has_beam)
            print(f"  [{label}]  loaded  ({n_params:,} params)")
        except Exception as e:
            print(f"  [{label}]  FAILED to load: {e}")

    if not models:
        print("\nNo models loaded. Provide at least one --ckptN argument.")
        return

    # ── evaluate ──────────────────────────────────────────────────────────────
    print(f"\nRunning evaluation (beam_width={args.beam_width}) …\n")
    all_results = {}
    for label, (model, sv, tv, has_beam) in models.items():
        print(f"  Evaluating {label} …")
        res = evaluate_model(model, val_pairs, sv, tv, device,
                             n, has_beam, args.beam_width)
        all_results[label] = (res, has_beam)

    # ── print table ───────────────────────────────────────────────────────────
    SEP = "─" * 90
    print(f"\n{'':=<90}")
    print("  BLEU COMPARISON TABLE")
    print(f"{'':=<90}")
    hdr = f"{'Model':<42} {'Decode':<8} {'BLEU-1':>8} {'BLEU-2':>8} {'BLEU-4':>8} {'ms/sent':>9}"
    print(hdr); print(SEP)

    for label, (res, has_beam) in all_results.items():
        for decode_name, r in res.items():
            label_col = label if decode_name == "greedy" else ""
            print(f"{label_col:<42} {decode_name:<8} "
                  f"{r['bleu1']:>8.4f} {r['bleu2']:>8.4f} {r['bleu4']:>8.4f} "
                  f"{r['ms']:>9.1f}")
        print(SEP)

    # ── per-model best BLEU-4 summary ────────────────────────────────────────
    print("\n  BEST BLEU-4 PER MODEL")
    print(f"  {'Model':<42} {'Decode':<8} {'BLEU-4':>8}")
    print(f"  {'─'*60}")
    for label, (res, _) in all_results.items():
        best_decode = max(res, key=lambda k: res[k]["bleu4"])
        best_b4     = res[best_decode]["bleu4"]
        print(f"  {label:<42} {best_decode:<8} {best_b4:>8.4f}")

    # ── beam vs greedy delta for models that have both ────────────────────────
    has_both = [(l, r) for l, (r, hb) in all_results.items() if hb]
    if has_both:
        print(f"\n  BEAM-5 vs GREEDY DELTA (BLEU-4)")
        print(f"  {'Model':<42} {'Δ BLEU-4':>10} {'Δ ms/sent':>12}")
        print(f"  {'─'*66}")
        for label, res in has_both:
            delta_b4 = res["beam5"]["bleu4"] - res["greedy"]["bleu4"]
            delta_ms = res["beam5"]["ms"]    - res["greedy"]["ms"]
            sign = "+" if delta_b4 >= 0 else ""
            print(f"  {label:<42} {sign}{delta_b4:>9.4f} {delta_ms:>+12.1f}")

    print(f"\n{'':=<90}\n")

    # ── qualitative examples (first 5 val sentences) ─────────────────────────
    print("QUALITATIVE EXAMPLES (first 5 validation sentences)\n")
    for i, (src_toks, ref_toks) in enumerate(val_pairs[:5], 1):
        src_str = " ".join(src_toks); ref_str = " ".join(ref_toks)
        print(f"  [{i}] EN  : {src_str}")
        print(f"       REF : {ref_str}")
        for label, (model, sv, tv, has_beam) in models.items():
            src = tokens_from_sentence(src_str, sv)
            if src is None: continue
            g = ids_to_str(model.translate(src), tv)
            g_b = bleu_score(g.split(), ref_toks)
            short = label.split("(")[0].strip()
            print(f"       {short:<10} greedy: {g}  (BLEU={g_b:.3f})")
            if has_beam:
                b = ids_to_str(model.beam_search(src, beam_width=args.beam_width), tv)
                b_b = bleu_score(b.split(), ref_toks)
                print(f"       {short:<10} beam-5: {b}  (BLEU={b_b:.3f})")
        print()


if __name__ == "__main__":
    main()