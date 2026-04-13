"""
Seq2Seq English → Russian Translation with LSTM Encoder-Decoder
Usage:
    python homework-5.py --data rus.txt --epochs 20
    python homework-5.py --data rus.txt --translate "Hello, how are you?"
"""

import argparse
import random
import re
import time
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
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
HIDDEN_DIM  = 1024
N_LAYERS    = 2
DROPOUT     = 0.3
BATCH_SIZE  = 128
LR          = 0.001
CLIP        = 1.0
TF_RATIO    = 0.5      # teacher-forcing ratio
MAX_LEN     = 50       # max tokens per sentence (for filtering)
MAX_GEN     = 60       # max tokens to generate at inference


# ─────────────────────────────────────────────
# 2.  TEXT PREPROCESSING
# ─────────────────────────────────────────────
def unicode_to_ascii(s: str) -> str:
    """Normalise unicode characters to ASCII where possible."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize_en(s: str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_ru(s: str) -> str:
    """Keep Cyrillic letters and spaces only."""
    s = s.lower().strip()
    s = re.sub(r"[^а-яёА-ЯЁ\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_pairs(path: str, max_len: int = MAX_LEN):
    """
    Load tab-separated EN\tRU pairs from *path*.
    Lines have the form:
        English sentence<TAB>Russian sentence<TAB>attribution...
    We only keep the first two columns.
    """
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
            en_tok = en.split()
            ru_tok = ru.split()
            if len(en_tok) < 1 or len(ru_tok) < 1:
                continue
            if len(en_tok) > max_len or len(ru_tok) > max_len:
                continue
            pairs.append((en_tok, ru_tok))
    return pairs


# ─────────────────────────────────────────────
# 3.  VOCABULARY
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
# 4.  DATASET & DATALOADER
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
        # decoder input: <sos> ... <eos>
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
# 5.  MODEL
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0,
                            bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (src_len, B)
        embedded = self.dropout(self.embedding(src))          # (src_len, B, E)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell   # (n_layers, B, H)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden, cell):
        # token: (B,)  →  unsqueeze to (1, B)
        token = token.unsqueeze(0)
        embedded = self.dropout(self.embedding(token))        # (1, B, E)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        logits = self.fc_out(output.squeeze(0))               # (B, vocab_size)
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=TF_RATIO):
        """
        src : (src_len, B)
        tgt : (tgt_len, B)  — includes <sos> at position 0
        """
        tgt_len, batch_size = tgt.shape
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        # First decoder input is <sos>
        dec_input = tgt[0]   # (B,)

        for t in range(1, tgt_len):
            logits, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[t] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(dim=1)
            dec_input = tgt[t] if teacher_force else top1

        return outputs  # (tgt_len, B, vocab_size)

    @torch.no_grad()
    def translate(self, src_tensor, max_len=MAX_GEN):
        """
        src_tensor : (src_len,) – single sentence, no batch dim.
        Returns list of token indices (excluding <sos>/<eos>).
        """
        self.eval()
        src = src_tensor.unsqueeze(1).to(self.device)   # (src_len, 1)
        hidden, cell = self.encoder(src)

        dec_input = torch.tensor([SOS_IDX], device=self.device)
        generated = []

        for _ in range(max_len):
            logits, hidden, cell = self.decoder(dec_input, hidden, cell)
            pred = logits.argmax(dim=1)
            if pred.item() == EOS_IDX:
                break
            generated.append(pred.item())
            dec_input = pred

        return generated


# ─────────────────────────────────────────────
# 6.  TRAINING HELPERS
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)          # (tgt_len, B, V)

        # Flatten: skip position 0 (which is <sos>)
        output_flat = output[1:].reshape(-1, output.shape[-1])
        tgt_flat    = tgt[1:].reshape(-1)

        loss = criterion(output_flat, tgt_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt, teacher_forcing_ratio=0.0)
        output_flat = output[1:].reshape(-1, output.shape[-1])
        tgt_flat    = tgt[1:].reshape(-1)
        loss = criterion(output_flat, tgt_flat)
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def epoch_time(start, end):
    elapsed = end - start
    return int(elapsed // 60), int(elapsed % 60)


# ─────────────────────────────────────────────
# 7.  TRANSLATE HELPER (str → str)
# ─────────────────────────────────────────────
def translate_sentence(sentence: str, model: Seq2Seq,
                        src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                        device) -> str:
    tokens = normalize_en(sentence).split()
    if not tokens:
        return ""
    src_ids = torch.tensor(src_vocab.encode(tokens), dtype=torch.long)
    pred_ids = model.translate(src_ids)
    return " ".join(tgt_vocab.decode(pred_ids))


# ─────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Seq2Seq EN→RU translator")
    parser.add_argument("--data",      default="rus.txt",
                        help="Path to tab-separated EN-RU file")
    parser.add_argument("--epochs",    type=int, default=15)
    parser.add_argument("--batch",     type=int, default=BATCH_SIZE)
    parser.add_argument("--embed",     type=int, default=EMBED_DIM)
    parser.add_argument("--hidden",    type=int, default=HIDDEN_DIM)
    parser.add_argument("--layers",    type=int, default=N_LAYERS)
    parser.add_argument("--dropout",   type=float, default=DROPOUT)
    parser.add_argument("--lr",        type=float, default=LR)
    parser.add_argument("--clip",      type=float, default=CLIP)
    parser.add_argument("--tf",        type=float, default=TF_RATIO,
                        help="Teacher-forcing ratio (0–1)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--save",      default="best_model.pt")
    parser.add_argument("--translate", default=None,
                        help="Translate a single sentence and exit")
    args = parser.parse_args()

    # ── reproducibility ──
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ──
    print(f"Loading data from '{args.data}' …")
    pairs = load_pairs(args.data)
    print(f"  {len(pairs):,} sentence pairs after filtering (max_len={MAX_LEN})")

    random.shuffle(pairs)
    split = int(0.9 * len(pairs))
    train_pairs, val_pairs = pairs[:split], pairs[split:]
    print(f"  Train: {len(train_pairs):,}  |  Val: {len(val_pairs):,}")

    # ── vocabularies ──
    src_vocab = Vocabulary("en")
    tgt_vocab = Vocabulary("ru")
    src_vocab.build(p[0] for p in train_pairs)
    tgt_vocab.build(p[1] for p in train_pairs)
    print(f"  EN vocab: {len(src_vocab):,}  |  RU vocab: {len(tgt_vocab):,}")

    # ── datasets & loaders ──
    train_ds = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds   = TranslationDataset(val_pairs,   src_vocab, tgt_vocab)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          collate_fn=collate_fn)

    # ── model ──
    encoder = Encoder(len(src_vocab), args.embed, args.hidden,
                      args.layers, args.dropout)
    decoder = Decoder(len(tgt_vocab), args.embed, args.hidden,
                      args.layers, args.dropout)
    model   = Seq2Seq(encoder, decoder, device).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # ── loss & optimiser ──
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── inference-only mode ──
    if args.translate is not None:
        ckpt_path = Path(args.save)
        if not ckpt_path.exists():
            print(f"No checkpoint found at '{args.save}'. Train the model first.")
            return
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        src_vocab = checkpoint["src_vocab"]
        tgt_vocab = checkpoint["tgt_vocab"]
        result = translate_sentence(args.translate, model, src_vocab, tgt_vocab, device)
        print(f"\nEN: {args.translate}")
        print(f"RU: {result}")
        return

    # ── training loop ──
    best_val_loss = float("inf")

    # Demo sentences to watch progress
    demo = ["go", "help me", "run", "i know", "who won"]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_dl, optimizer, criterion,
                                 args.clip, device)
        val_loss   = evaluate(model, val_dl, criterion, device)
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
              f"Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        # Quick qualitative check every 5 epochs
        if epoch % 5 == 0:
            print("  Quick translations:")
            for s in demo:
                tr = translate_sentence(s, model, src_vocab, tgt_vocab, device)
                print(f"    '{s}'  →  '{tr}'")
            model.train()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to '{args.save}'")

    # ── final demo ──
    checkpoint = torch.load(args.save, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print("\nFinal translations (best checkpoint):")
    test_sents = [
        "go",
        "help me",
        "run",
        "stop it",
        "i know",
        "who won",
        "wake up",
        "be brave",
        "i am old",
        "i will try",
    ]
    for s in test_sents:
        tr = translate_sentence(s, model, src_vocab, tgt_vocab, device)
        print(f"  {s:<25} →  {tr}")


if __name__ == "__main__":
    main()