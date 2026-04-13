# ============================================================
#  Seq2Seq Machine Translation — English → Russian
#  LSTM-based encoder-decoder with teacher forcing, trained
#  on the Tatoeba English-Russian sentence pairs (rus.txt).
# ============================================================

import random
import re
import unicodedata

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# ── 1. Hyperparameters & Constants ───────────────────────────

DATA_FILE            = "rus.txt"
CHECKPOINT_PATH      = "best_model.pt"

MAX_LEN              = 15       # max tokens per sentence (both sides)
NUM_SAMPLES          = 30_000   # maximum sentence pairs to load
TRAIN_RATIO          = 0.9      # fraction of pairs used for training

EMBED_DIM            = 256
HIDDEN_DIM           = 1024
DROPOUT              = 0.3

NUM_EPOCHS           = 20
BATCH_SIZE           = 64
LEARNING_RATE        = 0.001
GRAD_CLIP            = 1.0
TEACHER_FORCING_RATIO = 0.5     # during training


# ── 2. Data Loading & Preprocessing ──────────────────────────

def unicode_to_ascii(s):
    """Strip combining diacritics (accents) from a Unicode string."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def clean_en(text):
    """Lowercase, strip accents, and keep only ASCII letters and .!?"""
    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r"[^a-zA-Z.!?]+", " ", text)
    return text.strip()


def clean_ru(text):
    """Lowercase and keep only Cyrillic letters and .!?"""
    text = text.lower().strip()
    text = re.sub(r"[^а-яёА-ЯЁ.!?]+", " ", text)
    return text.strip()


def load_data(filepath, max_len=MAX_LEN, num_samples=NUM_SAMPLES):
    """
    Load tab-separated EN/RU pairs from *filepath*.

    Filters pairs where either side exceeds *max_len* tokens and
    returns up to *num_samples* pairs in random order.
    """
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            en = clean_en(parts[0])
            ru = clean_ru(parts[1])
            if len(en.split()) <= max_len and len(ru.split()) <= max_len:
                pairs.append((en, ru))
            if len(pairs) >= num_samples:
                break

    random.shuffle(pairs)
    return pairs


pairs = load_data(DATA_FILE)
print(f"Total pairs: {len(pairs)}")
print("Examples:")
for en, ru in pairs[:5]:
    print(f"  EN: {en}")
    print(f"  RU: {ru}")
    print()


# ── 3. Vocabulary ─────────────────────────────────────────────

class Vocabulary:
    """
    Simple word-level vocabulary with reserved special tokens:
      0 → <pad>
      1 → <sos>  (start of sequence)
      2 → <eos>  (end of sequence)
      3 → <unk>  (unknown word)
    """

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, name):
        self.name     = name
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.n_words  = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1


# Build vocabularies on training split only (avoids leaking val data)
split_idx   = int(len(pairs) * TRAIN_RATIO)
train_pairs = pairs[:split_idx]
val_pairs   = pairs[split_idx:]

en_vocab = Vocabulary("english")
ru_vocab = Vocabulary("russian")

for en, ru in train_pairs:
    en_vocab.add_sentence(en)
    ru_vocab.add_sentence(ru)

print(f"English vocab size: {en_vocab.n_words}")
print(f"Russian vocab size: {ru_vocab.n_words}")
print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")


# ── 4. Dataset & DataLoader ───────────────────────────────────

class TranslationDataset(Dataset):
    """
    Wraps a list of (en, ru) string pairs.

    Each item returns:
      en_ids : LongTensor of source token indices
      ru_ids : LongTensor of <sos> + target token indices + <eos>
    Unknown words are mapped to UNK_IDX (3).
    """

    def __init__(self, pairs, en_vocab, ru_vocab):
        self.pairs    = pairs
        self.en_vocab = en_vocab
        self.ru_vocab = ru_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, ru = self.pairs[idx]
        en_ids = [self.en_vocab.word2idx.get(w, Vocabulary.UNK_IDX) for w in en.split()]
        ru_ids = (
            [Vocabulary.SOS_IDX]
            + [self.ru_vocab.word2idx.get(w, Vocabulary.UNK_IDX) for w in ru.split()]
            + [Vocabulary.EOS_IDX]
        )
        return (
            torch.tensor(en_ids, dtype=torch.long),
            torch.tensor(ru_ids, dtype=torch.long),
        )


def collate_fn(batch):
    """Pad variable-length sequences to the longest in the batch."""
    en_batch, ru_batch = zip(*batch)
    en_padded = nn.utils.rnn.pad_sequence(
        en_batch, batch_first=False, padding_value=Vocabulary.PAD_IDX
    )
    ru_padded = nn.utils.rnn.pad_sequence(
        ru_batch, batch_first=False, padding_value=Vocabulary.PAD_IDX
    )
    return en_padded, ru_padded


train_dataset = TranslationDataset(train_pairs, en_vocab, ru_vocab)
val_dataset   = TranslationDataset(val_pairs,   en_vocab, ru_vocab)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

en_batch, ru_batch = next(iter(train_loader))
print(f"\nEN batch shape: {en_batch.shape}")
print(f"RU batch shape: {ru_batch.shape}")


# ── 5. Seq2Seq Model (Encoder / Decoder) ─────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")


class Encoder(nn.Module):
    """
    Single-layer LSTM encoder.

    Reads the source sequence and produces a context (hidden, cell)
    that is passed to the decoder.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=Vocabulary.PAD_IDX)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    """
    Single-layer LSTM decoder with a linear projection to vocabulary logits.

    At each step it receives one target token, the previous hidden state,
    and the previous cell state.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=Vocabulary.PAD_IDX)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=False)
        self.fc        = nn.Linear(hidden_dim, vocab_size)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        x        = x.unsqueeze(0)                          # (1, batch)
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))            # (batch, vocab)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    Encoder-decoder wrapper with optional teacher forcing.

    During training (teacher_forcing_ratio > 0) the ground-truth
    target token is fed to the decoder at each step with the given
    probability; otherwise the decoder's own prediction is used.
    """

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    def forward(self, src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        batch_size = src.shape[1]
        trg_len    = trg.shape[0]
        trg_vocab  = self.decoder.fc.out_features

        outputs     = torch.zeros(trg_len, batch_size, trg_vocab).to(self.device)
        hidden, cell = self.encoder(src)
        input_token  = trg[0]          # first token is always <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input_token   = trg[t] if teacher_force else output.argmax(1)

        return outputs


encoder = Encoder(en_vocab.n_words, EMBED_DIM, HIDDEN_DIM, DROPOUT).to(device)
decoder = Decoder(ru_vocab.n_words, EMBED_DIM, HIDDEN_DIM, DROPOUT).to(device)
model   = Seq2Seq(encoder, decoder, device).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(model)


# ── 6. Training & Evaluation Functions ───────────────────────

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD_IDX)


def train_epoch(model, loader, optimizer, criterion, clip=GRAD_CLIP):
    """Run one training epoch and return the average cross-entropy loss."""
    model.train()
    epoch_loss = 0

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

        # Flatten for cross-entropy: skip the first <sos> step
        output = output[1:].reshape(-1, output.shape[-1])
        trg    = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(model, loader, criterion):
    """Evaluate the model on *loader* (no teacher forcing) and return average loss."""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            output   = model(src, trg, teacher_forcing_ratio=0.0)

            output = output[1:].reshape(-1, output.shape[-1])
            trg    = trg[1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


# ── 7. Training Loop ──────────────────────────────────────────

best_val_loss = float("inf")
train_losses  = []
val_losses    = []

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss   = evaluate(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save checkpoint whenever validation improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(
        f"Epoch {epoch + 1:2d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train PPL: {np.exp(train_loss):7.2f} | "
        f"Val PPL: {np.exp(val_loss):7.2f}"
    )

print(f"\nBest Val Loss: {best_val_loss:.4f}")

# ── Loss curve ──
plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses,
         marker="o", label="Train Loss", color="steelblue")
plt.plot(range(1, NUM_EPOCHS + 1), val_losses,
         marker="s", label="Val Loss",   color="coral")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_val_loss.png", dpi=150)
plt.show()
print("Loss curve saved to training_val_loss.png")


# ── 8. Translation ────────────────────────────────────────────

# Restore the best checkpoint before inference
model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
model.eval()


def translate(sentence, max_len=50):
    """
    Translate an English sentence (str) to Russian using greedy decoding.

    Parameters
    ----------
    sentence : str | Tensor
        A raw English string or a pre-encoded source tensor
        of shape (seq_len, 1).
    max_len : int
        Maximum number of tokens to generate.

    Returns
    -------
    str
        The translated Russian sentence (space-separated tokens).
    """
    if isinstance(sentence, str):
        sentence = clean_en(sentence)
        tokens   = [en_vocab.word2idx.get(w, Vocabulary.UNK_IDX) for w in sentence.split()]
        src      = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)
    else:
        src = sentence

    with torch.no_grad():
        hidden, cell = model.encoder(src)

    input_token = torch.tensor(
        [ru_vocab.word2idx["<sos>"]], dtype=torch.long
    ).to(device)
    translated = []

    with torch.no_grad():
        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            top_token = output.argmax(1)
            word      = ru_vocab.idx2word[top_token.item()]
            if word == "<eos>":
                break
            translated.append(word)
            input_token = top_token

    return " ".join(translated)


# ── Fixed test sentences ──
print("=" * 55)
print("TRANSLATION EXAMPLES")
print("=" * 55)

test_sentences = [
    "i love you",
    "he is very old",
    "she is beautiful",
    "we are going home",
    "i don t know",
    "let s go",
    "how are you",
    "i am happy",
    "the cat is here",
    "good morning",
]

for en in test_sentences:
    ru = translate(en)
    print(f"  EN: {en}")
    print(f"  RU: {ru}")
    print()

# ── Validation set: ground truth vs. prediction ──
print("=" * 55)
print("VALIDATION SET EXAMPLES (Ground Truth vs Predicted)")
print("=" * 55)

for en, ru_true in val_pairs[:8]:
    ru_pred = translate(en)
    print(f"  EN:   {en}")
    print(f"  TRUE: {ru_true}")
    print(f"  PRED: {ru_pred}")
    print()
