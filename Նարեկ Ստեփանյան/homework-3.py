import numpy as np
import json
import time

np.random.seed(42)

FRANKENSTEIN_TEXT = """
You will rejoice to hear that no disaster has accompanied the commencement of
an enterprise which you have regarded with such evil forebodings. I arrived
here yesterday, and my first task is to assure my dear sister of my welfare and
increasing confidence in the success of my undertaking.
I am already far north of London, and as I walk in the streets of Petersburgh,
I feel a cold northern breeze play upon my cheeks, which braces my nerves and
fills me with delight. Do you understand this feeling? This breeze, which has
travelled from the regions towards which I am advancing, gives me a foretaste
of those icy climes. Inspirited by this wind of promise, my daydreams become
more fervent and vivid. I try in vain to be persuaded that the pole is the seat
of frost and desolation; it ever presents itself to my imagination as the
region of beauty and delight. There, Margaret, the sun is for ever visible, its
broad disk just skirting the horizon and diffusing a perpetual splendour.
I have no friend, Margaret: when I am glowing with the enthusiasm of success,
there will be none to participate my joy; if I am assailed by disappointment,
no one will endeavour to sustain me in dejection. I shall commit my thoughts to
paper, it is true; but that is a poor medium for the communication of feeling.
I desire the company of a man who could sympathise with me, whose eyes would
reply to mine. You may deem me romantic, my dear sister, but I bitterly feel
the want of a friend. I have no one near me, gentle yet courageous, possessed
of a cultivated as well as of a capacious mind, whose tastes are like my own,
to approve or amend my plans.
I am by birth a Genevese, and my family is one of the most distinguished of
that republic. My ancestors had been for many years counsellors and syndics,
and my father had filled several public situations with honour and reputation.
He was respected by all who knew him for his integrity and indefatigable
attention to public business.
Natural philosophy is the genius that has regulated my fate; I desire,
therefore, in this narration, to state those facts which led to my predilection
for that science. When I was thirteen years of age we all went on a party of
pleasure to the baths near Thonon; the inclemency of the weather obliged us to
remain a day confined to the inn. In this house I chanced to find a volume of
the works of Cornelius Agrippa. I opened it with apathy; the theory which he
attempts to demonstrate and the wonderful facts which he relates soon changed
this feeling into enthusiasm. A new light seemed to dawn upon my mind.
We were brought up together; there was not quite a year difference in our ages.
I need not say that we were strangers to any species of disunion or dispute.
Harmony was the soul of our companionship, and the diversity and contrast that
subsisted in our characters drew us nearer together. Elizabeth was of a calmer
and more concentrated disposition; but, with all my ardour, I was capable of a
more intense application and was more deeply smitten with the thirst for
knowledge. She busied herself with following the aerial creations of the poets.
Unhappy man! Do you share my madness? Have you drunk also of the intoxicating
draught? Hear me; let me reveal my tale, and you will dash the cup from your lips!
You seek for knowledge and wisdom, as I once did; and I ardently hope that the
gratification of your wishes may not be a serpent to sting you, as mine has been.
I do not know that the relation of my disasters will be useful to you; yet, when
I reflect that you are pursuing the same course, exposing yourself to the same
dangers which have rendered me what I am, I imagine that you may deduce an apt
moral from my tale, one that may direct you if you succeed in your undertaking
and console you in case of failure. Prepare to hear of occurrences which are
usually deemed marvellous.
Strange and harrowing must be his story, frightful the storm which embraced
the gallant vessel on its course and wrecked it. His eyes have generally an
expression of wildness, and even madness, but there are moments when, if anyone
performs an act of kindness towards him or does him any the most trifling
service, his whole countenance is lighted up, as it were, with a beam of
benevolence and sweetness that I never saw equalled. But he is generally
melancholy and despairing, and sometimes he gnashes his teeth, as if impatient
of the weight of woes that oppresses him.
Nothing contributes so much to tranquillise the mind as a steady purpose, a
point on which the soul may fix its intellectual eye. This expedition has been
the favourite dream of my early years. I have read with ardour the accounts of
the various voyages which have been made in the prospect of arriving at the
North Pacific Ocean through the seas which surround the pole.
""".strip()


chars = sorted(set(FRANKENSTEIN_TEXT))
vocab_size = len(chars)
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

print(f"Text length   : {len(FRANKENSTEIN_TEXT):,} characters")
print(f"Vocabulary    : {vocab_size} unique characters")
print(f"Chars         : {''.join(chars[:30])}...")

def encode(text):
    return [char2idx[c] for c in text]

def one_hot(idx, size):
    v = np.zeros((size, 1))
    v[idx] = 1.0
    return v

# Build training sequences
SEQ_LEN = 25

def make_sequences(text, seq_len=SEQ_LEN):
    encoded = encode(text)
    X, y = [], []
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i:i+seq_len])
        y.append(encoded[i+seq_len])
    return X, y

X_all, y_all = make_sequences(FRANKENSTEIN_TEXT)
split = int(0.85 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

print(f"Train samples : {len(X_train):,}")
print(f"Test  samples : {len(X_test):,}\n")

class VanillaRNN:
    """Character-level Vanilla RNN.
       Input : one-hot encoded character (vocab_size,)
       Output: softmax distribution over next character (vocab_size,)
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.005):
        self.H = hidden_size
        self.lr = lr
        scale = 0.01
        self.Wx = np.random.randn(hidden_size, input_size)  * scale
        self.Wh = np.random.randn(hidden_size, hidden_size) * scale
        self.bh = np.zeros((hidden_size, 1))
        self.Wy = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(self, x_indices):
        """x_indices: list of int indices, returns probs (output_size,)"""
        h = np.zeros((self.H, 1))
        self.cache = []
        for idx in x_indices:
            x = one_hot(idx, self.Wx.shape[1])
            h_new = np.tanh(self.Wx @ x + self.Wh @ h + self.bh)
            self.cache.append((x, h, h_new))
            h = h_new
        self.h_last = h
        logits = self.Wy @ h + self.by
        probs = self.softmax(logits)
        self.probs = probs
        return probs

    def backward(self, y_true_idx):
        """Cross-entropy loss + BPTT"""
        probs = self.probs
        dlogits = probs.copy()
        dlogits[y_true_idx] -= 1.0          # d(cross-entropy)/d(logits)
        loss = -np.log(probs[y_true_idx, 0] + 1e-9)

        dWy = dlogits @ self.h_last.T
        dby = dlogits.copy()
        dh  = self.Wy.T @ dlogits

        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)

        for x, h_prev, h in reversed(self.cache):
            dtanh = (1 - h**2) * dh
            dWx  += dtanh @ x.T
            dWh  += dtanh @ h_prev.T
            dbh  += dtanh
            dh    = self.Wh.T @ dtanh

        # Gradient clipping
        for g in [dWx, dWh, dbh, dWy, dby]:
            np.clip(g, -5, 5, out=g)

        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh
        self.bh -= self.lr * dbh
        self.Wy -= self.lr * dWy
        self.by -= self.lr * dby

        return float(loss)

    def sample(self, seed_indices, n=100, temperature=0.8):
        """Generate text by sampling from the model."""
        indices = list(seed_indices)
        for _ in range(n):
            probs = self.forward(indices[-SEQ_LEN:])
            p = probs[:, 0]
            p = np.exp(np.log(p + 1e-9) / temperature)
            p /= p.sum()
            next_idx = np.random.choice(len(p), p=p)
            indices.append(next_idx)
        return ''.join(idx2char[i] for i in indices)

    @property
    def num_params(self):
        return sum(w.size for w in [self.Wx, self.Wh, self.bh, self.Wy, self.by])

# LSTM


class LSTM:
    """Character-level LSTM.
       3 gates: forget, input, output + cell state.
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.005):
        self.H = hidden_size
        self.lr = lr
        scale = 0.01
        I = input_size

        self.Wf = np.random.randn(hidden_size, hidden_size + I) * scale
        self.Wi = np.random.randn(hidden_size, hidden_size + I) * scale
        self.Wg = np.random.randn(hidden_size, hidden_size + I) * scale
        self.Wo = np.random.randn(hidden_size, hidden_size + I) * scale

        self.bf = np.ones ((hidden_size, 1))   # bias forget gate = 1 (good init)
        self.bi = np.zeros((hidden_size, 1))
        self.bg = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        self.Wy = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))

    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    def tanh(self, x):    return np.tanh(np.clip(x, -10, 10))
    def softmax(self, x):
        e = np.exp(x - np.max(x)); return e / e.sum()

    def forward(self, x_indices):
        h = np.zeros((self.H, 1))
        c = np.zeros((self.H, 1))
        self.cache = []

        for idx in x_indices:
            x   = one_hot(idx, self.Wf.shape[1] - self.H)
            xh  = np.vstack([h, x])
            f   = self.sigmoid(self.Wf @ xh + self.bf)
            i   = self.sigmoid(self.Wi @ xh + self.bi)
            g   = self.tanh   (self.Wg @ xh + self.bg)
            o   = self.sigmoid(self.Wo @ xh + self.bo)
            c_n = f * c + i * g
            h_n = o * self.tanh(c_n)
            self.cache.append((x, h, c, xh, f, i, g, o, c_n, h_n))
            h, c = h_n, c_n

        self.h_last = h
        logits = self.Wy @ h + self.by
        probs  = self.softmax(logits)
        self.probs = probs
        return probs

    def backward(self, y_true_idx):
        probs   = self.probs
        dlogits = probs.copy()
        dlogits[y_true_idx] -= 1.0
        loss = -np.log(probs[y_true_idx, 0] + 1e-9)

        dWy = dlogits @ self.h_last.T
        dby = dlogits.copy()
        dh  = self.Wy.T @ dlogits
        dc  = np.zeros_like(dh)

        dWf=np.zeros_like(self.Wf); dWi=np.zeros_like(self.Wi)
        dWg=np.zeros_like(self.Wg); dWo=np.zeros_like(self.Wo)
        dbf=np.zeros_like(self.bf); dbi=np.zeros_like(self.bi)
        dbg=np.zeros_like(self.bg); dbo=np.zeros_like(self.bo)

        for (x, h_prev, c_prev, xh, f, i, g, o, c_n, h) in reversed(self.cache):
            tanh_c = self.tanh(c_n)
            do = dh * tanh_c;            dc += dh * o * (1 - tanh_c**2)
            df = dc * c_prev;            di = dc * g
            dg = dc * i;                 dc  = dc * f

            dWo += (do * o*(1-o)) @ xh.T;  dbo += do * o*(1-o)
            dWf += (df * f*(1-f)) @ xh.T;  dbf += df * f*(1-f)
            dWi += (di * i*(1-i)) @ xh.T;  dbi += di * i*(1-i)
            dWg += (dg * (1-g**2)) @ xh.T; dbg += dg * (1-g**2)

            dxh = (self.Wo.T@(do*o*(1-o)) + self.Wf.T@(df*f*(1-f)) +
                   self.Wi.T@(di*i*(1-i)) + self.Wg.T@(dg*(1-g**2)))
            dh = dxh[:self.H]

        for g in [dWf,dWi,dWg,dWo,dbf,dbi,dbg,dbo,dWy,dby]:
            np.clip(g, -5, 5, out=g)

        self.Wf-=self.lr*dWf; self.Wi-=self.lr*dWi
        self.Wg-=self.lr*dWg; self.Wo-=self.lr*dWo
        self.bf-=self.lr*dbf; self.bi-=self.lr*dbi
        self.bg-=self.lr*dbg; self.bo-=self.lr*dbo
        self.Wy-=self.lr*dWy; self.by-=self.lr*dby

        return float(loss)

    def sample(self, seed_indices, n=100, temperature=0.8):
        indices = list(seed_indices)
        for _ in range(n):
            probs = self.forward(indices[-SEQ_LEN:])
            p = probs[:, 0]
            p = np.exp(np.log(p + 1e-9) / temperature)
            p /= p.sum()
            next_idx = np.random.choice(len(p), p=p)
            indices.append(next_idx)
        return ''.join(idx2char[i] for i in indices)

    @property
    def num_params(self):
        return sum(w.size for w in [self.Wf,self.Wi,self.Wg,self.Wo,
                                     self.bf,self.bi,self.bg,self.bo,self.Wy,self.by])

def train(model, X_train, y_train, X_test, y_test, epochs=12, name="Model"):
    train_losses, test_losses = [], []
    times = []

    for epoch in range(epochs):
        t0 = time.time()
        perm = np.random.permutation(len(X_train))
        epoch_loss = 0.0

        for k, idx in enumerate(perm):
            model.forward(X_train[idx])
            loss = model.backward(y_train[idx])
            epoch_loss += loss

        # Test loss
        test_loss = 0.0
        for idx in range(len(X_test)):
            model.forward(X_test[idx])
            test_loss += -np.log(model.probs[y_test[idx], 0] + 1e-9)

        avg_train = epoch_loss / len(X_train)
        avg_test  = test_loss  / len(X_test)
        elapsed   = time.time() - t0

        train_losses.append(float(avg_train))
        test_losses.append(float(avg_test))
        times.append(elapsed)

        # Perplexity = exp(cross-entropy loss)
        ppl = np.exp(avg_test)
        print(f"  [{name}] Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f} | "
              f"Perplexity: {ppl:.2f} | Time: {elapsed:.1f}s")

    return train_losses, test_losses, times


HIDDEN = 64
EPOCHS = 12

rnn  = VanillaRNN(vocab_size, HIDDEN, vocab_size, lr=0.005)
lstm = LSTM      (vocab_size, HIDDEN, vocab_size, lr=0.005)

print(f"\nVanilla RNN Parameters : {rnn.num_params:,}")
print(f"LSTM Parameters        : {lstm.num_params:,}\n")

print("=" * 65)
print("TRAINING VANILLA RNN on Frankenstein text")
print("=" * 65)
rnn_train_loss, rnn_test_loss, rnn_times = train(
    rnn, X_train, y_train, X_test, y_test, EPOCHS, "RNN")

print("\n" + "=" * 65)
print("TRAINING LSTM on Frankenstein text")
print("=" * 65)
lstm_train_loss, lstm_test_loss, lstm_times = train(
    lstm, X_train, y_train, X_test, y_test, EPOCHS, "LSTM")


def accuracy(model, X_test, y_test):
    correct = 0
    for i in range(len(X_test)):
        probs = model.forward(X_test[i])
        pred  = int(np.argmax(probs))
        if pred == y_test[i]:
            correct += 1
    return correct / len(X_test)

print("\n" + "=" * 65)
print("FINAL EVALUATION")
print("=" * 65)

rnn_acc  = accuracy(rnn,  X_test, y_test)
lstm_acc = accuracy(lstm, X_test, y_test)

rnn_ppl  = np.exp(rnn_test_loss[-1])
lstm_ppl = np.exp(lstm_test_loss[-1])

print(f"\nRNN  — Test Loss: {rnn_test_loss[-1]:.4f}  |  "
      f"Perplexity: {rnn_ppl:.2f}  |  Accuracy: {rnn_acc*100:.1f}%")
print(f"LSTM — Test Loss: {lstm_test_loss[-1]:.4f}  |  "
      f"Perplexity: {lstm_ppl:.2f}  |  Accuracy: {lstm_acc*100:.1f}%")

# Text generation
SEED_TEXT = "I feel a cold northern breeze"
seed_idx  = encode(SEED_TEXT)

print("\n" + "=" * 65)
print("TEXT GENERATION (seed: 'I feel a cold northern breeze')")
print("=" * 65)

rnn_text  = rnn.sample (seed_idx, n=120, temperature=0.7)
lstm_text = lstm.sample(seed_idx, n=120, temperature=0.7)

print(f"\n[RNN  generated]:\n{rnn_text}\n")
print(f"[LSTM generated]:\n{lstm_text}\n")

# Improvement
loss_improvement = (rnn_test_loss[-1] - lstm_test_loss[-1]) / rnn_test_loss[-1] * 100
ppl_improvement  = (rnn_ppl - lstm_ppl) / rnn_ppl * 100
acc_improvement  = (lstm_acc - rnn_acc) / rnn_acc * 100

print("=" * 65)
print("SUMMARY — LSTM vs RNN improvement")
print("=" * 65)
print(f"  Test Loss  : RNN={rnn_test_loss[-1]:.4f}  LSTM={lstm_test_loss[-1]:.4f}  "
      f"→ LSTM better by {loss_improvement:.1f}%")
print(f"  Perplexity : RNN={rnn_ppl:.2f}   LSTM={lstm_ppl:.2f}   "
      f"→ LSTM better by {ppl_improvement:.1f}%")
print(f"  Accuracy   : RNN={rnn_acc*100:.1f}%    LSTM={lstm_acc*100:.1f}%    "
      f"→ LSTM better by {acc_improvement:.1f}%")
print(f"  Parameters : RNN={rnn.num_params:,}   LSTM={lstm.num_params:,}   "
      f"→ LSTM uses {lstm.num_params/rnn.num_params:.1f}× more")

# Save results for visualization
results = {
    "rnn_train_loss" : rnn_train_loss,
    "rnn_test_loss"  : rnn_test_loss,
    "lstm_train_loss": lstm_train_loss,
    "lstm_test_loss" : lstm_test_loss,
    "rnn_times"      : rnn_times,
    "lstm_times"     : lstm_times,
    "rnn_acc"        : rnn_acc,
    "lstm_acc"       : lstm_acc,
    "rnn_ppl"        : float(rnn_ppl),
    "lstm_ppl"       : float(lstm_ppl),
    "rnn_params"     : rnn.num_params,
    "lstm_params"    : lstm.num_params,
    "vocab_size"     : vocab_size,
    "text_length"    : len(FRANKENSTEIN_TEXT),
    "seq_len"        : SEQ_LEN,
    "hidden_size"    : HIDDEN,
    "rnn_text"       : rnn_text,
    "lstm_text"      : lstm_text,
}

with open("/home/claude/results.json", "w") as f:
    json.dump(results, f)
print("\nResults saved.")