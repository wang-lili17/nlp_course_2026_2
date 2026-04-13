#Text Classification with RNNs

import re, time
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ──────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ──────────────────────────────────────────────────────────
VOCAB_SIZE  = 10_000
EMBED_DIM   = 100
HIDDEN_DIM  = 128
NUM_LAYERS  = 1
BATCH_SIZE  = 8
LR          = 0.001
EPOCHS      = 10
MAX_LEN     = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"PyTorch : {torch.__version__}")
print(f"Device  : {DEVICE}\n")

INLINE_DATA = [
    (1, "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me. The first thing that struck me about Oz was its brutality and unflinching onscreen violence, which set in right from the word go."),
    (1, "A wonderful little production. The filming technique is very unassuming, very old Hollywood quality. The acting is very good, down to the bit parts. The music is great. The story and the screenplay are great."),
    (1, "I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air conditioned theater and watching this light-hearted comedy. The plot was implausible and a bit ridiculous, but it was intentional and worked perfectly."),
    (1, "Basically there's a family where a little boy thinks there's a zombie in his closet and his parents are fighting all the time. This movie is made by the same director as the Shining and the acting, writing and general artistry are quite astounding."),
    (1, "Petter Mattei's Love in the Time of Money is a visually stunning film with a very complex narrative. It is a masterpiece of story telling and is not for the average movie goer."),
    (1, "Probably my all-time favourite movie, a story of two friends and their adventures in a world where dreams can come true. The movie is based on a novel and the screenplay does an excellent job of translating the story to film."),
    (1, "I sure would like to see a resurrection of a classic TV show known as The Lone Ranger with a cast of excellent actors. The show was original, fresh and the storyline was excellent, not like today's trashy cable channels."),
    (1, "This film was absolutely brilliant. I was afraid this would be a cheap remake of the old Japanese horror movies but it was actually an interesting film with great acting, directing and special effects."),
    (1, "Being an African-American actor, I was truly amazed at the scope of this film. The director and writer did an excellent job of capturing the essential pain and beauty of the African-American experience throughout history."),
    (1, "I've seen this story before but my husband and I were completely fans and then we were hook. Our 14 year old son didn't stop yelling at the screen, something very unusual for him. Great movie, must watch."),
    (1, "Superbly written, the screenplay is absolutely marvelous. The cast and crew did an amazing job. Best film I have seen in years and years and years. The script is clever and witty. Highly recommended."),
    (1, "This is a masterpiece. I was mesmerized throughout the entire film. The cinematography is stunning, the performances are unforgettable, and the story is deeply moving. A must-see for any serious film lover."),
    (1, "What a delightful surprise this film turned out to be. Initially I wasn't sure about the premise but after ten minutes I was completely sold. The chemistry between the leads is electric and the direction is masterful."),
    (1, "I cannot recommend this film enough. Every scene is crafted with care and intelligence. The dialogue is sharp, the performances are nuanced, and the ending is both surprising and deeply satisfying."),
    (1, "A truly exceptional piece of cinema. The director manages to balance humor and pathos with incredible skill. The lead performance is career-best work, and the supporting cast is uniformly excellent. Bravo."),
    (0, "I am sorry, I don't understand what all the people who loved this movie saw in it. It seems very contrived and highly improbable. The story involves some dangerous criminals who take over a bus full of people."),
    (0, "This film is nothing short of a disaster. The plot makes no sense, the acting is wooden and unconvincing, and the direction seems to have been done by someone who has never watched a movie before. Avoid at all costs."),
    (0, "Absolutely dreadful. I cannot believe this film was made and released in theaters. The screenplay is nonsensical, the characters are one-dimensional, and the special effects look like they were done on a budget of about fifty dollars."),
    (0, "I wanted to like this movie, I really did. But after sitting through two hours of tedious dialogue and completely unbelievable plot twists, I walked out feeling cheated and bored. A tremendous waste of time."),
    (0, "This is easily one of the worst films I have ever seen. The acting is atrocious, the story is non-existent, and the direction is completely incoherent. How this got a theatrical release is beyond me."),
    (0, "Painful to watch. Every single element of this production fails. The script is full of clichés and plot holes, the performances are laughably bad, and the editing makes the whole thing feel like a fever dream."),
    (0, "I have never been so bored in a cinema in my entire life. This film drags along at a snail's pace, going nowhere and meaning nothing. Not even the usually reliable lead actor can save this sinking ship."),
    (0, "Terrible in every way. The story is derivative, the characters are unlikeable, and the tone is all over the place. The director clearly had no vision and the film suffers catastrophically as a result. Zero stars."),
    (0, "What a colossal disappointment. Given the talent involved I expected something at least competent, but this is a mess from start to finish. Plot threads are introduced and abandoned, characters make no sense whatsoever."),
    (0, "One of the most unpleasant viewing experiences I can remember. The film is ugly, mean-spirited, and completely without redemption. I genuinely cannot find a single positive thing to say about it."),
    (1, "Stunning performances all around. The film moves at a perfect pace and never outstays its welcome. I left the theater feeling uplifted and moved. This is what cinema is supposed to be."),
    (1, "A rare gem. Intelligent, funny, and deeply humane, this film reminded me why I love movies. The script crackles with wit and the direction is assured without ever being showy. Loved every minute."),
    (0, "Avoid. The filmmakers mistake loudness for excitement and confusion for complexity. By the midpoint I had completely stopped caring about any of the characters. A thoroughly unpleasant and unrewarding experience."),
    (0, "The worst kind of Hollywood garbage. Cynically manufactured to appeal to the lowest common denominator. The script feels like it was written by a committee over a long weekend and the direction is equally uninspired."),
    (1, "Beautifully shot and brilliantly acted. This film deserves all the praise it has received and more. The central relationship feels utterly real and the emotional payoff in the final act is genuinely devastating in the best way."),

    (1, "Genuinely one of the finest films to come out of Hollywood in a decade. Sharp writing, superb direction, and a lead performance that is simply breathtaking. I cannot recommend this highly enough."),
    (0, "A muddled and thoroughly boring attempt at drama. The film tries very hard to be profound but succeeds only in being pretentious and dull. A complete waste of the considerable talents involved."),
    (1, "Funny, touching, and wonderfully crafted. This film has everything you could want from an evening at the movies. The ensemble cast is outstanding and the director keeps things moving at exactly the right pace."),
    (0, "Dreadful. Simply dreadful. The script is embarrassingly bad, the acting is wooden at best, and the direction has all the imagination of a bus timetable. Hard to believe anyone thought this was worth making."),
    (1, "A masterwork of subtle storytelling. Every detail is carefully considered and every performance is pitch-perfect. This is the kind of film that stays with you long after the credits roll."),
    (0, "Completely incoherent. By the third act I had absolutely no idea what was happening or why I should care. The film seems to be deliberately confusing in an attempt to disguise its fundamental emptiness."),
    (1, "Warm, wise, and wonderfully entertaining. This film has a big heart and the confidence to wear it on its sleeve. I laughed, I cried, and I left the cinema feeling genuinely grateful for the experience."),
    (0, "A cynical cash-grab dressed up as meaningful cinema. The script is riddled with clichés, the characters are cardboard cut-outs, and the ending is both predictable and deeply unsatisfying. Save your money."),
    (1, "One of those rare films that gets everything right. The screenplay is intelligent without being alienating, the direction is confident without being flashy, and the performances are uniformly excellent."),
    (0, "Incompetent filmmaking at its most baffling. Characters behave in ways no real human being would ever behave, the plot defies all logic, and the climax is laughably anticlimactic. An astonishing failure."),
    (1, "I went in with low expectations and came out completely blown away. This film is funny, moving, and genuinely surprising. The lead performance is a revelation and the direction is inspired."),
    (0, "This film made me angry. Not in a good thought-provoking way, but in a I-cannot-believe-I-just-wasted-two-hours way. The plot is nonsensical, the characters are hateful, and the filmmaking is utterly incompetent."),
    (1, "Perfect in almost every way. The script is beautifully constructed, the performances are deeply felt, and the direction is assured and inventive. This is the work of a filmmaker at the very top of their game."),
    (0, "Embarrassingly bad. The dialogue is so clunky and unnatural that every scene becomes an ordeal to sit through. The lead actor looks visibly uncomfortable with the material and I cannot blame them one bit."),
    (1, "An absolute triumph. Funny and heartbreaking in equal measure, this film demonstrates that popular cinema can also be genuinely great art. I will be thinking about this one for a very long time indeed."),
    (0, "Soulless and tedious. This film goes through the motions of being a thriller without generating a single moment of genuine tension or excitement. A professionally made but utterly lifeless piece of product."),
    (1, "This exceeded every expectation I had. The storytelling is inventive, the world-building is meticulous, and the emotional core of the film is rock solid. Outstanding work from everyone involved."),
    (0, "A film so bad it almost becomes interesting as a study in how not to make movies. Almost, but not quite. Mostly it is just boring, ugly, and completely without merit. Remarkable in all the wrong ways."),
    (1, "Extraordinary. I have rarely seen a film that balances so many tones so effortlessly. Moving without being maudlin, funny without being silly, and exciting without being exhausting. A genuine achievement."),
    (0, "Where to begin with this catastrophe. The script has more holes than a colander, the performances range from adequate to actively terrible, and the direction seems to actively work against any kind of engagement."),
]

# ── split 80/20 train/test ──
np.random.seed(42)
idx = np.random.permutation(len(INLINE_DATA))
split = int(0.8 * len(idx))
train_idx, test_idx = idx[:split], idx[split:]

raw_train = [INLINE_DATA[i] for i in train_idx]
raw_test  = [INLINE_DATA[i] for i in test_idx]

train_texts  = [t for _, t in raw_train]
train_labels = [l for l, _ in raw_train]
test_texts   = [t for _, t in raw_test]
test_labels  = [l for l, _ in raw_test]

print(f"Inline sample  →  train: {len(train_texts)}, test: {len(test_texts)}")
print("NOTE: Replace INLINE_DATA section with full IMDB loader for real training.\n")

def simple_tokenize(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.lower().split()

counter = Counter()
for text in train_texts:
    counter.update(simple_tokenize(text))

vocab_words = [w for w, _ in counter.most_common(VOCAB_SIZE - 2)]
word2idx    = {w: i + 2 for i, w in enumerate(vocab_words)}
PAD_IDX, UNK_IDX = 0, 1
print(f"Vocabulary size : {len(word2idx) + 2}")

def encode(text):
    return [word2idx.get(tok, UNK_IDX) for tok in simple_tokenize(text)]

def pad_or_truncate(seq, maxlen=MAX_LEN):
    seq = seq[:maxlen]
    return seq + [PAD_IDX] * (maxlen - len(seq))

x_train = np.array([pad_or_truncate(encode(t)) for t in train_texts], dtype=np.int64)
x_test  = np.array([pad_or_truncate(encode(t)) for t in test_texts],  dtype=np.int64)
y_train = np.array(train_labels, dtype=np.float32)
y_test  = np.array(test_labels,  dtype=np.float32)
print(f"x_train : {x_train.shape},  x_test : {x_test.shape}\n")

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
    batch_size=BATCH_SIZE, num_workers=0
)

class SentimentClassifier(nn.Module):
    def __init__(self, rnn_type: str):
        super().__init__()
        assert rnn_type in ("RNN", "LSTM", "GRU")
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_IDX)
        cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
        self.rnn = cls(EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.fc      = nn.Linear(HIDDEN_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        if self.rnn_type == "LSTM":
            _, (h, _) = self.rnn(emb)
        else:
            _, h = self.rnn(emb)
        return self.sigmoid(self.fc(h[-1])).squeeze(1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def train_model(rnn_type: str):
    model     = SentimentClassifier(rnn_type).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    print(f"\n{'='*60}")
    print(f"  Training {rnn_type}  (params: {model.n_params():,})")
    print(f"{'='*60}")
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Train Acc':>10}  {'Time':>6}")
    print(f"  {'─'*38}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = correct = total = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            total_loss += loss.item() * bs
            correct    += ((preds >= 0.5).float() == yb).sum().item()
            total      += bs
        print(f"  {epoch:>5}  {total_loss/total:>8.4f}  {correct/total:>10.4f}  "
              f"{time.time()-t0:>5.2f}s")
    return model

def evaluate_model(model, rnn_type: str):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            all_preds.extend((model(xb) >= 0.5).cpu().int().tolist())
            all_labels.extend(yb.int().tolist())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n  ── Test Results: {rnn_type} ───────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Confusion Matrix  (rows=true, cols=predicted):")
    print(f"               Pred NEG   Pred POS")
    print(f"    True NEG     {cm[0,0]:>5}      {cm[0,1]:>5}")
    print(f"    True POS     {cm[1,0]:>5}      {cm[1,1]:>5}")
    return dict(model=rnn_type, accuracy=acc, precision=prec,
                recall=rec, f1=f1, cm=cm)

results = []
for arch in ("RNN", "LSTM", "GRU"):
    mdl = train_model(arch)
    results.append(evaluate_model(mdl, arch))

print(f"\n\n{'='*60}")
print("  FINAL COMPARISON — Test Set")
print(f"{'='*60}")
print(f"  {'Model':<8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
print(f"  {'─'*50}")
for r in results:
    print(f"  {r['model']:<8} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
          f"{r['recall']:>10.4f} {r['f1']:>8.4f}")
best = max(results, key=lambda r: r["f1"])
print(f"\n  Best model by F1 : {best['model']}  (F1 = {best['f1']:.4f})")
print(f"{'='*60}\n")