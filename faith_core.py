import torch
import torch.nn as nn
import torch.optim as optim
import re
import json
import random
import requests
from collections import Counter
import nltk
from nltk.corpus import wordnet
 
# WordNet's corpus data isn't bundled by `pip install nltk` -- it has to be
# downloaded separately. This makes sure it's present on first run, whether
# that's your local machine or a fresh deploy (Render/HidenCloud/etc.) that
# has never downloaded it before.
try:
    wordnet.synsets("test")
except LookupError:
    print("Downloading NLTK WordNet corpus (first run only)...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
 
save_path = "chat_dataset.json"
 
 
# -----------------------------
# Helper: Normalize input
# -----------------------------
def normalize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = text.strip()
    return text
 
 
# -----------------------------
# Persistence: dataset + intent_memory together
# -----------------------------
def load_state():
    try:
        with open(save_path, "r") as f:
            raw = json.load(f)
            if isinstance(raw, list):
                return raw, {}
            return raw.get("dataset", []), raw.get("intent_memory", {})
    except FileNotFoundError:
        return [], {}
 
 
def save_state(dataset, intent_memory):
    with open(save_path, "w") as f:
        json.dump({"dataset": dataset, "intent_memory": intent_memory}, f)
 
 
# -----------------------------
# Fallback dictionary source: WordNet only covers content words (nouns,
# verbs, adjectives, adverbs) -- it deliberately has no entries for
# function words like "you", "how", "the", etc. This free, no-key API
# (sourced from Wiktionary, a general-purpose dictionary) fills that gap.
# https://dictionaryapi.dev
# -----------------------------
def _lookup_wiktionary(word):
    try:
        resp = requests.get(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}",
            timeout=5,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        for entry in data:
            for meaning in entry.get("meanings", []):
                for definition in meaning.get("definitions", []):
                    if definition.get("definition"):
                        return definition["definition"]
    except (requests.RequestException, ValueError, KeyError, IndexError):
        pass
    return None
 
 
# -----------------------------
# Word-sense selection: WordNet often has several unrelated senses for
# one word (e.g. "are" = the verb "to be" OR a rare unit of land area).
# Naively taking the first synset can surface a technically-correct but
# very unintuitive definition. This also checks the word's base/lemma
# form per part of speech (so "are"/"is"/"was" -> "be", "running" -> "run"
# are found at all, since WordNet stores inflected verb forms under their
# base lemma, not as separate headwords) and prefers a verb sense when
# one exists, since short common words are usually meant as verbs in
# everyday chat. This is a heuristic, not true disambiguation -- it won't
# be right 100% of the time, but should be right far more often than
# picking blindly.
# -----------------------------
def _best_synset(word):
    candidates = list(wordnet.synsets(word))
    for pos in (wordnet.VERB, wordnet.NOUN, wordnet.ADJ, wordnet.ADV):
        base = wordnet.morphy(word, pos)
        if base and base != word:
            candidates.extend(wordnet.synsets(base, pos=pos))
    if not candidates:
        return None
    verb_candidates = [s for s in candidates if s.pos() == 'v']
    return (verb_candidates or candidates)[0]
 
 
# -----------------------------
# Shared WordNet definition lookup
# Used by BOTH Faith.generate() (step 2) and the !explain command, so
# there is exactly one place that knows how to look up / cache a
# definition instead of two separate implementations that can drift.
# Returns (definition_text_or_None, was_already_known_bool)
# -----------------------------
def lookup_definition(ai, word):
    word = word.strip().lower()
    dict_key = f"definition of {word}"
 
    if dict_key in ai.intent_memory and ai.intent_memory[dict_key]:
        return random.choice(ai.intent_memory[dict_key]), True
 
    definition = None
    best = _best_synset(word)
    if best:
        definition = best.definition()
    else:
        # WordNet has nothing (common for function words) -- try the
        # general-purpose fallback before giving up.
        definition = _lookup_wiktionary(word)
 
    if definition:
        ai.intent_memory.setdefault(dict_key, [])
        ai.intent_memory[dict_key].append(definition)
        save_state(ai.dataset, ai.intent_memory)
        return definition, False
 
    return None, False
 
 
# -----------------------------
# Transformer Block
# -----------------------------
class FaithBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x
 
 
# -----------------------------
# MiniFaith Model
# -----------------------------
class MiniFaith(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, max_len=40):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([FaithBlock(d_model, nhead) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
 
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = h.permute(1, 0, 2)
 
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )
 
        for block in self.blocks:
            h = block(h, attn_mask=mask)
        h = h.permute(1, 0, 2)
        return self.fc_out(h)
 
 
# -----------------------------
# Faith AI Wrapper
# -----------------------------
class Faith:
    def __init__(self, max_len=40, data=None, intent_memory=None):
        self.vocab = []
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.model = None
        self.max_len = max_len
        self.dataset = data if data else []
        self.intent_memory = intent_memory if intent_memory else {}
 
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
 
    def encode(self, text):
        return [self.word_to_idx.get(w, 1) for w in self.preprocess(text)]
 
    def decode(self, seq):
        return ' '.join(self.idx_to_word.get(i, '<UNK>') for i in seq)
 
    def pad(self, seq, pad_val=0):
        return seq[:self.max_len] + [pad_val] * (self.max_len - len(seq))
 
    def build_vocab(self, data, max_vocab=10000):
        words = []
        for item in data:
            words += self.preprocess(item['question'])
            words += self.preprocess(item['answer'])
        counts = Counter(words)
        self.vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + [w for w, _ in counts.most_common(max_vocab)]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
 
    def prepare_data(self, data):
        X, Y = [], []
        for item in data:
            q_ids = self.encode(item['question'])
            a_ids = self.encode(item['answer'])
            full = [2] + q_ids + a_ids + [3]
 
            inp = full[:-1]
            tgt = full[1:]
 
            prompt_len = 1 + len(q_ids)
            tgt = [(-100 if i < prompt_len - 1 else t) for i, t in enumerate(tgt)]
 
            X.append(self.pad(inp, pad_val=0))
            Y.append(self.pad(tgt, pad_val=-100))
        return torch.tensor(X), torch.tensor(Y)
 
    def train(self, data, epochs=20, batch_size=8, lr=0.0003, patience=15, grad_clip=1.0):
        self.build_vocab(data)
        X, Y = self.prepare_data(data)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )
        self.model = MiniFaith(len(self.vocab), max_len=self.max_len)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
 
        best_loss = float('inf')
        patience_counter = 0
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                out = self.model(xb)
                B, T, V = out.shape
                loss = criterion(out.reshape(B * T, V), yb.reshape(B * T))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            if avg_loss < best_loss - 1e-3:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
 
    def fine_tune(self, steps=1, lr=0.0005, batch_size=8, grad_clip=1.0):
        if not self.model or len(self.dataset) < 5:
            return
        X, Y = self.prepare_data(self.dataset[-64:])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.model.train()
        for _ in range(steps):
            for xb, yb in loader:
                out = self.model(xb)
                B, T, V = out.shape
                loss = criterion(out.reshape(B * T, V), yb.reshape(B * T))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                optimizer.step()
 
    def generate(self, text, max_new_tokens=40, temperature=1.0):
        key = normalize(text)
        
        # 1. Check intent memory cache first
        if key in self.intent_memory and self.intent_memory[key]:
            return random.choice(self.intent_memory[key])
            
        # 2. Look up word meanings if asked or if a single word is entered
        words_in_input = self.preprocess(text)
        if len(words_in_input) == 1 or "mean" in words_in_input:
            target_word = words_in_input[-1] if words_in_input[-1] != "mean" else words_in_input[0]
            definition, already_known = lookup_definition(self, target_word)
            if definition:
                if already_known:
                    return f"I remember learning that: {definition}"
                return f"Looking that up... it means: {definition}"
 
        if not self.model:
            return "Train me first."
 
        # 3. Neural network transformer fallback
        self.model.eval()
        seq = [2] + self.encode(text)
        generated = []
        for _ in range(max_new_tokens):
            if len(seq) >= self.max_len:
                break
            x = torch.tensor([seq])
            with torch.no_grad():
                logits_all = self.model(x)
                logits = logits_all[0, -1]
                logits[0] = -float('inf')
            probs = torch.softmax(logits / temperature, dim=0)
            idx = torch.multinomial(probs, 1).item()
            if idx == 3:
                break
            generated.append(idx)
            seq.append(idx)
        self.model.train()
        return self.decode(generated)
 
    def learn_from_conversation(self, user_input, ai_response, reward=5):
        key = normalize(user_input)
        self.intent_memory.setdefault(key, [])
        self.intent_memory[key].append(ai_response)
 
        # 4. Automate looking up unknown words via WordNet
        try:
            all_words = self.preprocess(user_input) + self.preprocess(ai_response)
            meaningful_words = [w for w in set(all_words) if len(w) > 3]
            
            for word in meaningful_words:
                definition, already_known = lookup_definition(self, word)
                if definition and not already_known:
                    print(f"[Faith Learned] {word}: {definition}")
        except Exception as e:
            print(f"[WordNet Learning Error] {e}")
 
        if reward > 0:
            self.dataset.append({"question": user_input, "answer": ai_response})
            self.fine_tune(steps=reward)
 
        save_state(self.dataset, self.intent_memory)
 
 
def init_ai(max_len=40):
    print("Initializing Faith Neural Network...")
    saved_data, saved_intent_memory = load_state()
    ai = Faith(max_len=max_len, data=saved_data, intent_memory=saved_intent_memory)
 
    training_data = [] if saved_data else [{"question": "null", "answer": "null"}]
    full_data = training_data + saved_data
    ai.train(full_data, epochs=20, batch_size=8, patience=15)
    print("Faith Training cicle Complete.")
    return ai

