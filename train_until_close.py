"""
train_until_close.py

Adds an interactive "fix this answer" loop on top of an existing Faith
instance: instead of just rating 0-5, you can supply the answer you
WANTED, and it will repeatedly fine-tune on that exact pair, regenerating
and checking similarity each round, until the output is close enough (or
it gives up after max_attempts).

Import this next to faith_core.py / your console script and call
train_until_close(ai, user_input, desired_answer) from your rating loop.
"""

import difflib
import torch
import torch.nn as nn
import torch.optim as optim

import faith_core as faith  # reuses your existing Faith/save_state/normalize


def _similarity(a, b):
    """Word-level similarity, 0.0-1.0. Not exact match -- 'close enough'."""
    a_words = faith.normalize(a).split()
    b_words = faith.normalize(b).split()
    return difflib.SequenceMatcher(None, a_words, b_words).ratio()


def _fine_tune_on_pair(ai, user_input, desired_answer, steps=3, lr=0.0005):
    """Like Faith.fine_tune(), but targets ONE specific pair instead of
    the last 64 dataset entries, so it actually converges on what you want
    rather than being diluted by everything else recently seen."""
    pair = [{"question": user_input, "answer": desired_answer}]
    X, Y = ai.prepare_data(pair)
    optimizer = optim.Adam(ai.model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    ai.model.train()
    for _ in range(steps):
        out = ai.model(X)
        B, T, V = out.shape
        loss = criterion(out.reshape(B * T, V), Y.reshape(B * T))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ai.model.parameters(), 1.0)
        optimizer.step()
    return loss.item()


def train_until_close(ai, user_input, desired_answer,
                       max_attempts=15, similarity_threshold=0.8,
                       steps_per_attempt=3, temperature=0.6, verbose=True):
    """Repeatedly fine-tune ai on (user_input, desired_answer) until a
    fresh generation is similar enough to what you wanted, or attempts
    run out. Always saves the desired pair to dataset + intent_memory
    at the end, regardless of whether generation converged, so exact
    repeats of user_input get the right answer immediately next time."""

    if not ai.model:
        print("Model not trained yet -- can't fine-tune. Train it first.")
        return None

    best_response = None
    best_score = -1.0

    for attempt in range(1, max_attempts + 1):
        loss = _fine_tune_on_pair(ai, user_input, desired_answer, steps=steps_per_attempt)
        response = ai.generate(user_input, temperature=temperature)
        score = _similarity(response, desired_answer)

        if verbose:
            print(f"  attempt {attempt}: loss={loss:.4f} similarity={score:.2f} -> '{response}'")

        if score > best_score:
            best_score, best_response = score, response

        if score >= similarity_threshold:
            if verbose:
                print(f"Close enough after {attempt} attempt(s).")
            break
    else:
        if verbose:
            print(f"Gave up after {max_attempts} attempts. "
                  f"Best so far ({best_score:.2f}): '{best_response}'")

    # Always remember the desired answer directly, so exact repeats of
    # this input are served correctly from intent_memory even if the
    # generative model itself never fully converged on it.
    ai.intent_memory.setdefault(faith.normalize(user_input), [])
    if desired_answer not in ai.intent_memory[faith.normalize(user_input)]:
        ai.intent_memory[faith.normalize(user_input)].append(desired_answer)
    ai.dataset.append({"question": user_input, "answer": desired_answer})
    faith.save_state(ai.dataset, ai.intent_memory)

    return best_response, best_score
