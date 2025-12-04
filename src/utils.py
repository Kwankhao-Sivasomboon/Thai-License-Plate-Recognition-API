import torch
import numpy as np

def best_path_decode(log_probs, int_to_char, blank=0):
    preds = log_probs.argmax(-1)
    results = []
    for seq in preds:
        text = []
        prev = None
        for p in seq.tolist():
            if p != blank and p != prev:
                text.append(int_to_char[str(p)])
            prev = p
        results.append("".join(text))
    return results

def beam_search_decode(log_probs, int_to_char, beam_width=3, blank=0):
    T, C = log_probs.shape
    beam = [(0, [], -1)]
    for t in range(T):
        next_beam = []
        probs = log_probs[t]
        top_k_probs, top_k_indices = torch.topk(torch.exp(probs), beam_width)
        for score, seq, last_char in beam:
            if not seq:
                for p, idx in zip(top_k_probs, top_k_indices):
                    new_score = score + np.log(p.item() + 1e-8)
                    next_beam.append((new_score, seq + [idx.item()], idx.item()))
            else:
                for p, idx in zip(top_k_probs, top_k_indices):
                    new_score = score + np.log(p.item() + 1e-8)
                    next_beam.append((new_score, seq + [idx.item()], idx.item()))
        beam = sorted(next_beam, key=lambda x: x[0], reverse=True)[:beam_width]
    best_score, best_seq, _ = beam[0]
    text = []
    prev = None
    for p in best_seq:
        if p != blank and p != prev:
            text.append(int_to_char[str(p)])
        prev = p
    return "".join(text)