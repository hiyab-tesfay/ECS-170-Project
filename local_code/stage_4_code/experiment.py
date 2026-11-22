"""Reproducible training, checkpointing, metrics, and autoregressive decoding."""

import csv
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .Dataset_Loader import detokenize, encode, tokenize
from .Method_RNN import TextClassifier, TextGenerator


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(request):
    if request == 'auto':
        request = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if request == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    if request == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable in this execution context')
    return torch.device(request)


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def collate_classification(batch):
    seq, labels = zip(*batch)
    return pad_sequence([torch.tensor(s) for s in seq], batch_first=True), torch.tensor(labels)


def collate_generation(batch):
    seq = [torch.tensor(s) for s, _ in batch]
    padded = pad_sequence(seq, batch_first=True)
    return padded[:, :-1], padded[:, 1:]


def loaders(data, task, batch_size, seed):
    collate = collate_classification if task == 'classification' else collate_generation
    return {split: DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train',
                              generator=torch.Generator().manual_seed(seed),
                              collate_fn=collate, num_workers=0)
            for split, dataset in data.items()}


def pass_epoch(model, batches, device, task, optimizer=None, clip=1.0):
    model.train(optimizer is not None)
    total_loss, total, correct = 0., 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=0 if task == 'generation' else -100, reduction='sum')
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            if task == 'classification':
                logits = model(x)
                summed_loss = criterion(logits, y)
                count = y.numel()
                correct += int((logits.argmax(-1) == y).sum().item())
            else:
                logits, _ = model(x)
                summed_loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                count = int(y.ne(0).sum().item())
                correct += int(((logits.argmax(-1) == y) & y.ne(0)).sum().item())
            loss = summed_loss / count
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += summed_loss.item()
            total += count
    return {'loss': total_loss / total, 'accuracy': correct / total}


def save_history(path, history):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(history)


def train(task, data, vocab, config, output_dir, *, epochs, batch_size, learning_rate,
          patience, seed, device, pretrained=None, weight_decay=1e-5, clip=1.0):
    output_dir = Path(output_dir)
    if (output_dir / 'metrics.json').exists():
        raise FileExistsError(f'Completed experiment already exists: {output_dir}. Choose a new output directory.')
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    cls = TextClassifier if task == 'classification' else TextGenerator
    model = cls(vocab_size=len(vocab), pretrained=pretrained, **config).to(device)
    settings = dict(task=task, model=config, epochs=epochs, batch_size=batch_size,
                    learning_rate=learning_rate, patience=patience, seed=seed,
                    device=str(device), weight_decay=weight_decay, gradient_clip=clip,
                    optimizer='Adam', checkpoint_selection='minimum validation cross entropy',
                    learning_rate_schedule='validation plateau, patience 2, factor 0.5' if task == 'classification' else 'cosine to 10% of initial learning rate',
                    initialization='Xavier input/output; gate-wise orthogonal recurrent; zero bias; '
                                   + ('pretrained GloVe embeddings' if pretrained is not None else 'random N(0, 1) embeddings'),
                    python=platform.python_version(), torch=str(torch.__version__),
                    numpy=np.__version__, platform=platform.platform(), threads=torch.get_num_threads(),
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    json_write(output_dir / 'config.json', settings)
    json_write(output_dir / 'vocabulary.json', vocab)
    batches = loaders(data, task, batch_size, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
                 if task == 'classification' else
                 torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.1))
    history, best, best_epoch, stale = [], float('inf'), 0, 0
    sync(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']
        training = pass_epoch(model, batches['train'], device, task, optimizer, clip)
        validation = pass_epoch(model, batches['validation'], device, task)
        sync(device)
        row = dict(epoch=epoch, training_loss=training['loss'], validation_loss=validation['loss'],
                   training_accuracy=training['accuracy'], validation_accuracy=validation['accuracy'],
                   learning_rate=lr, seconds=time.perf_counter() - epoch_start)
        history.append(row)
        save_history(output_dir / 'history.csv', history)
        if validation['loss'] < best - 1e-5:
            best, best_epoch, stale = validation['loss'], epoch, 0
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'config': settings, 'vocabulary': vocab, 'epoch': epoch}, output_dir / 'best.pt')
        else:
            stale += 1
        if task == 'classification':
            scheduler.step(validation['loss'])
        else:
            scheduler.step()
        print(f'{task}/{config["cell"]} epoch {epoch}/{epochs} '
              f'loss={training["loss"]:.4f} val_loss={validation["loss"]:.4f} '
              f'accuracy={training["accuracy"]:.4f} val_accuracy={validation["accuracy"]:.4f} '
              f'seconds={row["seconds"]:.1f}', flush=True)
        if patience > 0 and stale >= patience:
            break
    duration = time.perf_counter() - start
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'config': settings, 'vocabulary': vocab, 'epoch': history[-1]['epoch']}, output_dir / 'last.pt')
    model.load_state_dict(torch.load(output_dir / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
    metrics = dict(best_epoch=best_epoch, best_validation_loss=best,
                   epochs_run=len(history), training_seconds=duration,
                   parameters=settings['parameters'], device=str(device))
    if task == 'classification':
        labels, predictions, probabilities = [], [], []
        model.eval()
        with torch.no_grad():
            for x, y in batches['test']:
                probs = model(x.to(device)).softmax(-1).cpu().numpy()
                labels.extend(y.tolist())
                predictions.extend(probs.argmax(-1).tolist())
                probabilities.extend(probs[:, 1].tolist())
        report = classification_report(labels, predictions, target_names=['negative', 'positive'], output_dict=True, zero_division=0)
        acc = float(accuracy_score(labels, predictions))
        metrics.update(accuracy=acc, meets_85_percent=acc >= 0.85,
                       precision_macro=report['macro avg']['precision'], recall_macro=report['macro avg']['recall'],
                       f1_macro=report['macro avg']['f1-score'], classification_report=report,
                       confusion_matrix=confusion_matrix(labels, predictions).tolist(), test_examples=len(labels))
        # Wilson interval quantifies uncertainty of this fixed test evaluation.
        n, z = len(labels), 1.96
        centre = (acc + z*z/(2*n)) / (1 + z*z/n)
        half = z * math.sqrt(acc*(1-acc)/n + z*z/(4*n*n)) / (1 + z*z/n)
        metrics['accuracy_wilson_95'] = [centre-half, centre+half]
        with (output_dir / 'predictions.csv').open('w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['test_index', 'true_label', 'predicted_label', 'positive_probability'])
            writer.writerows(zip(range(len(labels)), labels, predictions, probabilities))
    else:
        metrics['validation_perplexity'] = math.exp(min(best, 50))
        metrics['validation_token_accuracy'] = history[best_epoch - 1]['validation_accuracy']
    json_write(output_dir / 'metrics.json', metrics)
    return model, metrics


def load_checkpoint(path, device):
    # Only load checkpoints generated by this project; restricted tensor loader.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    cls = TextClassifier if checkpoint['config']['task'] == 'classification' else TextGenerator
    model = cls(vocab_size=len(checkpoint['vocabulary']), **checkpoint['config']['model'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint


def generate(model, vocab, prompt, device, *, max_words=60, temperature=0.7, top_k=10, seed=42):
    if len(prompt.split()) != 3:
        raise ValueError('Supply exactly three starting words separated by spaces')
    if temperature <= 0 or top_k < 1 or max_words < 1:
        raise ValueError('temperature, top_k and max_words must be positive')
    mapping = {w: i for i, w in enumerate(vocab)}
    words = tokenize(prompt)
    unknown = [w for w in words if w not in mapping]
    ids = [2] + encode(words, mapping)
    rng = torch.Generator().manual_seed(seed)
    output = list(words)
    model.eval()
    with torch.no_grad():
        logits, state = model(torch.tensor([ids], device=device))
        for step in range(max_words):
            scores = logits[0, -1].float().cpu().clone()
            scores[:3] = -float('inf')  # Never emit padding, unknown, or BOS.
            if step < 3:
                scores[3] = -float('inf')
            # Mild repetition penalty; inference uses only the model and prompt.
            for t in set(ids[-12:]):
                if scores[t] > 0:
                    scores[t] /= 1.1
                else:
                    scores[t] *= 1.1
            scores /= temperature
            values, indices = scores.topk(min(top_k, int(torch.isfinite(scores).sum())))
            token = indices[torch.multinomial(values.softmax(-1), 1, generator=rng)].item()
            if token == 3:
                break
            ids.append(token)
            output.append(vocab[token])
            logits, state = model(torch.tensor([[token]], device=device), state)
    return {'prompt': prompt, 'text': detokenize(output), 'unknown_prompt_tokens': unknown,
            'temperature': temperature, 'top_k': top_k, 'seed': seed,
            'ended_with_eos': token == 3, 'generated_tokens': len(output) - len(words)}


def generation_samples(model, vocab, records, device):
    from difflib import SequenceMatcher
    prompts = ['what did the', 'why did the', 'what do you', 'how do you', 'a man walks', 'last night i']
    samples = []
    for prompt in prompts:
        for temperature in [0.5, 0.8]:
            for seed in [42, 43, 44]:
                sample = generate(model, vocab, prompt, device, temperature=temperature, seed=seed)
                generated = tokenize(sample['text'])
                nearest = max(records['train'], key=lambda r: SequenceMatcher(None, generated, r['tokens']).ratio())
                sample['nearest_training_joke'] = nearest['text']
                sample['training_similarity'] = SequenceMatcher(None, generated, nearest['tokens']).ratio()
                sample['exact_training_match'] = tuple(generated) in {tuple(r['tokens']) for r in records['train']}
                sample['distinct_bigram_ratio'] = len(set(zip(generated, generated[1:]))) / max(1, len(generated)-1)
                samples.append(sample)
    return samples
