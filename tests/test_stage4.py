"""Behavioral checks for leakage, masking, causality, and reusable checkpoints."""
import csv
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from local_code.stage_4_code.Dataset_Loader import classification_data, encode, generation_data, tokenize, vocabulary
from local_code.stage_4_code.Method_RNN import TextClassifier, TextGenerator
from local_code.stage_4_code.experiment import generate, load_checkpoint, seed_everything, train
from local_code.stage_4_code.Dataset_Loader import SequenceDataset


class Stage4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cleaning_preserves_negation_and_html_entities(self):
        self.assertEqual(tokenize("<br />I don’t like it &amp; NOT good!"),
                         ['i', "don't", 'like', 'it', '&', 'not', 'good', '!'])

    def test_vocab_never_adds_unseen_validation_word(self):
        vocab = vocabulary([['training', 'only']], 20)
        mapping = {w: i for i, w in enumerate(vocab)}
        self.assertEqual(encode(['unseen'], mapping), [1])
        self.assertNotIn('unseen', vocab)

    def test_review_loader_keeps_test_words_out_of_training_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'text_classification'
            for split in ['train', 'test']:
                for label in ['pos', 'neg']:
                    folder = base / split / label
                    folder.mkdir(parents=True)
                    for i in range(20 if split == 'train' else 2):
                        (folder / f'{i}.txt').write_text('training review words' if split == 'train' else 'exclusiveunseentesttoken')
            data, vocab, metadata, manifests = classification_data(directory)
            self.assertNotIn('exclusiveunseentesttoken', vocab)
            self.assertEqual(metadata['splits'], {'train': 36, 'validation': 4, 'test': 4})
            self.assertFalse(set(manifests['train']) & set(manifests['validation']))
            self.assertTrue(all(seq == [1] for seq in data['test'].sequences))

    def test_review_truncation_preserves_head_and_tail(self):
        mapping = {str(i): i + 4 for i in range(10)}
        self.assertEqual(encode(list(mapping), mapping, 4), [4, 5, 12, 13])

    def test_generation_split_precedes_examples_and_removes_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'text_generation'
            folder.mkdir()
            with (folder / 'data').open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['ID', 'Joke'])
                for i in range(30):
                    w.writerow([i, f'why did number {i} cross the road?'])
                w.writerow([31, 'why did number 1 cross the road?'])
            data, vocab, meta, records = generation_data(directory)
            train_ids = {r['id'] for r in records['train']}
            val_ids = {r['id'] for r in records['validation']}
            self.assertFalse(train_ids & val_ids)
            self.assertEqual(meta['removed_duplicate_or_short_rows'], 1)
            for ds in data.values():
                for seq in ds.sequences:
                    self.assertEqual((seq[0], seq[-1]), (2, 3))

    def test_all_cells_forward_and_gradients(self):
        for cell in ['rnn', 'lstm', 'gru']:
            with self.subTest(cell=cell):
                model = TextClassifier(20, embedding_dim=5, hidden_size=8, cell=cell)
                out = model(torch.tensor([[4, 5, 0], [6, 7, 8]]))
                self.assertEqual(tuple(out.shape), (2, 2))
                F.cross_entropy(out, torch.tensor([0, 1])).backward()
                self.assertTrue(torch.isfinite(model.output.weight.grad).all())
                self.assertEqual(model.embedding.weight.grad[0].abs().sum(), 0)

    def test_generator_does_not_see_future_tokens(self):
        for cell in ['rnn', 'lstm', 'gru']:
            model = TextGenerator(20, embedding_dim=5, hidden_size=8, cell=cell, dropout=0).eval()
            a, _ = model(torch.tensor([[2, 4, 5, 6]]))
            b, _ = model(torch.tensor([[2, 4, 9, 8]]))
            torch.testing.assert_close(a[:, :2], b[:, :2])

    def test_reverse_encoder_matches_packed_bidirectional_reference(self):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        from local_code.stage_4_code.Method_RNN import recurrent
        for cell in ['rnn', 'lstm', 'gru']:
            model = TextClassifier(20, embedding_dim=5, hidden_size=8, cell=cell, dropout=0).eval()
            reference = recurrent(cell, 5, 8, 1, 0, bidirectional=True).eval()
            state = reference.state_dict()
            for name, tensor in model.encoder.state_dict().items():
                state[name] = tensor
            for name, tensor in model.reverse_encoder.state_dict().items():
                state[name + '_reverse'] = tensor
            reference.load_state_dict(state)
            tokens = torch.tensor([[4, 5, 6], [7, 8, 0]])
            embeddings = model.embedding(tokens)
            packed = pack_padded_sequence(embeddings, [3, 2], batch_first=True, enforce_sorted=False)
            expected, _ = reference(packed)
            expected, _ = pad_packed_sequence(expected, batch_first=True)
            mask = tokens.ne(0).unsqueeze(-1)
            avg = (expected * mask).sum(1) / mask.sum(1)
            maximum = expected.masked_fill(~mask, -1e4).max(1).values
            expected_logits = model.output(torch.cat([avg, maximum], dim=1))
            torch.testing.assert_close(model(tokens), expected_logits, atol=1e-6, rtol=1e-5)

    def test_classifier_ignores_added_padding_in_both_directions(self):
        for cell in ['rnn', 'lstm', 'gru']:
            model = TextClassifier(20, embedding_dim=5, hidden_size=8, cell=cell, dropout=0).eval()
            a = model(torch.tensor([[4, 5, 6]]))
            b = model(torch.tensor([[4, 5, 6, 0, 0]]))
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_generator_stateful_decoding_matches_whole_sequence(self):
        for cell in ['rnn', 'lstm', 'gru']:
            model = TextGenerator(20, embedding_dim=5, hidden_size=8, cell=cell, dropout=0).eval()
            tokens = torch.tensor([[2, 4, 5, 6]])
            whole, _ = model(tokens)
            state, parts = None, []
            for t in tokens.unbind(1):
                output, state = model(t[:, None], state)
                parts.append(output)
            torch.testing.assert_close(whole, torch.cat(parts, dim=1), atol=1e-6, rtol=1e-5)

    def test_checkpoint_round_trip_and_three_word_prompt(self):
        data = {'train': SequenceDataset([[2, 4, 5, 6, 3]] * 4),
                'validation': SequenceDataset([[2, 4, 5, 6, 3]] * 2)}
        vocab = ['<pad>', '<unk>', '<bos>', '<eos>', 'what', 'did', 'the']
        with tempfile.TemporaryDirectory() as directory:
            model, metrics = train('generation', data, vocab,
                dict(cell='gru', embedding_dim=5, hidden_size=8, layers=1, dropout=0),
                directory, epochs=1, batch_size=2, learning_rate=0.01, patience=0,
                seed=42, device=torch.device('cpu'))
            loaded, checkpoint = load_checkpoint(Path(directory) / 'best.pt', torch.device('cpu'))
            tokens = torch.tensor([[2, 4, 5]])
            model.eval()
            torch.testing.assert_close(model(tokens)[0], loaded(tokens)[0])
            sample = generate(loaded, vocab, 'what did the', torch.device('cpu'), max_words=5)
            self.assertTrue(sample['text'].lower().startswith('what did the'))
            self.assertNotIn('<pad>', sample['text'])
            with self.assertRaises(ValueError):
                generate(loaded, vocab, 'two words', torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
