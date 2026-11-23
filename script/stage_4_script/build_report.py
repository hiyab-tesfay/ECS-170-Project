"""Build the five-page report from measured results and reviewed decoded samples."""
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'result/stage_4_result'
OUT = ROOT / 'report/stage_4/Stage_4_RNN_Report.pdf'
INK = colors.HexColor('#18354a')
BODY = ParagraphStyle('Body', fontName='Helvetica', fontSize=10, leading=11.7, spaceAfter=6)
HEAD = ParagraphStyle('Head', parent=BODY, fontName='Helvetica-Bold', textColor=INK, spaceBefore=8, spaceAfter=5, keepWithNext=True)
TITLE = ParagraphStyle('Title', parent=HEAD, spaceBefore=0, spaceAfter=9)
CELL = ParagraphStyle('Cell', parent=BODY, spaceAfter=0, leading=11)
CAPTION = ParagraphStyle('Caption', parent=BODY, textColor=colors.HexColor('#425766'), spaceAfter=7)


def read(path):
    return json.loads(Path(path).read_text())


def hist(folder):
    with (folder / 'history.csv').open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def p(text, style=BODY):
    return Paragraph(text, style)


def literal(text):
    return escape(text).replace('\n', '<br/>')


def table(rows, widths):
    data = [[p(literal(str(c)), CELL) for c in row] for row in rows]
    obj = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    obj.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eff3')),
        ('TEXTCOLOR', (0, 0), (-1, 0), INK),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.HexColor('#9ab0bf')),
        ('LINEBELOW', (0, 1), (-1, -1), 0.25, colors.HexColor('#dbe3e8')),
    ]))
    return obj


def figure(name, height):
    path = RESULTS / 'figures' / name
    from PIL import Image as PILImage
    with PILImage.open(path) as image:
        width, h = image.size
    target_width = 6.5 * inch
    actual_height = target_width * h / width
    if actual_height > height * inch:
        target_width *= height * inch / actual_height
        actual_height = height * inch
    return Image(str(path), width=target_width, height=actual_height, hAlign='CENTER')


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#c5d1d9'))
    canvas.line(inch, 0.64 * inch, 7.5 * inch, 0.64 * inch)
    canvas.setFont('Helvetica', 10)
    canvas.setFillColor(colors.HexColor('#425766'))
    canvas.drawString(inch, 0.43 * inch, 'ECS 170 | Stage 4 | Fall 2026')
    canvas.drawRightString(7.5 * inch, 0.43 * inch, f'{doc.page} / 5')
    canvas.restoreState()


def main():
    cells = ['rnn', 'lstm', 'gru']
    cm = {c: read(RESULTS / 'classification/comparison' / c / 'metrics.json') for c in cells}
    gm = {c: read(RESULTS / 'generation/comparison' / c / 'metrics.json') for c in cells}
    ch = {c: hist(RESULTS / 'classification/comparison' / c) for c in cells}
    gh = {c: hist(RESULTS / 'generation/comparison' / c) for c in cells}
    review = read(RESULTS / 'manual_generation_review.json')
    cd, gd = read(RESULTS / 'classification_dataset.json'), read(RESULTS / 'generation_dataset.json')
    ablation = read(RESULTS / 'classification/random_embeddings/gru/metrics.json')
    benchmark = read(RESULTS / 'device_benchmark.json')
    best_cell = max(cells, key=lambda c: cm[c]['accuracy'])
    best = cm[best_cell]
    story = []
    add = story.append

    # PAGE 1: task, architecture and data.
    add(p('TEXT CLASSIFICATION AND GENERATION WITH RECURRENT NETWORKS', TITLE))
    add(p('Section 1: Task Description', HEAD))
    add(p('We study two sequence-learning tasks: binary sentiment classification of movie reviews and '
          'word-by-word completion of short jokes from exactly three starting words. Separate models are trained '
          'with vanilla RNN, LSTM and GRU units for each task, giving six required experiments. The instructor\'s '
          'generation CSV specifically contains jokes, so the examples follow that dataset instruction.'))
    add(p(f'The strongest classifier is {best_cell.upper()} with <b>{best["accuracy"]:.2%}</b> test accuracy, '
          f'exceeding the 85% requirement by {(best["accuracy"]-.85)*100:.2f} percentage points. '
          'Generation is assessed through training curves and direct inspection of decoded completions.'))
    add(p('Section 2: Model Description', HEAD))
    add(figure('architectures.png', 1.65))
    add(p('Figure 1. Shared model structure; the recurrent cell is replaced while other settings remain fixed.', CAPTION))
    add(p('The vanilla RNN updates a tanh hidden state from the current embedding and previous state. LSTM '
          'adds a separate cell state and input, forget and output gates; GRU uses reset and update gates. '
          'The gated variants can preserve useful context across more steps.'))
    add(p('Classification embeds each word in 50 dimensions and applies independent 128-unit forward and reverse '
          'encoders. Their states are concatenated (256 dimensions), then masked mean and maximum pooling produce '
          'a 512-dimensional review representation. Dropout precedes a linear 512-to-2 sentiment head. Reverse '
          'processing reverses only valid words, so padding cannot change a review\'s prediction.'))
    add(p('Generation uses a causal 256-unit recurrent layer and a 256-to-vocabulary output at every step. '
          'A beginning token starts each joke; an end token teaches the model when to stop. Teacher forcing '
          'trains next-word prediction; inference feeds generated words back into the hidden state. No future '
          'words or corpus lookup are available to the decoder.'))
    add(p('Section 3: Experiment Settings', HEAD))
    add(p('3.1 Dataset Description', HEAD))
    add(p(f'The supplied IMDb corpus contains 25,000 training and 25,000 test reviews, balanced between positive '
          'and negative. A stratified seed-42 split reserves 2,500 training reviews for validation, leaving '
          '22,500 for fitting. The supplied test set is kept separate. HTML is removed and case/apostrophes '
          'normalized; punctuation and negation are retained. The vocabulary has 20,000 entries including four '
          'special tokens and is fitted only to training reviews. Reviews longer than 300 tokens retain their '
          f'first and last 150 tokens ({cd["truncated_training_fraction"]:.1%} of training reviews).'))
    add(p(f'The joke CSV has {gd["source_rows"]:,} rows. Removing {gd["removed_duplicate_or_short_rows"]} normalized '
          f'duplicate/too-short rows leaves {gd["unique_usable_rows"]:,}; splitting whole jokes before encoding '
          f'gives {gd["splits"]["train"]:,} training and {gd["splits"]["validation"]} validation jokes. '
          f'The separate training vocabulary has {gd["vocabulary_size"]:,} entries. No training joke exceeds '
          'the 96-token cap. Exact split identifiers are saved.'))
    add(PageBreak())

    # PAGE 2: reproducible settings, evaluation and classification convergence.
    add(p('3.2 Detailed Experimental Setups', HEAD))
    add(table([
        ['Setting', 'Classification', 'Generation'],
        ['Embedding / hidden units', '50 / 128 per direction', '50 / 256 causal'],
        ['Layers / dropout', '1 / 0.4', '1 / 0.2'],
        ['Batch / initial Adam lr', '64 / 0.001', '64 / 0.003'],
        ['Epoch cap / stopping', '12 / patience 4', '100 / fixed cap'],
        ['Learning-rate schedule', 'Halve on validation plateau', 'Cosine to 0.0003'],
    ], [1.8*inch, 2.35*inch, 2.35*inch]))
    add(Spacer(1, 6))
    add(p(f'Both tasks initialize embeddings from the supplied 2024 Wiki/Gigaword GloVe 50d ZIP, then fine-tune '
          f'them. Vocabulary coverage is {cd["glove"]["coverage"]:.2%} for reviews and '
          f'{gd["glove"]["coverage"]:.2%} for jokes. Missing vectors use N(0, 0.05); PAD is zero. '
          'Input/output weights use Xavier uniform initialization; recurrent matrices are orthogonal per gate; '
          'biases are zero. Adam weight decay is 0.00001 and gradient norms are clipped at 1.0. '
          'Python, NumPy and PyTorch use seed 42. Runs use PyTorch 2.2.2, Python 3.11 and two CPU threads. '
          + 'Classifier devices: ' + ', '.join(f'{c.upper()} on {cm[c]["device"].upper()}' for c in cells) +
          '. Generation runs use CPU.'))
    add(p('Classifiers are selected by minimum validation cross entropy and evaluated once on all 25,000 test '
          'reviews after fitting. Generation preserves both the best-validation and epoch-100 checkpoints: '
          'validation loss estimates held-out next-word performance, while the final checkpoint tests learned '
          'completion of familiar joke openings. These are reported separately.'))
    add(p('3.3 Evaluation Metrics', HEAD))
    add(p('For classification, accuracy is correct predictions / N. For each class, precision = TP/(TP+FP), '
          'recall = TP/(TP+FN), and F1 = 2PR/(P+R); macro scores average the two classes equally. Confusion '
          'matrices and Wilson 95% intervals are also saved. For generation, PAD-masked token cross entropy '
          'and perplexity = exp(loss) summarize prediction performance. Manual inspection checks grammar, '
          'prompt relevance, repetition, completion and similarity to training jokes. There is no required '
          'generation accuracy or perplexity threshold.'))
    add(p('3.4 Source Code', HEAD))
    add(p('<link href="https://github.com/hiyab-tesfay/ECS-170-Project/tree/stage4" color="#2468a0">'
          'github.com/hiyab-tesfay/ECS-170-Project - stage4 branch</link>. '
          'Implementation: local_code/stage_4_code/. Executables: script/stage_4_script/. '
          'The REPRODUCE guide includes exact commands, inference examples and dependency versions. '
          'The local submission archive includes trained weights for direct manual testing.'))
    add(p('3.5 Training Convergence Plot', HEAD))
    add(figure('classification_learning_curves.png', 1.85))
    add(p('Figure 2. Review classification: training (solid), validation (dashed), and selected epoch (vertical). '
          'Selection uses validation loss, not test accuracy.', CAPTION))
    add(PageBreak())

    # PAGE 3: measured convergence and performance.
    add(p('3.5 Training Convergence Plot (continued)', HEAD))
    add(figure('generation_learning_curves.png', 1.9))
    add(p('Figure 3. Joke generation: teacher-forced training loss falls while held-out loss eventually rises. '
          'This exposes overfitting rather than implying that memorized completions generalize.', CAPTION))
    add(p('3.6 Model Performance', HEAD))
    rows = [['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1', 'Epoch']]
    for c in cells:
        m = cm[c]
        rows.append([c.upper(), f'{m["accuracy"]:.2%}', f'{m["precision_macro"]:.2%}',
                     f'{m["recall_macro"]:.2%}', f'{m["f1_macro"]:.2%}', m['best_epoch']])
    add(table(rows, [1.0*inch, 1.08*inch, 1.12*inch, 1.1*inch, 1.1*inch, 1.1*inch]))
    add(Spacer(1, 5))
    low, high = best['accuracy_wilson_95']
    add(p(f'The {best_cell.upper()} model correctly predicts {round(best["accuracy"]*25000):,} / 25,000 reviews; '
          f'its Wilson 95% interval is [{low:.2%}, {high:.2%}]. '
          f'{sum(cm[c]["accuracy"] >= .85 for c in cells)} of the three models meet 85%. '
          'All metrics can be recomputed from saved prediction CSVs. The interval describes test-sample '
          'uncertainty and does not measure variation across training seeds.'))
    add(figure('confusion_matrices.png', 1.75))
    add(p('Figure 4. Counts on the supplied test split; rows are actual labels, columns predicted labels.', CAPTION))
    rows = [['Generator', 'Best val. epoch', 'Val. loss', 'Perplexity', 'Final train loss']]
    for c in cells:
        m = gm[c]
        rows.append([c.upper(), m['best_epoch'], f'{m["best_validation_loss"]:.3f}',
                     f'{m["validation_perplexity"]:.1f}', f'{gh[c][-1]["training_loss"]:.3f}'])
    add(table(rows, [1.0*inch, 1.4*inch, 1.1*inch, 1.25*inch, 1.75*inch]))
    add(Spacer(1, 5))
    add(p(f'Generation validation has {gd["validation_unknown_rate"]:.1%} unknown tokens under the training-only '
          'vocabulary, limiting interpretation of perplexity. Token scores include UNK targets and are not '
          'a measure of humor or semantic correctness. The next page presents actual model-decoded samples '
          'and compares them with training text.'))
    add(PageBreak())

    # PAGE 4: genuine outputs and manual evaluation.
    add(p('3.6 Model Performance: Manual Generation Tests', HEAD))
    add(p('Each model was decoded from six three-word prompts, two temperatures (0.5 and 0.8) and three seeds '
          '(42-44), producing 36 samples per checkpoint and 72 per model. The examples below are representative '
          'readable outputs from the final checkpoint, selected after inspection; they do not establish that '
          'every sample is readable. All decoded outputs, including failures, are retained.'))
    comparisons = []
    for c in cells:
        folder = RESULTS / 'generation/comparison' / c
        early = sum(s['exact_training_match'] for s in read(folder / 'samples_best.json'))
        late = sum(s['exact_training_match'] for s in read(folder / 'samples_last.json'))
        comparisons.append(f'{c.upper()} {early} to {late}')
    add(p('Exact normalized training matches per 36 samples, best-validation to final checkpoint: '
          + '; '.join(comparisons) + '. Capitalization differences are ignored by this comparison.'))
    for c in cells:
        item = review[c]
        sample = item['sample']
        add(p(f'{c.upper()} | Prompt: {literal(sample["prompt"])} | '
              f'Temperature {sample["temperature"]}, seed {sample["seed"]}', HEAD))
        add(p('<b>Model output:</b> ' + literal(sample['text'])))
        add(p('<b>Nearest training joke:</b> ' + literal(sample['nearest_training_joke'])))
        add(p(f'<b>Assessment:</b> {literal(item["assessment"])} '
              f'Token sequence similarity = {sample["training_similarity"]:.3f}; '
              f'exact training match = {str(sample["exact_training_match"]).lower()}.'))
    add(p('Correctness and Limitations', HEAD))
    add(p('A readable completion should preserve the opening, form a grammatical sentence or question/answer, '
          'and end without uncontrolled repetition. A plausible punchline is assessed manually; the model '
          'has no factual verifier. Similarity uses token-level SequenceMatcher against the nearest training '
          'joke and is a descriptive overlap measure, not a correctness score. Training matches demonstrate '
          'learned completion but also memorization.'))
    add(p('The final checkpoints generally fit familiar openings more closely than the early best-validation '
          'checkpoints. Rising validation loss confirms limited generalization on this small corpus. Novel '
          'or out-of-vocabulary prompts can produce incoherent text. The generation script reports unknown '
          'prompt tokens; it suppresses special output tokens and stops at EOS or 60 new tokens. The saved '
          'sample sweeps preserve both successful and unsuccessful cases for inspection.'))
    add(PageBreak())

    # PAGE 5: ablations, device experiment and reproducibility.
    add(p('3.7 Ablation Studies', HEAD))
    add(p('Replacing RNN with LSTM or GRU is the primary controlled architecture comparison. The split, '
          'vocabulary, embedding size, hidden size, optimizer, stopping rule and seed are shared within each '
          'task; parameter counts differ because the gated cells contain additional matrices. Classifier '
          'devices are chosen using the benchmark; device-dependent numerical differences are a limitation.'))
    rows = [['Cell', 'Classifier parameters', 'Generator parameters', 'Classifier epochs']]
    for c in cells:
        rows.append([c.upper(), f'{cm[c]["parameters"]:,}', f'{gm[c]["parameters"]:,}', cm[c]['epochs_run']])
    add(table(rows, [0.65*inch, 2.0*inch, 2.0*inch, 1.85*inch]))
    add(p('Additional embedding ablation', HEAD))
    add(p('A second GRU classifier replaces GloVe with random trainable embeddings, holding the data split, '
          'architecture and training policy fixed. This isolates the contribution of pretrained initialization.'))
    add(table([
        ['GRU embedding initialization', 'Test accuracy', 'Macro F1', 'Best epoch'],
        ['GloVe 50d', f'{cm["gru"]["accuracy"]:.2%}', f'{cm["gru"]["f1_macro"]:.2%}', cm['gru']['best_epoch']],
        ['Random', f'{ablation["accuracy"]:.2%}', f'{ablation["f1_macro"]:.2%}', ablation['best_epoch']],
    ], [2.5*inch, 1.4*inch, 1.3*inch, 1.3*inch]))
    delta = (cm['gru']['accuracy'] - ablation['accuracy']) * 100
    add(p(f'GloVe changes test accuracy by {delta:+.2f} percentage points relative to random initialization. '
          'This is one paired-seed comparison, not evidence of a statistically reliable average effect.'))
    add(p('Decoding-temperature comparison', HEAD))
    rows = [['Final generator', 'T=0.5 exact / 18', 'T=0.8 exact / 18', 'EOS reached / 36']]
    for c in cells:
        samples = read(RESULTS / 'generation/comparison' / c / 'samples_last.json')
        counts = [sum(s['exact_training_match'] for s in samples if s['temperature'] == t) for t in [0.5, 0.8]]
        rows.append([c.upper(), counts[0], counts[1], sum(s['ended_with_eos'] for s in samples)])
    add(table(rows, [1.25*inch, 1.75*inch, 1.75*inch, 1.75*inch]))
    add(p('Lower temperature concentrates probability on likely words; higher temperature samples more '
          'alternatives. Exact-match counts indicate memorization, not output quality. Readability varies '
          'across prompts and seeds, so manual examples complement these diagnostics.'))
    add(p('Optional CPU / Apple GPU comparison', HEAD))
    rows = [['Cell', 'CPU seconds / batch', 'MPS seconds / batch', 'Faster device']]
    for c in cells:
        values = {r['device']: r.get('seconds_per_training_batch') for r in benchmark if r['cell'] == c}
        cpu, gpu = values.get('cpu'), values.get('mps')
        rows.append([c.upper(), f'{cpu:.3f}' if cpu else 'Unavailable', f'{gpu:.3f}' if gpu else 'Unsupported',
                     'CPU' if gpu is None or cpu < gpu else 'MPS'])
    add(table(rows, [0.65*inch, 1.95*inch, 1.95*inch, 1.95*inch]))
    add(p('The benchmark synchronizes device work, excludes two warm-up batches and times five updates on '
          'the same padded 64-by-240 inputs. Device choices are reported in Section 3.2. These timings '
          'are specific to this hardware, software and model size; GPU availability does not guarantee a speedup.'))
    add(p('Reproducibility and scope', HEAD))
    add(p('Saved artifacts include configurations, vocabularies, split manifests, GloVe checksum, histories, '
          'weights, predictions and unedited sample sweeps. Behavioral tests cover padding equivalence, causal '
          'generation, split isolation, gradients and checkpoint reloads. Remaining limits are one training '
          'seed, review truncation, no independent generation test corpus, substantial joke overfitting and '
          'manual sample selection. The data and report template are the instructor-supplied materials.'))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=(8.5*inch, 11*inch),
                            rightMargin=inch, leftMargin=inch, topMargin=.7*inch,
                            bottomMargin=.82*inch, title='ECS 170 Stage 4: Recurrent Text Models',
                            author='ECS 170 Project', subject='Measured RNN, LSTM and GRU experiments')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    import fitz
    pdf = fitz.open(OUT)
    if len(pdf) != 5:
        raise RuntimeError(f'Expected five pages, produced {len(pdf)}. Adjust layout before delivery.')
    destination = ROOT / 'tmp/stage4_report_qa'
    destination.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(pdf, 1):
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(destination / f'page-{i}.png')
    print(f'Created {OUT} ({len(pdf)} pages); rendered QA images to {destination}')


if __name__ == '__main__':
    main()
