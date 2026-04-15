import json, sys
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

records = []
for split in ['train', 'validation', 'test']:
    with open('datasets/dataset_outputs/evaluation_dataset/splits/' + split + '.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for r in data:
            r['split'] = split
            records.append(r)

lengths = []
for r in records:
    toks = tokenizer(r['prompt_text'], truncation=False)
    lengths.append((len(toks['input_ids']), r['label'], r['split']))

lens = np.array([l[0] for l in lengths])

print('=== Token Length Distribution (all %d samples) ===' % len(lens))
print('%-15s %8s' % ('Statistic', 'Value'))
print('-' * 25)
for name, val in [('Min', np.min(lens)), ('p10', np.percentile(lens,10)),
                  ('p25', np.percentile(lens,25)), ('Median', np.median(lens)),
                  ('Mean', np.mean(lens)), ('p75', np.percentile(lens,75)),
                  ('p90', np.percentile(lens,90)), ('p95', np.percentile(lens,95)),
                  ('p99', np.percentile(lens,99)), ('Max', np.max(lens))]:
    print('%-15s %8.1f' % (name, val))

print()
print('=== Token Length Buckets ===')
buckets = [(0,32),(32,64),(64,128),(128,192),(192,256),(256,384),(384,512),(512,1024)]
print('%-15s %7s %7s %8s' % ('Range', 'Count', '%', 'Cumul%'))
print('-' * 40)
cumul = 0
for lo, hi in buckets:
    c = int(np.sum((lens >= lo) & (lens < hi)))
    pct = c / len(lens) * 100
    cumul += pct
    print('%3d-%-5d       %7d %6.1f%% %7.1f%%' % (lo, hi, c, pct, cumul))

fit_192 = int(np.sum(lens <= 192))
print()
print('Samples <= 192 tokens: %d / %d (%.1f%%)' % (fit_192, len(lens), fit_192/len(lens)*100))
print('Samples >  192 tokens: %d / %d (%.1f%%)' % (len(lens)-fit_192, len(lens), (len(lens)-fit_192)/len(lens)*100))

print()
print('=== Samples > 192 tokens by class ===')
for label in ['benign', 'jailbreak', 'harmful']:
    class_lens = [l[0] for l in lengths if l[1] == label]
    over = sum(1 for l in class_lens if l > 192)
    total = len(class_lens)
    print('%-12s %5d / %5d over 192 (%.1f%%)' % (label, over, total, over/total*100))

sys.stdout.flush()
