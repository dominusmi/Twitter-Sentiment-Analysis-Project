import numpy as np

def read_test(testset):
  id_gts = {}
  with open(testset, 'r') as fh:
    for line in fh:
      fields = line.split('\t')
      tweetid = fields[0]
      gt = fields[1]

      id_gts[tweetid] = gt

  return id_gts

def confusion(id_preds, testset, classifier):
  id_gts = read_test(testset)

  gts = []
  for m, c1 in id_gts.items():
    if not c1 in gts:
      gts.append(c1)

  gts = ['positive', 'negative', 'neutral']

  conf = {}
  for c1 in gts:
    conf[c1] = {}
    for c2 in gts:
      conf[c1][c2] = 0

  for tweetid, gt in id_gts.items():
    if tweetid in id_preds:
      pred = id_preds[tweetid]
    else:
      pred = 'neutral'
    conf[pred][gt] += 1

  print(''.ljust(12) + '  '.join(gts))

  confusion_matrix = np.zeros((3,3))

  for i, c1 in enumerate(gts):
    print(c1.ljust(12), end='')
    for j, c2 in enumerate(gts):
      if sum(conf[c1].values()) > 0:
        confusion_matrix[i,j] = (conf[c1][c2] / float(sum(conf[c1].values())))
        print('%.3f     ' % (conf[c1][c2] / float(sum(conf[c1].values()))), end='')
      else:
        print('0.000     ', end='')
    print('')

  print('')

  return confusion_matrix

def evaluate(id_preds, testset, classifier):
  id_gts = read_test(testset)

  acc_by_class = {}
  for gt in ['positive', 'negative', 'neutral']:
    acc_by_class[gt] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

  catf1s = {}

  ok = 0
  for tweetid, gt in id_gts.items():
    if tweetid in id_preds:
      pred = id_preds[tweetid]
    else:
      pred = 'neutral'

    if gt == pred:
      ok += 1
      acc_by_class[gt]['tp'] += 1
    else:
      acc_by_class[gt]['fn'] += 1
      acc_by_class[pred]['fp'] += 1

  catcount = 0
  itemcount = 0
  macro = {'p': 0, 'r': 0, 'f1': 0}
  micro = {'p': 0, 'r': 0, 'f1': 0}
  semevalmacro = {'p': 0, 'r': 0, 'f1': 0}

  microtp = 0
  microfp = 0
  microtn = 0
  microfn = 0
  for cat, acc in acc_by_class.items():
    catcount += 1

    microtp += acc['tp']
    microfp += acc['fp']
    microtn += acc['tn']
    microfn += acc['fn']

    p = 0
    if (acc['tp'] + acc['fp']) > 0:
      p = float(acc['tp']) / (acc['tp'] + acc['fp'])

    r = 0
    if (acc['tp'] + acc['fn']) > 0:
      r = float(acc['tp']) / (acc['tp'] + acc['fn'])

    f1 = 0
    if (p + r) > 0:
      f1 = 2 * p * r / (p + r)

    catf1s[cat] = f1

    n = acc['tp'] + acc['fn']

    npred = acc['tp'] + acc['fp']

    macro['p'] += p
    macro['r'] += r
    macro['f1'] += f1

    if cat in ['positive', 'negative']:
      semevalmacro['p'] += p
      semevalmacro['r'] += r
      semevalmacro['f1'] += f1

    itemcount += n

  micro['p'] = float(microtp) / float(microtp + microfp)
  micro['r'] = float(microtp) / float(microtp + microfn)
  micro['f1'] = 2 * float(micro['p']) * micro['r'] / float(micro['p'] + micro['r'])

  macrop = macro['p'] / catcount
  macror = macro['r'] / catcount
  macrof1 = macro['f1'] / catcount

  microp = micro['p']
  micror = micro['r']
  microf1 = micro['f1']

  semevalmacrop = semevalmacro['p'] / 2
  semevalmacror = semevalmacro['r'] / 2
  semevalmacrof1 = semevalmacro['f1'] / 2

  print(testset + ' (' + classifier + '): %.3f' % semevalmacrof1)
  return (semevalmacrop, semevalmacror, semevalmacrof1)
