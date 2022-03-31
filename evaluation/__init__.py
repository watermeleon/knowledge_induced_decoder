from .bleu import Bleu
from .meteor import Meteor
from .rouge import Rouge
from .cider import Cider
from .tokenizer import PTBTokenizer

def compute_scores(gts, gen):
    metrics = (Bleu(), Rouge(), Cider())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores
    all_score["METEOR"] = 42
    all_scores["METEOR"] =[42]
    return all_score, all_scores
