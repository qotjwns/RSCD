import os
import torch
import numpy as np
import time
import random


from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor



def save_checkpoint(args, data_name, epoch, encoder, encoder_feat, decoder, encoder_optimizer,
                encoder_feat_optimizer, decoder_optimizer, best_bleu4):
    """
    모델 체크포인트를 저장합니다.

    :param data_name: 처리된 데이터셋의 기본 이름
    :param epoch: 에폭 번호
    :param epochs_since_improvement: BLEU-4 개선 이후 지난 에폭 수
    :param encoder: 인코더 모델
    :param decoder: 디코더 모델
    :param encoder_optimizer: 미세 조정 시 인코더 가중치를 업데이트하는 옵티마이저
    :param decoder_optimizer: 디코더 가중치를 업데이트하는 옵티마이저
    :param bleu4: 현재 에폭의 검증 BLEU-4 점수
    :param is_best: 현재 체크포인트가 지금까지의 최선인지 여부
    """
    state = {'epoch': epoch,
             'best_bleu-4': best_bleu4,
             'encoder': encoder,
             'encoder_feat': encoder_feat,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'encoder_feat_optimizer': encoder_feat_optimizer,
             'decoder_optimizer': decoder_optimizer,
             }
    #filename = 'checkpoint_' + data_name + '_' + args.network + '.pth.tar'
    path = args.savepath #'./models_checkpoint/mymodel/3-times/'
    if os.path.exists(path)==False:
        os.makedirs(path)
        # 현재 체크포인트가 최선이면 이후 낮은 성능의 체크포인트가 덮어쓰지 않도록 사본을 저장합니다.
    torch.save(state, os.path.join(path, 'BEST_' + data_name))

    # torch.save(state, os.path.join(path, 'checkpoint_' + data_name +'_epoch_'+str(epoch) + '.pth.tar'))


def accuracy(scores, targets, k):
    """
    예측값과 정답 라벨로 top-k 정확도를 계산합니다.

    :param scores: 모델이 출력한 점수
    :param targets: 정답 라벨
    :param k: top-k 정확도의 k 값
    :return: top-k 정확도
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0차원 텐서
    return correct_total.item() * (100.0 / batch_size)


def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    # COCO 캡션 평가기는 샘플 id를 key로 갖는 딕셔너리를 기대합니다.
    hypo = {
        idx: [' '.join(str(x) for x in hypo)]
        for idx, hypo in enumerate(hypotheses)
    }
    ref = {
        idx: [' '.join(str(x) for x in reft) for reft in reftmp]
        for idx, reftmp in enumerate(references)
    }
    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        #print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict


def clip_gradient(optimizer, grad_clip):
    """
    그래디언트 폭주를 막기 위해 역전파로 계산된 그래디언트를 클리핑합니다.

    :param optimizer: 클리핑할 그래디언트를 가진 옵티마이저
    :param grad_clip: 클리핑 기준값
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
                
def adjust_learning_rate(optimizer, shrink_factor):
    """
    지정한 비율만큼 학습률을 줄입니다.

    :param optimizer: 학습률을 줄일 옵티마이저
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

class AverageMeter(object):
    """
    지표의 최근값, 평균, 합계, 개수를 추적합니다.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_file_str():
    ISOTIMEFORMAT='%Y-%m-%d-%H-%M-%S'
    string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string #+ '-{}'.format(random.randint(1, 10000))

def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()



