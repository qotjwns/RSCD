import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type = str, default = 'LEVIR_MCI', help= 'the name of the dataset')
parser.add_argument('--word_count_threshold', default=5, type=int)

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<UNK>': 1,
  '<START>': 2,
  '<END>': 3,
}
DATA_PATH_ROOT = 'xxxx'
def main(args):
    if args.dataset == 'LEVIR_MCI':
        input_captions_json = f'{DATA_PATH_ROOT}\Levir-MCI-dataset\LevirCCcaptions.json'
        input_image_dir = f'{DATA_PATH_ROOT}\Levir-MCI-dataset\images'
        input_vocab_json = ''
        output_vocab_json = 'vocab.json'
        save_dir = './data/LEVIR_MCI/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir + 'tokens/')):
        os.makedirs(os.path.join(save_dir + 'tokens/'))
    print('Loading captions')
    assert args.dataset in {'LEVIR_MCI'}

    if args.dataset == 'LEVIR_MCI':
        with open(input_captions_json, 'r') as f:
            data = json.load(f)
        # 각 이미지의 경로와 캡션을 읽습니다.
        max_length = -1
        all_cap_tokens = []
        for img in data['images']:
            captions = []    
            for c in img['sentences']:
                # 단어 빈도 갱신
                assert len(c['raw']) > 0, 'error: some image has no caption'
                captions.append(c['raw'])
            tokens_list = []
            for cap in captions:
                cap_tokens = tokenize(cap,
                                    add_start_token=True,
                                    add_end_token=True,
                                    punct_to_keep=[';', ','],
                                    punct_to_remove=['?', '.'])
                tokens_list.append(cap_tokens)
                max_length = max(max_length, len(cap_tokens))
            all_cap_tokens.append((img['filename'], tokens_list))

        # 토큰화된 캡션을 txt 파일로 저장
        print('Saving captions')
        for img, tokens_list in all_cap_tokens:
            i = img.split('.')[0]
            token_len = len(tokens_list)
            tokens_list = json.dumps(tokens_list)
            f = open(os.path.join(save_dir + 'tokens/' + i + '.txt'), 'w')
            f.write(tokens_list)
            f.close()


        #각 이미지 쌍에 5개의 주석이 있으므로 학습 목록 생성에 두 가지 전략을 사용할 수 있습니다.
        # a: token_id[0:4]를 직접 지정해 각 토큰 목록이 특정 캡션에 대응하도록 학습 목록을 생성합니다.
        # b: 학습 중 5개 캡션 중 하나를 무작위로 선택합니다.
          
            if i.split('_')[0] == 'train':
               f = open(os.path.join(save_dir + 'train' + '.txt'), 'a')
               f.write(img + '\n')
               f.close

            # if i.split('_')[0] == 'train':
            #     f = open(os.path.join(save_dir + 'train' + '.txt'), 'a')
            #     for j in range(token_len):
            #         f.write(img + '-' + str(j) + '\n')
            #     f.close

            elif i.split('_')[0] == 'val':
                f = open(os.path.join(save_dir + 'val' + '.txt'), 'a')
                f.write(img + '\n')
                f.close()
            elif i.split('_')[0] == 'test':
                f = open(os.path.join(save_dir + 'test' + '.txt'), 'a')
                f.write(img + '\n')
                f.close()

    print('max_length of the dataset:', max_length)
    # 어휘 사전을 새로 만들거나 디스크에서 불러옵니다.
    if input_vocab_json == '':
        print('Building vocab')
        word_freq = build_vocab(all_cap_tokens, args.word_count_threshold)
    else:
        print('Loading vocab')
        with open(input_vocab_json, 'r') as f:
            word_freq = json.load(f)
    if output_vocab_json != '':
        with open(os.path.join(save_dir + output_vocab_json), 'w') as f:
            json.dump(word_freq, f)


def tokenize(s, delim=' ',add_start_token=True, 
    add_end_token=True, punct_to_keep=None, punct_to_remove=None):
    """
    문자열 s를 문자열 토큰 목록으로 변환합니다. by
    지정한 구분자로 나누며, 필요하면 특정 문장부호를 유지하거나 제거할 수 있습니다.
    시작/종료 토큰을 추가할 수도 있습니다.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    for q in tokens:
        if q == '':
            tokens.remove(q)
    if tokens[0] == '':
        tokens.remove(tokens[0])
    if tokens[-1] == '':
        tokens.remove(tokens[-1])
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def build_vocab(sequences, min_token_count=1):#고유 단어 수를 계산하고 어휘 사전을 토큰화합니다.
    token_to_count = {}
    for it in sequences:
        for seq in it[1]:
            for token in seq:
                if token not in token_to_count:
                    token_to_count[token] = 0
                token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if token in token_to_idx.keys():
            continue
        if count > min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
