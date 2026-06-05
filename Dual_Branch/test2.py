import cv2
import os
import torch.optim
from torch.utils import data
import argparse
import json
import numpy as np
import time
from tqdm import tqdm
from data.LEVIR_MCI import LEVIRCCDataset
from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from utils_tool.metrics import Evaluator
from train_2 import LEVIRCCRGBFFTDataset, RGBFFTFusionEncoder
from checkpoint_loader import checkpoint_result_name, load_checkpoint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

GRID_LABELS = {
    (0, 0): "top left corner",
    (0, 1): "top",
    (0, 2): "top right corner",
    (1, 0): "left",
    (1, 1): "center",
    (1, 2): "right",
    (2, 0): "lower left corner",
    (2, 1): "lower",
    (2, 2): "lower right corner",
}

def count_objects(pred_mask, class_id, min_area=5):
    binary = (pred_mask == class_id).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area)

def grid_locations(pred_mask, class_id, min_pixels=5):
    binary = pred_mask == class_id
    if not np.any(binary):
        return ["No change"]

    height, width = binary.shape
    locations = []
    for row in range(3):
        y0 = round(row * height / 3)
        y1 = round((row + 1) * height / 3)
        for col in range(3):
            x0 = round(col * width / 3)
            x1 = round((col + 1) * width / 3)
            if int(binary[y0:y1, x0:x1].sum()) > min_pixels:
                locations.append(GRID_LABELS[(row, col)])
    return locations or ["No change"]

def build_dual_prior_record(name, pred_caption, pred_mask):
    return {
        "image": name,
        "dual_prior": {
            "global_caption": pred_caption.strip(),
            "road_count": count_objects(pred_mask, 1),
            "building_count": count_objects(pred_mask, 2),
            "road_locations": grid_locations(pred_mask, 1),
            "building_locations": grid_locations(pred_mask, 2),
        },
        "error": None,
    }

def save_dual_prior_records(records, save_path):
    jsonl_path = os.path.join(save_path, "dual_prior.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Saved dual_prior JSONL:", jsonl_path)

def save_mask(pred, gt, name, save_path,args):
    # 예측값 0,1,2를 검정, 노랑, 빨강으로 매핑합니다.
    # 정답값 0,1,2를 검정, 노랑, 빨강으로 매핑합니다.
    name = name[0]
    evaluator = Evaluator(num_class=3)
    evaluator.add_batch(gt, pred)
    mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
    Miou_str = round(mIoU_seg, 4)
    # MIoU 문자열을 score라는 JSON 파일에 저장합니다.
    json_name = os.path.join(save_path, 'score.json')
    if not os.path.exists(json_name):
        with open(json_name, 'a+') as f:
            key = name.split('.')[0]
            json.dump({f'{key}': {'MIoU':Miou_str}}, f)
        f.close()
    else:
        with open(os.path.join(save_path, 'score.json'), 'r') as file:
            data = json.load(file)
            key = name.split('.')[0]
            data[key] = {'MIoU': Miou_str}
        # JSON 파일에 기록
        with open(os.path.join(save_path, 'score.json'), 'w') as file:
            json.dump(data, file)
        file.close()

    # 마스크 저장
    pred = pred[0].astype(np.uint8)
    gt = gt[0].astype(np.uint8)
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    pred_rgb[pred == 1] = [0, 255, 255]
    pred_rgb[pred == 2] = [0, 0, 255]
    gt_rgb[gt == 1] = [0, 255, 255]
    gt_rgb[gt == 2] = [0, 0, 255]

    cv2.imwrite(os.path.join(save_path, name.split('.')[0] + f'_mask.png'), pred_rgb)
    cv2.imwrite(os.path.join(save_path, name.split('.')[0] + '_gt.png'), gt_rgb)
    # image_A와 image_B 저장
    img_A_path = os.path.join(args.data_folder, 'test/A', name)
    img_B_path = os.path.join(args.data_folder, 'test/B', name)
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    cv2.imwrite(os.path.join(save_path, name.split('.')[0] + '_A.png'), img_A)
    cv2.imwrite(os.path.join(save_path, name.split('.')[0] + '_B.png'), img_B)

def save_captions(pred_caption, ref_caption, hypotheses, references, name, save_path):
    name = name[0]
    # 0 반환
    score_dict = get_eval_score([references], [hypotheses])
    Bleu_4 = score_dict['Bleu_4']
    Bleu_4_str = round(Bleu_4, 4)
    Bleu_3 = score_dict['Bleu_3']
    Bleu_3_str = round(Bleu_3, 4)

    # JSON 읽기
    with open(os.path.join(save_path, 'score.json'), 'r') as file:
        data = json.load(file)
        key = name.split('.')[0]
        data[key]['Bleu_3'] = Bleu_3_str
        data[key]['Bleu_4'] = Bleu_4_str
    with open(os.path.join(save_path, 'score.json'), 'w') as file:
        json.dump(data, file)
    file.close()

    with open(os.path.join(save_path, name.split('.')[0] + f'_cap.txt'), 'w') as f:
        f.write('pred_caption: ' + pred_caption + '\n')
        f.write('ref_caption: ' + ref_caption + '\n')

def main(args):
    """
    테스트 설정ing.
    """

    with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    # 체크포인트 로드
    snapshot_full_path = args.checkpoint
    checkpoint = load_checkpoint(snapshot_full_path)

    args.result_path = os.path.join(args.result_path, checkpoint_result_name(snapshot_full_path))
    if os.path.exists(args.result_path) == False:
        os.makedirs(args.result_path)
    else:
        print('result_path is existed!')
        # 폴더 비우기
        for root, dirs, files in os.walk(args.result_path):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))


    encoder = RGBFFTFusionEncoder(args.network)
    encoder_trans = AttentiveEncoder(train_stage=None, n_layers=args.n_layers,
                                          feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                          heads=args.n_heads, dropout=args.dropout)
    decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                                      vocab_size=len(word_vocab), max_lengths=args.max_length,
                                      word_vocab=word_vocab, n_head=args.n_heads,
                                      n_layers=args.decoder_n_layers, dropout=args.dropout)

    encoder.load_state_dict(checkpoint['encoder_dict'])
    encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
    decoder.load_state_dict(checkpoint['decoder_dict'])
    # 사용 가능하면 GPU로 이동
    encoder.eval()
    encoder = encoder.cuda()
    encoder_trans.eval()
    encoder_trans = encoder_trans.cuda()
    decoder.eval()
    decoder = decoder.cuda()

    # 사용자 정의 데이터 로더
    if args.data_name == 'LEVIR_MCI':
        nochange_list = ["the scene is the same as before ", "there is no difference ",
                         "the two scenes seem identical ", "no change has occurred ",
                         "almost nothing has changed "]
        test_loader = data.DataLoader(
            LEVIRCCRGBFFTDataset(args.data_folder, args.fft_data_folder, args.list_path, 'test',
                                 args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 에폭 설정
    test_start_time = time.time()
    references = list()  # BLEU-4 계산용 정답 캡션
    hypotheses = list()  # 모델 예측 캡션
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc = 0
    nochange_acc = 0
    dual_prior_records = []
    evaluator = Evaluator(num_class=3)
    with torch.no_grad():
        for ind, (imgA, imgB, imgA_fft, imgB_fft, seg_label, token_all, token_all_len, _, _, name) in enumerate(
                tqdm(test_loader, desc='test_' + " EVALUATING AT BEAM SIZE " + str(1))):
            # 사용 가능하면 GPU로 이동
            imgA = imgA.cuda()
            imgB = imgB.cuda()
            imgA_fft = imgA_fft.cuda()
            imgB_fft = imgB_fft.cuda()
            token_all = token_all.squeeze(0).cuda()
            # decode_lengths = max(token_all_len.squeeze(0)).item()
            # 순전파 수행
            if encoder is not None:
                feat1, feat2 = encoder(imgA, imgB, imgA_fft, imgB_fft)
            feat1, feat2, seg_pre = encoder_trans(feat1, feat2)
            seq = decoder.sample(feat1, feat2, k=1)

            # 세그멘테이션 처리
            pred_seg = seg_pre.data.cpu().numpy()
            seg_label = seg_label.cpu().numpy()
            pred_seg = np.argmax(pred_seg, axis=1)

            # 변화 탐지 마스크를 저장할지 여부
            if args.save_mask:
                save_mask(pred_seg, seg_label, name, args.result_path, args)
            # 현재 배치를 평가기에 추가
            evaluator.add_batch(seg_label, pred_seg)

            # 캡션 평가
            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in {word_vocab['<START>'], word_vocab['<END>'],
                                                                       word_vocab['<NULL>']}],
                                  img_token))  # <start>와 padding 토큰 제거
            references.append(img_tokens)

            pred_seq = [w for w in seq if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)
            pred_caption = ""
            ref_caption = ""
            for i in pred_seq:
                pred_caption += (list(word_vocab.keys())[i]) + " "
            ref_caption = ""
            for i in img_tokens[0]:
                ref_caption += (list(word_vocab.keys())[i]) + " "
            ref_captions = ""
            for i in img_tokens:
                for j in i:
                    ref_captions += (list(word_vocab.keys())[j]) + " "
                ref_captions += ".    "

            pred_mask = pred_seg[0].astype(np.uint8)
            dual_prior_records.append(build_dual_prior_record(name[0], pred_caption, pred_mask))

            # 캡션 결과를 저장할지 여부
            if args.save_caption:
                save_captions(pred_caption, ref_captions, hypotheses[-1], references[-1], name, args.result_path)
            if ref_caption in nochange_list:
                nochange_references.append(img_tokens)
                nochange_hypotheses.append(pred_seq)
                if pred_caption in nochange_list:
                    nochange_acc = nochange_acc + 1
            else:
                change_references.append(img_tokens)
                change_hypotheses.append(pred_seq)
                if pred_caption not in nochange_list:
                    change_acc = change_acc + 1

        test_time = time.time() - test_start_time
        save_dual_prior_records(dual_prior_records, args.result_path)

        # 학습 중 빠른 검증

        Acc_seg = evaluator.Pixel_Accuracy()
        Acc_class_seg = evaluator.Pixel_Accuracy_Class()
        mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
        FWIoU_seg = evaluator.Frequency_Weighted_Intersection_over_Union()
        print(
            '검증 설정:\n' 'Acc_seg: {0:.5f}\t' 'Acc_class_seg: {1:.5f}\t' 'mIoU_seg: {2:.5f}\t' 'FWIoU_seg: {3:.5f}\t'
            .format(Acc_seg, Acc_class_seg, mIoU_seg, FWIoU_seg))
        print('IoU:', IoU)

        # 평가 점수 계산
        print('len(nochange_references):', len(nochange_references))
        print('len(change_references):', len(change_references))

        if len(nochange_references) > 0:
            print('nochange_metric:')
            nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
            Bleu_1 = nochange_metric['Bleu_1']
            Bleu_2 = nochange_metric['Bleu_2']
            Bleu_3 = nochange_metric['Bleu_3']
            Bleu_4 = nochange_metric['Bleu_4']
            Meteor = nochange_metric['METEOR']
            Rouge = nochange_metric['ROUGE_L']
            Cider = nochange_metric['CIDEr']
            print('BLEU-1: {0:.5f}\t' 'BLEU-2: {1:.5f}\t' 'BLEU-3: {2:.5f}\t'
                  'BLEU-4: {3:.5f}\t' 'Meteor: {4:.5f}\t' 'Rouge: {5:.5f}\t' 'Cider: {6:.5f}\t'
                  .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))
            print("nochange_acc:", nochange_acc / len(nochange_references))
        if len(change_references) > 0:
            print('change_metric:')
            change_metric = get_eval_score(change_references, change_hypotheses)
            Bleu_1 = change_metric['Bleu_1']
            Bleu_2 = change_metric['Bleu_2']
            Bleu_3 = change_metric['Bleu_3']
            Bleu_4 = change_metric['Bleu_4']
            Meteor = change_metric['METEOR']
            Rouge = change_metric['ROUGE_L']
            Cider = change_metric['CIDEr']
            print('BLEU-1: {0:.5f}\t' 'BLEU-2: {1:.5f}\t' 'BLEU-3: {2:.5f}\t'
                  'BLEU-4: {3:.5f}\t' 'Meteor: {4:.5f}\t' 'Rouge: {5:.5f}\t' 'Cider: {6:.5f}\t'
                  .format(Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))
            print("change_acc:", change_acc / len(change_references))

        score_dict = get_eval_score(references, hypotheses)
        Bleu_1 = score_dict['Bleu_1']
        Bleu_2 = score_dict['Bleu_2']
        Bleu_3 = score_dict['Bleu_3']
        Bleu_4 = score_dict['Bleu_4']
        Meteor = score_dict['METEOR']
        Rouge = score_dict['ROUGE_L']
        Cider = score_dict['CIDEr']
        print('테스트 설정 of Captioning:\n' 'Time: {0:.3f}\t' 'BLEU-1: {1:.5f}\t' 'BLEU-2: {2:.5f}\t' 'BLEU-3: {3:.5f}\t'
              'BLEU-4: {4:.5f}\t' 'Meteor: {5:.5f}\t' 'Rouge: {6:.5f}\t' 'Cider: {7:.5f}\t'
              .format(test_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation_FFT')

    # 데이터 파라미터
    parser.add_argument('--sys', default='win', help='system win or linux')
    parser.add_argument('--data_folder',
        default=os.path.join(REPO_ROOT, 'data', 'coding', 'datasets', 'LEVIR-MCI-dataset', 'images'),
        help='folder with RGB image files')
    parser.add_argument('--fft_data_folder',
        default=os.path.join(REPO_ROOT, 'data', 'coding', 'datasets', 'LEVIR-MCI-dataset-fft', 'images'),
        help='folder with FFT image files')
    parser.add_argument('--list_path', default=os.path.join(SCRIPT_DIR, 'data', 'LEVIR_MCI') + os.sep, help='path of the data lists')
    parser.add_argument('--token_folder', default=os.path.join(SCRIPT_DIR, 'data', 'LEVIR_MCI', 'tokens') + os.sep, help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
    parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default="LEVIR_MCI", help='base name shared by data files.')

    # 테스트 설정
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument(
        '--checkpoint',
        default=os.path.join(
            SCRIPT_DIR,
            'weights',
            'Dual_Branch_FFT',
            'Dual_Branch_FFT.safetensors',
        ),
        help='path to .pth or .safetensors FFT checkpoint',
    )
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')
    parser.add_argument('--test_batchsize', type=int, default=1, help='batch_size for test')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    # 마스크와 캡션을 저장할지 여부
    parser.add_argument('--save_mask', action='store_true', help='save the result of masks')
    parser.add_argument('--save_caption', action='store_true', help='save the result of captions')
    parser.add_argument('--result_path', default="./predict_result/", help='path to save the result of masks and captions')
    # 백본 파라미터
    parser.add_argument('--network', default='segformer-mit_b1', help='define the backbone encoder to extract features')
    parser.add_argument('--encoder_dim', type=int, default=512,
                        help='the dimension of extracted features using backbone ')
    parser.add_argument('--feat_size', type=int, default=16,
                        help='define the output size of encoder to extract features')
    # 모델 파라미터
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in Attention인코더입니다.')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')

    args = parser.parse_args()

    main(args)
