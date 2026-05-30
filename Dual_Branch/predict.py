import sys

# Python 인터프리터 검색 경로에 특정 경로 추가
sys.path.append('/data/coding/Change_Agent/Multi_change')
import os.path

import cv2
import torch.optim
import argparse
import json

from skimage import measure

from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from imageio.v2 import imread


# compute_change_map(path_A, path_B) 함수: 두 이미지 사이의 변화 영역을 나타내는 마스크를 생성합니다.
'''
인자:
    path_A: 图像A的路径
    path_B: 图像B的路径
Returns:
    change_map: 变化区域的掩膜
'''
# def compute_change_mask(path_A, path_B):
#     import cv2
#     import numpy as np
#     img_A = cv2.imread(path_A)
#     img_B = cv2.imread(path_B)
#     change_map = (img_B-img_A).astype(np.uint8)
#     # 임계값 처리
#     change_map = cv2.cvtColor(change_map, cv2.COLOR_BGR2GRAY)
#     change_map = cv2.threshold(change_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     cv2.imwrite('E:\change_map.png', change_map)
#     return 'I have save the changed mask in E:\change_map.png'

# compute_change_caption(path_A, path_B) 함수: 두 이미지 사이의 변화를 설명하는 텍스트를 생성합니다.
'''
인자:
    path_A: 图像A的路径
    path_B: 图像B的路径
Returns:
    caption: 变化描述文本
'''
class Change_Perception(object):
    def define_args(self):


        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        #print(script_dir)
        parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')

        parser.add_argument('--data_folder', default='/data/coding/LEVIR-MCI-dataset/images/',
                            help='folder with data files')
        parser.add_argument('--list_path', default='/data/coding/Change_Agent/Multi_change/data/LEVIR_MCI/',
                            help='path of the data lists')
        parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
        parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')

        # 추론 설정
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
        parser.add_argument('--checkpoint', default='/data/coding/Change_Agent/Multi_change/models_ckpt/MCI_model.pth',help='path to checkpoint')
        parser.add_argument('--result_path', default="/data/coding/dataset_extra",
                            help='path to save the result of masks and captions')

        # 백본 파라미터
        parser.add_argument('--network', default='segformer-mit_b1',
                            help='define the backbone encoder to extract features')
        parser.add_argument('--encoder_dim', type=int, default=512,
                            help='the dimension of extracted features using backbone ')
        parser.add_argument('--feat_size', type=int, default=16,
                            help='define the output size of encoder to extract features')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
        # 모델 파라미터
        parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
        parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in Attention인코더입니다.')
        parser.add_argument('--decoder_n_layers', type=int, default=1)
        parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')

        #args = parser.parse_args()
        args = parser.parse_args(args=[])
        #args=parser.parse_known_args()

        return args

    def __init__(self,):
        """
        학습과 검증을 수행합니다.
        """
        args = self.define_args()
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)
        # 체크포인트 로드
        snapshot_full_path = args.checkpoint

        checkpoint = torch.load(snapshot_full_path)
        self.encoder = Encoder(args.network)
        self.encoder_trans = AttentiveEncoder(train_stage=None, n_layers=args.n_layers,
                                         feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
                                         heads=args.n_heads, dropout=args.dropout)
        self.decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
                                     vocab_size=len(self.word_vocab), max_lengths=args.max_length,
                                     word_vocab=self.word_vocab, n_head=args.n_heads,
                                     n_layers=args.decoder_n_layers, dropout=args.dropout)

        self.encoder.load_state_dict(checkpoint['encoder_dict'])
        self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
        self.decoder.load_state_dict(checkpoint['decoder_dict'])
        # 사용 가능하면 GPU로 이동
        self.encoder.eval()
        self.encoder = self.encoder.cuda()
        self.encoder_trans.eval()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder.eval()
        self.decoder = self.decoder.cuda()


    def preprocess(self, path_A, path_B):

        imgA = imread(path_A)
        imgB = imread(path_B)
        imgA = np.asarray(imgA, np.float32)
        imgB = np.asarray(imgB, np.float32)

        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)
        for i in range(len(self.mean)):
            imgA[i, :, :] -= self.mean[i]
            imgA[i, :, :] /= self.std[i]
            imgB[i, :, :] -= self.mean[i]
            imgB[i, :, :] /= self.std[i]

        if imgA.shape[1] != 256 or imgA.shape[2] != 256:
            imgA = cv2.resize(imgA, (256, 256))
            imgB = cv2.resize(imgB, (256, 256))

        imgA = torch.FloatTensor(imgA)
        imgB = torch.FloatTensor(imgB)
        imgA = imgA.unsqueeze(0)  # (1, 3, 256, 256)
        imgB = imgB.unsqueeze(0)

        return imgA, imgB

    def generate_change_caption(self, path_A, path_B):
        #print('변화 캡션 추론 시작')
        imgA, imgB = self.preprocess(path_A, path_B)
        # 사용 가능하면 GPU로 이동
        imgA = imgA.cuda()
        imgB = imgB.cuda()
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        seq = self.decoder.sample(feat1, feat2, k=1)
        pred_seq = [w for w in seq if w not in {self.word_vocab['<START>'], self.word_vocab['<END>'], self.word_vocab['<NULL>']}]
        pred_caption = ""
        for i in pred_seq:
            pred_caption += (list(self.word_vocab.keys())[i]) + " "

        caption ='there is road change'
        caption = pred_caption
        #print('변화 캡션 결과:', caption)
        return caption

    def change_detection(self, path_A, path_B, savepath_mask):
        #print('변화 탐지 추론 시작')
        imgA, imgB = self.preprocess(path_A, path_B)
        # 사용 가능하면 GPU로 이동
        imgA = imgA.cuda()
        imgB = imgB.cuda()
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        # 세그멘테이션 처리
        pred_seg = seg_pre.data.cpu().numpy()
        pred_seg = np.argmax(pred_seg, axis=1)
        # 이미지 저장
        pred = pred_seg[0].astype(np.uint8)
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[pred == 1] = [0, 255, 255]
        pred_rgb[pred == 2] = [0, 0, 255]

        cv2.imwrite(savepath_mask, pred_rgb)
        #print('마스크 저장 위치:', savepath_mask)

        #print('변화 탐지 추론 종료')
        return pred # (256,256,3)
        # return '변화 탐지 성공.'

    def compute_object_num(self, changed_mask, object):
        print("compute num start")
        # 연결 성분 개수 계산
        mask = changed_mask
        mask_cp = 0 * mask.copy()
        if object == 'road':
            mask_cp[mask == 1] = 255
        elif object == 'building':
            mask_cp[mask == 2] = 255
        lbl = measure.label(mask_cp, connectivity=2)
        props = measure.regionprops(lbl)
        # 반복문으로 bounding box 추출
        bboxes = []
        for prop in props:
            # print('Found bbox', prop.bbox, 'area:', prop.area)
            if prop.area > 5:
                bboxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
        num = len(bboxes)
        # 시각화
        # mask_array_copy = mask.copy()*255
        # for bbox in bboxes:
        #     print('Found bbox', bbox)
        #     cv2.rectangle(mask_array_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), 2)
        # cv2.namedWindow('findCorners', 0)
        # cv2.resizeWindow('findCorners', 600, 600)
        # cv2.imshow('findCorners', mask_array_copy)
        # cv2.waitKey(0)
        print('Found', num, object)
        print('compute num end')
        # return
        num_str = 'Found ' + str(num) + ' changed ' + object
        return num_str

    # 추가 도구 함수 설계:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')
    #parser.add_argument('--imgA_path')
    #parser.add_argument('--imgB_path')
    #parser.add_argument('--mask_save_path')

    #args = parser.parse_args()

    imgA_path = '/data/coding/LEVIR-MCI-dataset/images/train/A/train_000010.png'
    imgB_path = '/data/coding/LEVIR-MCI-dataset/images/train/B/train_000010.png'
    mask_save_path='./CDmask.png'

    #print(12123)

    Change_Perception = Change_Perception()
    Change_Perception.generate_change_caption(imgA_path, imgB_path)
    Change_Perception.change_detection(imgA_path, imgB_path, mask_save_path)
