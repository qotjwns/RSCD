import argparse
import json
import math

import torch.optim
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
from tqdm import tqdm

from data.LEVIR_MCI import LEVIRCCDataset
from model.model_decoder import DecoderTransformer
from model.model_encoder_att import AttentiveEncoder, Encoder
from utils_tool.metrics import Evaluator
from utils_tool.utils import *


class LEVIRCCRGBFFTDataset(data.Dataset):
    """
    RGB 원본 이미지와 FFT 이미지가 같은 split/list 순서로 있다는 전제의 paired dataset입니다.
    기존 LEVIRCCDataset의 label/token 처리는 RGB 쪽 것을 그대로 사용합니다.
    """

    def __init__(
        self,
        rgb_data_folder,
        fft_data_folder,
        list_path,
        split,
        token_folder=None,
        vocab_file=None,
        max_length=41,
        allow_unk=0,
    ):
        self.rgb_dataset = LEVIRCCDataset(
            rgb_data_folder, list_path, split, token_folder, vocab_file, max_length, allow_unk
        )
        self.fft_dataset = LEVIRCCDataset(
            fft_data_folder, list_path, split, token_folder, vocab_file, max_length, allow_unk
        )
        if len(self.rgb_dataset) != len(self.fft_dataset):
            raise ValueError(
                f"RGB/FFT dataset length mismatch: {len(self.rgb_dataset)} != {len(self.fft_dataset)}"
            )

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, index):
        rgb = self.rgb_dataset[index]
        fft = self.fft_dataset[index]

        img_a, img_b, seg_label, token_all, token_all_len, token, token_len, name = rgb
        img_a_fft, img_b_fft = fft[0], fft[1]

        return (
            img_a,
            img_b,
            img_a_fft,
            img_b_fft,
            seg_label,
            token_all,
            token_all_len,
            token,
            token_len,
            name,
        )


class ScalarGatedFeatureFusion(nn.Module):
    """
    각 feature level마다 trainable scalar gate를 둡니다.
    gate가 1에 가까우면 RGB를, 0에 가까우면 FFT를 더 신뢰합니다.
    """

    def __init__(self, max_features=8, init_rgb_weight=0.8):
        super().__init__()
        init_rgb_weight = min(max(init_rgb_weight, 1e-4), 1 - 1e-4)
        init_logit = math.log(init_rgb_weight / (1.0 - init_rgb_weight))
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(init_logit))) for _ in range(max_features)]
        )

    def forward(self, rgb_features, fft_features):
        if len(rgb_features) != len(fft_features):
            raise ValueError(
                f"RGB/FFT feature length mismatch: {len(rgb_features)} != {len(fft_features)}"
            )
        if len(rgb_features) > len(self.gates):
            raise ValueError(
                f"Need at least {len(rgb_features)} fusion gates, but only {len(self.gates)} exist."
            )

        fused = []
        for idx, (rgb_feat, fft_feat) in enumerate(zip(rgb_features, fft_features)):
            gate = torch.sigmoid(self.gates[idx])
            fused.append(gate * rgb_feat + (1.0 - gate) * fft_feat)
        return fused


class RGBFFTFusionEncoder(nn.Module):
    """
    T1/T2 RGB와 T1/T2 FFT를 각각 같은 구조의 encoder에 통과시킨 뒤 feature list를 fusion합니다.
    """

    def __init__(self, network, fusion_init_rgb_weight=0.8):
        super().__init__()
        self.encoder_rgb = Encoder(network)
        self.encoder_fft = Encoder(network)
        self.fusion_a = ScalarGatedFeatureFusion(init_rgb_weight=fusion_init_rgb_weight)
        self.fusion_b = ScalarGatedFeatureFusion(init_rgb_weight=fusion_init_rgb_weight)

    def forward(self, img_a, img_b, img_a_fft, img_b_fft):
        rgb_a_features, rgb_b_features = self.encoder_rgb(img_a, img_b)
        fft_a_features, fft_b_features = self.encoder_fft(img_a_fft, img_b_fft)

        fused_a_features = self.fusion_a(rgb_a_features, fft_a_features)
        fused_b_features = self.fusion_b(rgb_b_features, fft_b_features)
        return fused_a_features, fused_b_features

    def fine_tune(self, fine_tune_backbone=True, fine_tune_fusion=True):
        self.encoder_rgb.fine_tune(fine_tune_backbone)
        self.encoder_fft.fine_tune(fine_tune_backbone)

        for fusion in [self.fusion_a, self.fusion_b]:
            for p in fusion.parameters():
                p.requires_grad = fine_tune_fusion

    def load_single_encoder_state(self, encoder_state, strict=False):
        self.encoder_rgb.load_state_dict(encoder_state, strict=strict)
        self.encoder_fft.load_state_dict(encoder_state, strict=strict)


class Trainer(object):
    def __init__(self, args):
        self.start_train_goal = args.train_goal
        self.args = args
        random_str = str(random.randint(10, 100))
        name = 'rgb_fft_' + time_file_str() + f'_{args.data_variant}_train_goal_{args.train_goal}_' + random_str
        self.args.savepath = os.path.join(args.savepath, name)
        if os.path.exists(self.args.savepath) == False:
            os.makedirs(self.args.savepath)
        self.log = open(os.path.join(self.args.savepath, '{}.log'.format(name)), 'w')
        print_log('=>datset: {}'.format(args.data_name), self.log)
        print_log('=>network: {}'.format(args.network), self.log)
        print_log('=>encoder_lr: {}'.format(args.encoder_lr), self.log)
        print_log('=>decoder_lr: {}'.format(args.decoder_lr), self.log)
        print_log('=>num_epochs: {}'.format(args.num_epochs), self.log)
        print_log('=>train_batchsize: {}'.format(args.train_batchsize), self.log)
        print_log('=>data_folder: {}'.format(args.data_folder), self.log)
        print_log('=>fft_data_folder: {}'.format(args.fft_data_folder), self.log)
        print_log('=>data_variant: {}'.format(args.data_variant), self.log)
        print_log('=>checkpoint: {}'.format(args.checkpoint), self.log)

        self.best_bleu4 = 0.4
        self.MIou = 0.4
        self.Sum_Metric = 0.4
        self.start_epoch = 0
        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)

        self.build_model()

        self.criterion_cap = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_det = torch.nn.CrossEntropyLoss().cuda()

        if args.data_name == 'LEVIR_MCI':
            self.train_loader = data.DataLoader(
                LEVIRCCRGBFFTDataset(
                    args.data_folder,
                    args.fft_data_folder,
                    args.list_path,
                    'train',
                    args.token_folder,
                    args.vocab_file,
                    args.max_length,
                    args.allow_unk,
                ),
                batch_size=args.train_batchsize,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
            )
            self.val_loader = data.DataLoader(
                LEVIRCCRGBFFTDataset(
                    args.data_folder,
                    args.fft_data_folder,
                    args.list_path,
                    'val',
                    args.token_folder,
                    args.vocab_file,
                    args.max_length,
                    args.allow_unk,
                ),
                batch_size=args.val_batchsize,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
            )

        self.index_i = 0
        self.hist = np.zeros((args.num_epochs * 2 * len(self.train_loader), 7))
        self.evaluator = Evaluator(num_class=3)
        self.best_model_path = None
        self.best_epoch = 0

    def build_model(self):
        args = self.args
        self.encoder = RGBFFTFusionEncoder(args.network, args.fusion_init_rgb_weight)
        self.encoder_trans = AttentiveEncoder(
            train_stage=args.train_stage,
            n_layers=args.n_layers,
            feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
            heads=args.n_heads,
            dropout=args.dropout,
        )
        self.decoder = DecoderTransformer(
            encoder_dim=args.encoder_dim,
            feature_dim=args.feature_dim,
            vocab_size=len(self.word_vocab),
            max_lengths=args.max_length,
            word_vocab=self.word_vocab,
            n_head=args.n_heads,
            n_layers=args.decoder_n_layers,
            dropout=args.dropout,
        )

        if args.checkpoint is not None:
            self.load_checkpoint(args.checkpoint)

        if args.train_stage == 's2':
            if args.checkpoint is None:
                raise ValueError('Error: checkpoint is None.')
            args.fine_tune_encoder = False
            self.encoder.fine_tune(fine_tune_backbone=False, fine_tune_fusion=True)
            self.encoder_trans.fine_tune(args.train_goal)
            fine_tune_capdecoder = False if args.train_goal == 0 else True
            self.decoder.fine_tune(fine_tune_capdecoder)
        elif args.train_stage == 's1':
            self.encoder.fine_tune(
                fine_tune_backbone=args.fine_tune_encoder,
                fine_tune_fusion=True,
            )
            fine_tune_capdecoder = True
        else:
            raise ValueError(f'Unsupported train_stage: {args.train_stage}')

        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        self.encoder_optimizer = (
            torch.optim.Adam(params=encoder_params, lr=args.encoder_lr)
            if len(encoder_params) > 0
            else None
        )
        self.encoder_trans_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
            lr=args.encoder_lr,
        )
        self.decoder_optimizer = (
            torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
                lr=args.decoder_lr,
            )
            if fine_tune_capdecoder
            else None
        )

        self.encoder = self.encoder.cuda()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder = self.decoder.cuda()
        self.encoder_lr_scheduler = (
            torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, step_size=5, gamma=1.0)
            if self.encoder_optimizer is not None
            else None
        )
        self.encoder_trans_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.encoder_trans_optimizer, step_size=5, gamma=1.0
        )
        self.decoder_lr_scheduler = (
            torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, step_size=5, gamma=1.0)
            if fine_tune_capdecoder
            else None
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print('Load Model from {}'.format(checkpoint_path))

        encoder_state = checkpoint.get('encoder_dict', None) if isinstance(checkpoint, dict) else None
        if encoder_state is None and isinstance(checkpoint, dict):
            encoder_state = checkpoint

        if encoder_state is not None:
            if any(k.startswith('encoder_rgb.') or k.startswith('encoder_fft.') for k in encoder_state.keys()):
                self.encoder.load_state_dict(encoder_state, strict=False)
            else:
                self.encoder.load_single_encoder_state(encoder_state, strict=False)

        if isinstance(checkpoint, dict) and 'encoder_trans_dict' in checkpoint:
            self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
        if isinstance(checkpoint, dict) and 'decoder_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_dict'], strict=False)

    def _encoder_train_mode(self):
        if self.start_train_goal != 2:
            self.encoder.train()
        else:
            if self.args.train_goal == 2:
                self.encoder.train()
            else:
                self.encoder.eval()
                self.encoder.fusion_a.train()
                self.encoder.fusion_b.train()

    def training(self, args, epoch):
        if self.start_train_goal != 2:
            self.encoder.train()
            self.encoder_trans.train()
            self.decoder.train()
        else:
            if self.args.train_goal == 2:
                self.encoder.train()
                self.encoder_trans.train()
                self.decoder.train()
            elif self.args.train_goal == 1:
                self._encoder_train_mode()
                self.encoder_trans.fine_tune(self.args.train_goal)
                self.decoder.train()
            else:
                self._encoder_train_mode()
                self.encoder_trans.fine_tune(self.args.train_goal)
                self.decoder.eval()

        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad()
        self.encoder_trans_optimizer.zero_grad()
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()

        train_bar = tqdm(
            self.train_loader,
            desc=f'train_epoch_{epoch}_goal_{self.args.train_goal}',
            total=len(self.train_loader),
            dynamic_ncols=True,
        )
        for id, (
            imgA,
            imgB,
            imgA_fft,
            imgB_fft,
            seg_label,
            _,
            _,
            token,
            token_len,
            _,
        ) in enumerate(train_bar):
            start_time = time.time()
            accum_steps = max(1, 64 // args.train_batchsize)

            imgA = imgA.cuda()
            imgB = imgB.cuda()
            imgA_fft = imgA_fft.cuda()
            imgB_fft = imgB_fft.cuda()
            seg_label = seg_label.cuda()
            token = token.squeeze(1).cuda()
            token_len = token_len.cuda()

            feat1, feat2 = self.encoder(imgA, imgB, imgA_fft, imgB_fft)
            feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
            if self.args.train_goal != 0:
                scores, caps_sorted, decode_lengths, sort_ind = self.decoder(
                    feat1, feat2, token, token_len
                )
                targets = caps_sorted[:, 1:]
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                cap_loss = self.criterion_cap(scores, targets.to(torch.int64))
                raw_cap_loss = cap_loss.detach()

            det_loss = self.criterion_det(seg_pre, seg_label.to(torch.int64))
            raw_det_loss = det_loss.detach()
            if self.args.train_goal == 0:
                if self.start_train_goal == 2:
                    if epoch < 100:
                        det_loss = det_loss
                loss = det_loss
            elif self.args.train_goal == 1:
                loss = cap_loss
            else:
                if args.train_stage == 's1':
                    det_loss = det_loss / det_loss.detach().item()
                    cap_loss = cap_loss / cap_loss.detach().item()
                loss = det_loss + cap_loss

            loss = loss / accum_steps
            loss.backward()

            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_value_(self.decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_value_(self.encoder_trans.parameters(), args.grad_clip)
                if self.encoder_optimizer is not None:
                    torch.nn.utils.clip_grad_value_(self.encoder.parameters(), args.grad_clip)

            if (id + 1) % accum_steps == 0 or (id + 1) == len(self.train_loader):
                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.step()
                self.encoder_trans_optimizer.step()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.step()

                if self.decoder_lr_scheduler is not None:
                    self.decoder_lr_scheduler.step()
                self.encoder_trans_lr_scheduler.step()
                if self.encoder_lr_scheduler is not None:
                    self.encoder_lr_scheduler.step()

                if self.decoder_optimizer is not None:
                    self.decoder_optimizer.zero_grad()
                self.encoder_trans_optimizer.zero_grad()
                if self.encoder_optimizer is not None:
                    self.encoder_optimizer.zero_grad()

            self.hist[self.index_i, 0] = time.time() - start_time
            if self.args.train_goal == 0 or self.args.train_goal == 2:
                self.hist[self.index_i, 1] = raw_det_loss.item()
                self.hist[self.index_i, 2] = det_loss.item()
                self.hist[self.index_i, 3] = accuracy(
                    seg_pre.permute(0, 2, 3, 1).reshape(-1, seg_pre.size(1)),
                    seg_label.reshape(-1),
                    1,
                )
            if self.args.train_goal == 1 or self.args.train_goal == 2:
                self.hist[self.index_i, 4] = raw_cap_loss.item()
                self.hist[self.index_i, 5] = cap_loss.item()
                self.hist[self.index_i, 6] = accuracy(scores, targets, 5)

            self.index_i += 1
            if self.index_i % args.print_freq == 0:
                hist_window = self.hist[self.index_i - args.print_freq:self.index_i]
                print_log(
                    'Training Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time: {3:.3f}\t'
                    'Raw Det_Loss: {4:.4f}\t'
                    'Norm Det_Loss: {5:.4f}\t'
                    'Det Acc: {6:.3f}\t'
                    'Raw Cap_loss: {7:.5f}\t'
                    'Norm Cap_loss: {8:.5f}\t'
                    'Text_Top-5 Acc: {9:.3f}'.format(
                        epoch,
                        id,
                        len(self.train_loader),
                        np.mean(hist_window[:, 0]) * args.print_freq,
                        np.mean(hist_window[:, 1]),
                        np.mean(hist_window[:, 2]),
                        np.mean(hist_window[:, 3]),
                        np.mean(hist_window[:, 4]),
                        np.mean(hist_window[:, 5]),
                        np.mean(hist_window[:, 6]),
                    ),
                    self.log,
                )

    def validation(self, epoch):
        word_vocab = self.word_vocab
        self.decoder.eval()
        self.encoder_trans.eval()
        self.encoder.eval()

        val_start_time = time.time()
        references = list()
        hypotheses = list()

        self.evaluator.reset()
        with torch.no_grad():
            for ind, (
                imgA,
                imgB,
                imgA_fft,
                imgB_fft,
                seg_label,
                token_all,
                token_all_len,
                _,
                _,
                _,
            ) in enumerate(tqdm(self.val_loader, desc='val_' + "EVALUATING AT BEAM SIZE " + str(1))):
                imgA = imgA.cuda()
                imgB = imgB.cuda()
                imgA_fft = imgA_fft.cuda()
                imgB_fft = imgB_fft.cuda()
                token_all = token_all.squeeze(0).cuda()

                feat1, feat2 = self.encoder(imgA, imgB, imgA_fft, imgB_fft)
                feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
                if self.args.train_goal != 0 or self.start_train_goal == 2:
                    seq = self.decoder.sample(feat1, feat2, k=1)

                if self.args.train_goal != 1 or self.start_train_goal == 2:
                    pred_seg = seg_pre.data.cpu().numpy()
                    seg_label = seg_label.cpu().numpy()
                    pred_seg = np.argmax(pred_seg, axis=1)
                    self.evaluator.add_batch(seg_label, pred_seg)

                if self.args.train_goal != 0 or self.start_train_goal == 2:
                    img_token = token_all.tolist()
                    img_tokens = list(
                        map(
                            lambda c: [
                                w
                                for w in c
                                if w
                                not in {
                                    word_vocab['<START>'],
                                    word_vocab['<END>'],
                                    word_vocab['<NULL>'],
                                }
                            ],
                            img_token,
                        )
                    )
                    references.append(img_tokens)

                    pred_seq = [
                        w
                        for w in seq
                        if w not in {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}
                    ]
                    hypotheses.append(pred_seq)
                    assert len(references) == len(hypotheses)

                    if ind % self.args.print_freq == 0:
                        pred_caption = ""
                        ref_caption = ""
                        for i in pred_seq:
                            pred_caption += (list(word_vocab.keys())[i]) + " "
                        ref_caption = ""
                        for i in img_tokens:
                            for j in i:
                                ref_caption += (list(word_vocab.keys())[j]) + " "
                            ref_caption += ".    "

            val_time = time.time() - val_start_time
            if self.args.train_goal != 1 or self.start_train_goal == 2:
                Acc_seg = self.evaluator.Pixel_Accuracy()
                Acc_class_seg = self.evaluator.Pixel_Accuracy_Class()
                mIoU_seg, IoU = self.evaluator.Mean_Intersection_over_Union()
                FWIoU_seg = self.evaluator.Frequency_Weighted_Intersection_over_Union()
                print_log(
                    '\nDetection_검증 설정:\n'
                    'Acc_seg: {0:.5f}\t'
                    'Acc_class_seg: {1:.5f}\t'
                    'mIoU_seg: {2:.5f}\t'
                    'FWIoU_seg: {3:.5f}\t '.format(
                        Acc_seg, Acc_class_seg, mIoU_seg, FWIoU_seg
                    ),
                    self.log,
                )
                print_log('Iou: {}'.format(IoU), self.log)

            if self.args.train_goal != 0 or self.start_train_goal == 2:
                score_dict = get_eval_score(references, hypotheses)
                Bleu_1 = score_dict['Bleu_1']
                Bleu_2 = score_dict['Bleu_2']
                Bleu_3 = score_dict['Bleu_3']
                Bleu_4 = score_dict['Bleu_4']
                Meteor = score_dict['METEOR']
                Rouge = score_dict['ROUGE_L']
                Cider = score_dict['CIDEr']
                print_log(
                    'Captioning_검증 설정:\n'
                    'Time: {0:.3f}\t'
                    'BLEU-1: {1:.5f}\t'
                    'BLEU-2: {2:.5f}\t'
                    'BLEU-3: {3:.5f}\t'
                    'BLEU-4: {4:.5f}\t'
                    'Meteor: {5:.5f}\t'
                    'Rouge: {6:.5f}\t'
                    'Cider: {7:.5f}\t'.format(
                        val_time, Bleu_1, Bleu_2, Bleu_3, Bleu_4, Meteor, Rouge, Cider
                    ),
                    self.log,
                )

        if self.start_train_goal != 2:
            if self.args.train_goal == 0:
                Bleu_4 = 0
            if self.args.train_goal == 1:
                mIoU_seg = 0
            if Bleu_4 > self.best_bleu4 or mIoU_seg > self.MIou or Bleu_4 + mIoU_seg > self.Sum_Metric:
                self.best_bleu4 = max(Bleu_4, self.best_bleu4)
                self.MIou = max(mIoU_seg, self.MIou)
                self.Sum_Metric = max(Bleu_4 + mIoU_seg, self.Sum_Metric)
                print('Save Model')
                state = self.make_checkpoint_state()
                checkpoint_name = f'epoch_{epoch}_Dual_Branch_RGB_FFT.pth'
                best_model_path = os.path.join(self.args.savepath, checkpoint_name)
                torch.save(state, best_model_path)
                self.best_model_path = best_model_path
        elif self.start_train_goal == 2:
            Sum_Metric = mIoU_seg + Bleu_4
            if (
                (self.args.train_goal == 2 and Sum_Metric >= self.Sum_Metric)
                or (self.args.train_goal == 1 and Bleu_4 >= self.best_bleu4)
                or (self.args.train_goal == 0 and mIoU_seg > self.MIou)
            ):
                self.best_bleu4 = max(Bleu_4, self.best_bleu4) if self.args.train_goal == 1 else Bleu_4
                self.MIou = max(mIoU_seg, self.MIou) if self.args.train_goal == 0 else mIoU_seg
                self.Sum_Metric = max(Sum_Metric, self.Sum_Metric) if self.args.train_goal == 2 else Sum_Metric
                print('Save Model')
                state = self.make_checkpoint_state()
                checkpoint_name = f'epoch_{epoch}_Dual_Branch_RGB_FFT.pth'
                best_model_path = os.path.join(self.args.savepath, checkpoint_name)
                torch.save(state, best_model_path)
                self.best_epoch = epoch
                self.best_model_path = best_model_path

    def make_checkpoint_state(self):
        return {
            'encoder_dict': self.encoder.state_dict(),
            'encoder_rgb_dict': self.encoder.encoder_rgb.state_dict(),
            'encoder_fft_dict': self.encoder.encoder_fft.state_dict(),
            'fusion_a_dict': self.encoder.fusion_a.state_dict(),
            'fusion_b_dict': self.encoder.fusion_b.state_dict(),
            'encoder_trans_dict': self.encoder_trans.state_dict(),
            'decoder_dict': self.decoder.state_dict(),
        }


def parse_args():
    parser = argparse.ArgumentParser(description='RGB+FFT fine-tuning for Dual_Branch')

    default_data_folder = './data/coding/datasets/LEVIR-MCI-dataset/images'
    fft_data_folder = './data/coding/datasets/LEVIR-MCI-dataset-fft/images'
    parser.add_argument('--sys', default='win', help='system win or linux')
    parser.add_argument('--data_folder', default=default_data_folder, help='folder with RGB image files')
    parser.add_argument('--fft_data_folder', default=fft_data_folder, help='folder with FFT image files')
    parser.add_argument('--list_path', default='./Dual_Branch/data/LEVIR_MCI/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./Dual_Branch/data/LEVIR_MCI/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='vocab json file name')
    parser.add_argument('--max_length', type=int, default=41, help='max caption length')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default='LEVIR_MCI', help='base name shared by data files.')

    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='RGB train.py checkpoint or train_2.py RGB+FFT checkpoint used for initialization/resume',
    )
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches')

    parser.add_argument('--train_goal', type=int, default=2, help='0:det; 1:cap; 2:two tasks')
    parser.add_argument(
        '--train_stage',
        default='s1',
        help='s1: train/fine-tune RGB+FFT; s2: freeze backbone and tune task branch',
    )
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--train_batchsize', type=int, default=64, help='batch_size for training')
    parser.add_argument('--num_epochs', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder/fusion')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder')
    parser.add_argument('--grad_clip', type=float, default=None, help='clip gradients at an absolute value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument(
        '--fusion_init_rgb_weight',
        type=float,
        default=0.8,
        help='initial gate weight for RGB features; FFT starts with 1 - this value',
    )

    parser.add_argument('--val_batchsize', type=int, default=1, help='batch_size for validation')
    parser.add_argument('--savepath', default='./models_ckpt/')

    parser.add_argument('--network', default='segformer-mit_b1', help='backbone encoder')
    parser.add_argument('--encoder_dim', type=int, default=512, help='dimension of extracted features')
    parser.add_argument('--feat_size', type=int, default=16, help='output feature size')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of attentive encoder layers')
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')
    args = parser.parse_args()
    args.data_variant = 'RGB_FFT'
    return args


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.args.num_epochs)

    if args.train_goal == 2:
        for goal in [2, 1, 0]:
            print_log(f'Current train_goal={goal}:\n', trainer.log)
            trainer.args.train_goal = goal
            if goal == 2:
                trainer.args.train_stage = 's1'
                for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                    trainer.training(trainer.args, epoch)
                    trainer.validation(epoch)
                    if epoch - trainer.best_epoch > 50:
                        trainer.start_epoch = trainer.best_epoch + 1
                        break
                    elif epoch == trainer.args.num_epochs - 1:
                        trainer.start_epoch = trainer.best_epoch + 1
                        trainer.args.num_epochs = trainer.start_epoch + args.num_epochs
            else:
                trainer.args.train_stage = 's2'
                if trainer.best_model_path is None or not os.path.exists(trainer.best_model_path):
                    trainer.best_model_path = os.path.join(trainer.args.savepath, 'Dual_Branch_RGB_FFT_latest.pth')
                    torch.save(trainer.make_checkpoint_state(), trainer.best_model_path)
                trainer.args.checkpoint = trainer.best_model_path
                trainer.build_model()
                for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
                    trainer.training(trainer.args, epoch)
                    trainer.validation(epoch)
                    if trainer.args.train_goal == 1 and epoch - trainer.best_epoch > 50:
                        trainer.start_epoch = trainer.best_epoch + 1
                        trainer.args.num_epochs = trainer.start_epoch + trainer.args.num_epochs
                        break
    else:
        for epoch in range(trainer.start_epoch, trainer.args.num_epochs):
            trainer.training(trainer.args, epoch)
            trainer.validation(epoch)
