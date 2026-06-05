import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from imageio.v2 import imread, imwrite
from PIL import Image
from skimage import measure
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_LORA_ADAPTER_PATH = ROOT / "weights" / "adapter"
sys.path.insert(0, str(ROOT))

from checkpoint_loader import load_checkpoint  # noqa: E402
from model.model_decoder import DecoderTransformer  # noqa: E402
from model.model_encoder_att import AttentiveEncoder, Encoder  # noqa: E402
from train_2 import RGBFFTFusionEncoder  # noqa: E402


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


class DualBranchGuide:
    def __init__(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("This Dual-Branch decoder uses .cuda() internally, so CUDA is required.")

        self.device = torch.device("cuda")
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

        with open(args.vocab_json, "r", encoding="utf-8") as f:
            self.word_vocab = json.load(f)
        self.id_to_word = {idx: word for word, idx in self.word_vocab.items()}
        self.dual_mode = args.dual_mode
        self.fft_data_root = args.fft_data_root

        checkpoint = load_checkpoint(args.dual_checkpoint, map_location="cpu")
        if self.dual_mode == "rgb_fft":
            self.encoder = RGBFFTFusionEncoder(args.network)
        else:
            self.encoder = Encoder(args.network)
        self.encoder_trans = AttentiveEncoder(
            train_stage=None,
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

        self.encoder.load_state_dict(checkpoint["encoder_dict"])
        self.encoder_trans.load_state_dict(checkpoint["encoder_trans_dict"], strict=False)
        self.decoder.load_state_dict(checkpoint["decoder_dict"])

        self.encoder.eval().to(self.device)
        self.encoder_trans.eval().to(self.device)
        self.decoder.eval().to(self.device)

    def preprocess(self, image_path):
        image = imread(image_path)
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[-1] > 3:
            image = image[..., :3]
        if image.shape[:2] != (256, 256):
            image = np.asarray(Image.fromarray(image.astype(np.uint8)).resize((256, 256), Image.BILINEAR))

        image = image.astype(np.float32).transpose(2, 0, 1)
        for channel in range(3):
            image[channel] -= self.mean[channel]
            image[channel] /= self.std[channel]
        return torch.FloatTensor(image).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def infer(self, image_a, image_b, mask_path=None):
        img_a = self.preprocess(image_a)
        img_b = self.preprocess(image_b)

        if self.dual_mode == "rgb_fft":
            img_a_fft = self.preprocess(rgb_to_fft_path(image_a, self.fft_data_root))
            img_b_fft = self.preprocess(rgb_to_fft_path(image_b, self.fft_data_root))
            feat1, feat2 = self.encoder(img_a, img_b, img_a_fft, img_b_fft)
        else:
            feat1, feat2 = self.encoder(img_a, img_b)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
        seq = self.decoder.sample(feat1, feat2, k=1)

        ignore = {
            self.word_vocab["<START>"],
            self.word_vocab["<END>"],
            self.word_vocab["<NULL>"],
        }
        caption_tokens = [self.id_to_word[token] for token in seq if token not in ignore]
        global_caption = " ".join(caption_tokens).strip()

        pred_mask = torch.argmax(seg_pre, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
        if mask_path is not None:
            self.save_mask(pred_mask, mask_path)

        return {
            "global_caption": global_caption,
            "mask": pred_mask,
            "road_count": self.count_objects(pred_mask, 1),
            "building_count": self.count_objects(pred_mask, 2),
            "road_locations": self.grid_locations(pred_mask, 1),
            "building_locations": self.grid_locations(pred_mask, 2),
        }

    def save_mask(self, pred_mask, mask_path):
        mask_path = Path(mask_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        rgb[pred_mask == 1] = [0, 255, 255]
        rgb[pred_mask == 2] = [255, 0, 0]
        imwrite(mask_path, rgb)

    @staticmethod
    def count_objects(pred_mask, class_id, min_area=5):
        binary = pred_mask == class_id
        labeled = measure.label(binary, connectivity=2)
        props = measure.regionprops(labeled)
        return sum(1 for prop in props if prop.area > min_area)

    @staticmethod
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


class QwenVL:
    def __init__(
        self,
        model_path,
        max_new_tokens,
        max_pixels,
        attn_implementation,
        use_lora=False,
        lora_adapter_path=None,
    ):
        try:
            from modelscope import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        if use_lora:
            from peft import PeftModel

        kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
        if use_lora:
            lora_adapter_path = Path(lora_adapter_path).expanduser()
            if not lora_adapter_path.exists():
                raise FileNotFoundError(f"LoRA adapter path not found: {lora_adapter_path}")
            model = PeftModel.from_pretrained(model, str(lora_adapter_path))
        self.model = model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"
        self.process_vision_info = process_vision_info
        self.gen_config = {"max_new_tokens": max_new_tokens}

    @torch.inference_mode()
    def generate(self, prompt, images):
        return self.generate_batch([prompt], [images])[0]

    @torch.inference_mode()
    def generate_batch(self, prompts, images_batch):
        messages_batch = []
        texts = []
        for prompt, images in zip(prompts, images_batch):
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            messages_batch.append(messages)
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    add_vision_id=True,
                )
            )

        image_inputs = []
        video_inputs = []
        for messages in messages_batch:
            sample_image_inputs, sample_video_inputs = self.process_vision_info(messages)
            if sample_image_inputs:
                image_inputs.extend(sample_image_inputs)
            if sample_video_inputs:
                video_inputs.extend(sample_video_inputs)

        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, **self.gen_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        responses = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        return [response.strip() for response in responses]


def parse_path_maps(path_maps):
    maps = []
    for item in path_maps:
        if "=" not in item:
            raise ValueError(f"--path-map must be SRC=DST, got: {item}")
        src, dst = item.split("=", 1)
        maps.append((src, dst))
    return maps


def remap_path(path, path_maps):
    mapped = path
    for src, dst in path_maps:
        if mapped.startswith(src):
            mapped = dst + mapped[len(src) :]
    return mapped


def rgb_to_fft_path(image_path, fft_data_root):
    image_path = Path(image_path)
    split = image_path.parent.parent.name
    branch = image_path.parent.name
    return str(Path(fft_data_root) / split / branch / image_path.name)


def load_samples(dataset_path):
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        json_files = sorted(dataset_path.glob("*.json"))
    else:
        json_files = [dataset_path]

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        for idx, sample in enumerate(samples):
            yield json_file.stem, idx, sample


def get_instruction_and_reference(sample):
    instruction = ""
    reference = ""
    for turn in sample.get("conversations", []):
        if turn.get("from") == "human" and not instruction:
            instruction = turn.get("value", "")
        elif turn.get("from") == "gpt" and not reference:
            reference = turn.get("value", "")
    return instruction, reference


def build_visual_guided_prompt(raw_instruction, dual_prior, include_mask):
    road_locations = ",".join(dual_prior["road_locations"])
    building_locations = ",".join(dual_prior["building_locations"])
    mask_note = ""
    if include_mask:
        mask_note = (
            "\nThe third image is the predicted change mask from the Dual-Branch module. "
            "Cyan indicates road changes and red indicates building changes."
        )

    return (
        "Visual-guided context from the Dual-Branch module:\n"
        f"- Global summary caption: {dual_prior['global_caption'] or 'No caption'}\n"
        f"- Changed road count: {dual_prior['road_count']}\n"
        f"- Changed road locations: {road_locations}\n"
        f"- Changed building count: {dual_prior['building_count']}\n"
        f"- Changed building locations: {building_locations}"
        f"{mask_note}\n\n"
        "User instruction from ChangeIMTI:\n"
        f"{raw_instruction}"
    )


def sample_id_from_images(images):
    if not images:
        return ""
    return Path(images[0]).stem


def resolve_dual_checkpoint(args):
    if args.dual_checkpoint is not None:
        return args.dual_checkpoint
    if args.dual_mode == "rgb_fft":
        return str(ROOT / "weights" / "Dual_Branch_FFT" / "Dual_Branch_FFT.safetensors")
    return str(ROOT / "weights" / "Dual_Branch" / "Dual_Branch.safetensors")


def main():
    parser = argparse.ArgumentParser(description="Qwen inference with Dual-Branch visual-guided prompts.")
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Qwen2.5-VL model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--dataset-json",
        default=str(REPO_ROOT / "data" / "coding" / "muti_task_data" / "test_task_data"),
        help="A ChangeIMTI json file or a directory containing task json files.",
    )
    parser.add_argument(
        "--dual-mode",
        choices=["rgb", "rgb_fft"],
        default="rgb",
        help="rgb: use RGB Dual_Branch prior; rgb_fft: use RGB+FFT Dual_Branch prior.",
    )
    parser.add_argument(
        "--dual-checkpoint",
        default=None,
        help="Optional path to a .pth or .safetensors checkpoint. If omitted, selected by --dual-mode.",
    )
    parser.add_argument(
        "--fft-data-root",
        default=str(REPO_ROOT / "data" / "coding" / "datasets" / "LEVIR-MCI-dataset-fft" / "images"),
        help="Root folder for FFT images with split/A,B subfolders.",
    )
    parser.add_argument("--vocab-json", default=str(ROOT / "data" / "LEVIR_MCI" / "vocab.json"))
    parser.add_argument("--output-jsonl", default=str(REPO_ROOT / "changevg_qwen_predictions.jsonl"))
    parser.add_argument("--mask-dir", default=str(REPO_ROOT / "changevg_qwen_masks"))
    parser.add_argument(
        "--mode",
        choices=["qwen_only", "guided_text", "guided_full"],
        default="guided_text",
        help="qwen_only: original JSON prompt only; guided_text: add Dual-Branch priors; guided_full: add priors and mask image.",
    )
    parser.add_argument("--path-map", action="append", default=[f"/data/coding={REPO_ROOT / 'data' / 'coding'}"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--batch-size", type=int, default=512, help="Number of Qwen samples to generate at once.")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--use-lora",
        "--use_lora",
        action="store_true",
        dest="use_lora",
        help="Load Qwen with a PEFT LoRA adapter.",
    )
    parser.add_argument(
        "--lora-adapter-path",
        "--lora_adapter_path",
        default=str(DEFAULT_LORA_ADAPTER_PATH),
        dest="lora_adapter_path",
        help="LoRA adapter directory containing adapter_config.json and adapter_model.safetensors.",
    )

    parser.add_argument("--network", default="segformer-mit_b1")
    parser.add_argument("--encoder-dim", type=int, default=512)
    parser.add_argument("--feat-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--decoder-n-layers", type=int, default=1)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=41)
    args = parser.parse_args()
    args.dual_checkpoint = resolve_dual_checkpoint(args)

    path_maps = parse_path_maps(args.path_map)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_dual = args.mode in {"guided_text", "guided_full"}
    include_mask = args.mode == "guided_full"
    dual = DualBranchGuide(args) if use_dual else None
    qwen = QwenVL(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
        use_lora=args.use_lora,
        lora_adapter_path=args.lora_adapter_path,
    )

    samples = list(load_samples(args.dataset_json))
    if args.limit is not None:
        samples = samples[args.start : args.start + args.limit]
    else:
        samples = samples[args.start :]

    batch_size = max(1, args.batch_size)
    with open(output_path, "w", encoding="utf-8") as out:
        for batch_start in tqdm(range(0, len(samples), batch_size), desc="ChangeVG-Qwen inference"):
            batch_samples = samples[batch_start : batch_start + batch_size]
            records = []
            prompts = []
            qwen_images_batch = []

            for task_name, sample_index, sample in batch_samples:
                raw_instruction, reference = get_instruction_and_reference(sample)
                images = [remap_path(path, path_maps) for path in sample.get("images", [])]
                if len(images) < 2:
                    raise ValueError(f"Sample has fewer than two images: task={task_name}, index={sample_index}")

                dual_prior = None
                mask_path = None
                prompt = raw_instruction
                qwen_images = images[:2]

                if use_dual:
                    mask_path = str(Path(args.mask_dir) / task_name / f"{sample_id_from_images(images)}_mask.png")
                    dual_prior = dual.infer(images[0], images[1], mask_path=mask_path if include_mask else None)
                    prompt = build_visual_guided_prompt(raw_instruction, dual_prior, include_mask=include_mask)
                    if include_mask:
                        qwen_images = images[:2] + [mask_path]

                records.append({
                    "task": task_name,
                    "index": sample_index,
                    "sample_id": sample_id_from_images(images),
                    "mode": args.mode,
                    "dual_mode": args.dual_mode,
                    "use_lora": args.use_lora,
                    "lora_adapter_path": args.lora_adapter_path if args.use_lora else None,
                    "images": images,
                    "mask_path": mask_path,
                    "instruction": raw_instruction,
                    "prompt": prompt,
                    "prediction": "",
                    "reference": reference,
                    "dual_prior": {
                        key: value for key, value in (dual_prior or {}).items() if key != "mask"
                    },
                    "error": None,
                })
                prompts.append(prompt)
                qwen_images_batch.append(qwen_images)

            try:
                predictions = qwen.generate_batch(prompts, qwen_images_batch)
                for record, prediction in zip(records, predictions):
                    record["prediction"] = prediction
            except Exception as exc:
                error = repr(exc)
                for record in records:
                    record["error"] = error

            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
