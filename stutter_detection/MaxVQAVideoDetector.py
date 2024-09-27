import os
import yaml
import torch
import numpy as np

from ExplainableVQA.open_clip.src import open_clip
from ExplainableVQA.DOVER.dover import DOVER
from ExplainableVQA.DOVER.dover.datasets import UnifiedFrameSampler, get_single_view
from ExplainableVQA.model import TextEncoder, MaxVQA, EnhancedVisualEncoder

torch.cuda.empty_cache()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAXVQA_CONF = os.path.join(ROOT_DIR, "stutter_detection/ExplainableVQA/maxvqa.yml")
MAXVQA_WEIGHTS = os.path.join(ROOT_DIR, "stutter_detection/ExplainableVQA/maxvqa_maxwell.pt")
DOVER_WEIGHTS = os.path.join(ROOT_DIR, "stutter_detection/ExplainableVQA/DOVER/pretrained_weights/DOVER.pth")


# Setup variables & parameters
dimension_names = [
    "overall quality score",
    "contents", "composition", "color", "lighting", "camera (temporal) trajectory",
    "aesthetic perspective",
    "sharpness", "focus", "noise", "motion blur", "flicker", "exposure", "compression artefacts", "motion (frame) fluency",
    "technical perspective",
]

positive_descs = [
    "high quality",
    "good content", "organized composition", "vibrant color", "contrastive lighting", "consistent trajectory",
    "good aesthetics",
    "sharp", "in-focus", "noiseless", "clear-motion", "stable", "well-exposed", "original", "fluent",
    "clear",
]

negative_descs = [
    "low quality",
    "bad content", "chaotic composition", "faded color", "gloomy lighting", "incoherent trajectory",
    "bad aesthetics",
    "fuzzy", "out-of-focus", "noisy", "blurry-motion", "shaky", "poorly-exposed", "compressed", "choppy",
    "severely degraded",
]

context = "X"
pos_prompts = [ f"a {context} {desc} photo" for desc in positive_descs]
neg_prompts = [ f"a {context} {desc} photo" for desc in negative_descs]


# Define processing outer functions
def encode_text_prompts(prompts, tokenizer, model, device='cpu'):
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        embedding = model.token_embedding(text_tokens)
        text_features = model.encode_text(text_tokens).float()

    return text_tokens, embedding, text_features


def setup_models(text_prompts, opt, aesthetic_clip_len, technical_num_clips, device='cpu', use_aesthetic_features=False):
    # Initialize fast-vqa encoder
    fast_vqa_encoder = DOVER(**opt["model"]["args"]).to(device)
    fast_vqa_encoder.load_state_dict(
        torch.load(DOVER_WEIGHTS, map_location=device),
        strict=False
    )

    # Initialize CLIP model
    clip_model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
    clip_model = clip_model.to(device)

    # Encode initialized text prompts
    tokenizer = open_clip.get_tokenizer("RN50")
    text_tokens, embedding, text_feats = encode_text_prompts(text_prompts, tokenizer, clip_model, device=device)
    text_encoder = TextEncoder(clip_model)

    # Visual encoder
    technical_clip_len = opt["inference"]["args"]["sample_types"]["technical"]["clip_len"]
    visual_encoder = EnhancedVisualEncoder(clip_model, fast_vqa_encoder, aesthetic_clip_len, technical_clip_len, technical_num_clips)

    # Initialise data samplers
    frame_interval = opt["inference"]["args"]["sample_types"]["technical"]["frame_interval"]

    temporal_samplers = {
        "technical": UnifiedFrameSampler(
            technical_clip_len,
            technical_num_clips,
            frame_interval
        )
    }

    # Generate MaxVQA model
    maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True, device=device)

    state_dict = torch.load(MAXVQA_WEIGHTS, map_location=device)
    maxvqa.load_state_dict(state_dict)
    maxvqa.initialize_inference(text_encoder)

    return text_encoder, visual_encoder, temporal_samplers, maxvqa


def extract_video_features(video, encoder, opt, temporal_samplers, use_aesthetic_features=False, device='cpu'):
    # Video preprocessing module
    mean = torch.FloatTensor([123.675, 116.28, 103.53]).to(device).reshape(-1,1,1,1)
    std = torch.FloatTensor([58.395, 57.12, 57.375]).to(device).reshape(-1,1,1,1)

    sample_feature_type = {"technical": opt["inference"]["args"]["sample_types"]["technical"]}
    video_data, frame_idx = spatial_temporal_view_decomposition(video, sample_feature_type, temporal_samplers, device=device)

    # Assuming that video_data is the preprocessed video from above step
    if use_aesthetic_features:
        frame_idx["aesthetic"] = frame_idx["technical"][0::2]
        data = {
            "technical": (video_data["technical"] - mean ) / std,
            "aesthetic": (video_data["technical"][:, 0::2] - mean ) / std
        }
        vis_feats = encoder(data["technical"].to(device), data["aesthetic"].to(device))
    else:
        data = {"technical": (video_data["technical"] - mean ) / std}
        vis_feats = encoder(data["technical"].to(device))

    return vis_feats, frame_idx["technical"][0::2]


def spatial_temporal_view_decomposition(
    vreader, sample_types, samplers, is_train=False, augment=False, device='cpu'
):
    video = {}

    all_frame_inds = []
    frame_inds = {}
    for stype in samplers:
        frame_inds[stype] = samplers[stype](len(vreader), is_train)
        all_frame_inds.append(frame_inds[stype])

    ### Each frame is only decoded one time!!!
    all_frame_inds = np.concatenate(all_frame_inds, 0)
    frame_dict = {idx: vreader[idx] for idx in np.unique(all_frame_inds)}

    for stype in samplers:
        imgs = [frame_dict[idx] for idx in frame_inds[stype]]
        video[stype] = torch.stack(imgs, 0).to(device).permute(3, 0, 1, 2)

    sampled_video = {}
    for stype, sopt in sample_types.items():
        sampled_video[stype] = get_single_view(video[stype], stype, **sopt)

    return sampled_video, frame_inds


# Define actual detection module to be used
class VideoDetector():
    def __init__(self, frames=64, device='cpu') -> None:
        self.frames = frames
        self.device = torch.device(device)
        self.load_model()

    def load_model(self):
        with open(MAXVQA_CONF, 'r') as f:
            self.opt = yaml.safe_load(f)

        aesthetic_clip_len = self.frames               # length of single fragment for aesthetic analysis
        technical_num_clips = 2 * (self.frames // 32)  # no. frame fragments for techical analysis (each len 32)

        pos_neg_prompts = pos_prompts + neg_prompts

        self.text_encoder, self.visual_encoder, self.temporal_samplers, self.maxvqa = setup_models(
            pos_neg_prompts,
            self.opt,
            aesthetic_clip_len,
            technical_num_clips,
            device=self.device
        )

    def process(self, video=None):
        video = torch.Tensor(video).to(self.device)
        features = self.feature_extraction(video)
        results = self.predict(features)
        return results

    def feature_extraction(self, video_frames=None):
        # Extract features from test video
        vis_feats, sampled_frames = extract_video_features(
            video_frames,
            self.visual_encoder,
            self.opt,
            self.temporal_samplers,
            device=self.device
        )

        return vis_feats

    def predict(self, visual_features):
        # Run test video features through detection model
        combined_outputs = []

        with torch.no_grad():
            raw_outputs = self.maxvqa(
                visual_features,
                self.text_encoder,
                train=False,
                local=True
            )

        combined_outputs.append(raw_outputs)

        return combined_outputs
