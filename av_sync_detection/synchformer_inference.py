import argparse
import subprocess
from pathlib import Path

import torch
import torchvision
from omegaconf import OmegaConf

from dataset.dataset_utils import get_video_and_audio
from dataset.transforms import make_class_grid, quantize_offset
from utils.utils import check_if_file_exists_else_download, which_ffmpeg
from scripts.train_utils import get_model, get_transforms, prepare_inputs


def reencode_video(path, vfps=25, afps=16000, in_size=256):
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    new_path = Path.cwd() / 'vis' / f'{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.mp4'
    new_path.parent.mkdir(exist_ok=True)
    new_path = str(new_path)
    cmd = f'{which_ffmpeg()}'
    # no info/error printing
    cmd += ' -hide_banner -loglevel panic'
    cmd += f' -y -i {path}'
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -ar {afps}"
    cmd += f' {new_path}'
    subprocess.call(cmd.split())
    cmd = f'{which_ffmpeg()}'
    cmd += ' -hide_banner -loglevel panic'
    cmd += f' -y -i {new_path}'
    cmd += f' -acodec pcm_s16le -ac 1'
    cmd += f' {new_path.replace(".mp4", ".wav")}'
    subprocess.call(cmd.split())
    return new_path

def decode_single_video_prediction(off_logits, grid, item):
    label = item['targets']['offset_label'].item()
    print('Ground Truth offset (sec):', f'{label:.2f} ({quantize_offset(grid, label)[-1].item()})')
    print()
    print('Prediction Results:')
    off_probs = torch.softmax(off_logits, dim=-1)
    k = min(off_probs.shape[-1], 5)
    topk_logits, topk_preds = torch.topk(off_logits, k)
    # remove batch dimension
    assert len(topk_logits) == 1, 'batch is larger than 1'
    topk_logits = topk_logits[0]
    topk_preds = topk_preds[0]
    off_logits = off_logits[0]
    off_probs = off_probs[0]
    for target_hat in topk_preds:
        print(f'{grid[target_hat]:.2f}     {off_probs[target_hat]:.4f}')
    return off_probs

def patch_config(cfg):
    # the FE ckpts are already in the model ckpt
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    # old checkpoints have different names
    cfg.model.params.transformer.target = cfg.model.params.transformer.target\
                                             .replace('.modules.feature_selector.', '.sync_model.')
    return cfg

def main(args):
    vfps = 25
    afps = 16000
    in_size = 256
    cfg_path = f'./logs/sync_models/{args.exp_name}/cfg-{args.exp_name}.yaml'
    ckpt_path = f'./logs/sync_models/{args.exp_name}/{args.exp_name}.pt'

    # if the model does not exist try to download it from the server
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)

    # load config
    cfg = OmegaConf.load(cfg_path)

    # patch config
    cfg = patch_config(cfg)

    # checking if the provided video has the correct frame rates
    print(f'Using video: {args.vid_path}')
    v, _, info = torchvision.io.read_video(args.vid_path, pts_unit='sec')
    _, H, W, _ = v.shape
    if info['video_fps'] != vfps or info['audio_fps'] != afps or min(H, W) != in_size:
        print(f'Reencoding. vfps: {info["video_fps"]} -> {vfps};', end=' ')
        print(f'afps: {info["audio_fps"]} -> {afps};', end=' ')
        print(f'{(H, W)} -> min(H, W)={in_size}')
        args.vid_path = reencode_video(args.vid_path, vfps, afps, in_size)
    else:
        print(f'Skipping reencoding. vfps: {info["video_fps"]}; afps: {info["audio_fps"]}; min(H, W)={in_size}')

    device = torch.device(args.device)

    # load the model
    _, model = get_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    model.eval()

    # load visual and audio streams
    # rgb: (Tv, 3, H, W) in [0, 225], audio: (Ta,) in [-1, 1]
    rgb, audio, meta = get_video_and_audio(args.vid_path, get_meta=True)

    # making an item (dict) to apply transformations
    # NOTE: here is how it works:
    # For instance, if the model is trained on 5sec clips, the provided video is 9sec, and `v_start_i_sec=1.3`
    # the transform will crop out a 5sec-clip from 1.3 to 6.3 seconds and shift the start of the audio
    # track by `args.offset_sec` seconds. It means that if `offset_sec` > 0, the audio will
    # start by `offset_sec` earlier than the rgb track.
    # It is a good idea to use something in [-`max_off_sec`, `max_off_sec`] (-2, +2) seconds (see `grid`)
    item = dict(
        video=rgb, audio=audio, meta=meta, path=args.vid_path, split='test',
        targets={'v_start_i_sec': args.v_start_i_sec, 'offset_sec': args.offset_sec, },
    )

    # making the offset class grid similar to the one used in transforms
    max_off_sec = cfg.data.max_off_sec
    num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
    grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)
    if not (min(grid) <= item['targets']['offset_sec'] <= max(grid)):
        print(f'WARNING: offset_sec={item["targets"]["offset_sec"]} is outside the trained grid: {grid}')

    # applying the test-time transform
    item = get_transforms(cfg, ['test'])['test'](item)

    # prepare inputs for inference
    batch = torch.utils.data.default_collate([item])
    aud, vid, targets = prepare_inputs(batch, device)

    # forward pass
    with torch.set_grad_enabled(False):
        _, logits = model(
            vid.to(device, dtype=torch.float),
            aud.to(device, dtype=torch.float)
        )

    # simply prints the results of the prediction
    decode_single_video_prediction(logits, grid, item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_path', help='A path to .mp4 video', default='')
    parser.add_argument('--exp_name', help='In a format: xx-xx-xxTxx-xx-xx', default='24-01-04T16-39-21')
    parser.add_argument('--offset_sec', type=float, default=0.0)
    parser.add_argument('--v_start_i_sec', type=float, default=0.0)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()

    true_offset = "p1"
    video_source = "live-lounge"
    args.vid_path = f"../../../datasets/tv/{video_source}/offset-{true_offset}s.mp4"

    main(args)
