import torch
import torchvision
from omegaconf import OmegaConf

from dataset.dataset_utils import get_video_and_audio
from dataset.transforms import make_class_grid
from model.modules.attn_recorder import Recorder
from model.modules.feature_selector import CrossAttention
from utils.utils import check_if_file_exists_else_download
from scripts.example import reencode_video, reconstruct_video_from_input, decode_single_video_prediction
from scripts.train_utils import get_model, get_transforms, prepare_inputs


def main(exp_name, vid_path, vfps, afps, device, input_size, v_start_i_sec, offset_sec):
    cfg_path = f'./logs/sync_models/{exp_name}/cfg-{exp_name}.yaml'
    ckpt_path = f'./logs/sync_models/{exp_name}/{exp_name}.pt'

    # if the model does not exist try to download it from the server
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)

    # load config
    cfg = OmegaConf.load(cfg_path)

    # checking if the provided video has the correct frame rates
    print(f'Using video: {vid_path}')
    v, a, vid_meta = torchvision.io.read_video(vid_path, pts_unit='sec')
    T, H, W, C = v.shape
    if vid_meta['video_fps'] != vfps or vid_meta['audio_fps'] != afps or min(H, W) != input_size:
        print(f'Reencoding. vfps: {vid_meta["video_fps"]} -> {vfps};', end=' ')
        print(f'afps: {vid_meta["audio_fps"]} -> {afps};', end=' ')
        print(f'{(H, W)} -> min(H, W)={input_size}')
        vid_path = reencode_video(vid_path, vfps, afps, input_size)
    else:
        print(f'No need to reencode. vfps: {vid_meta["video_fps"]}; afps: {vid_meta["audio_fps"]}; min(H, W)={input_size}')

    device = torch.device(device)

    # load the model
    _, model = get_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])

    # Recorder wraps the model to access attention values
    # type2submodule = {'rgb': 'v_selector', 'audio': 'a_selector'}
    submodule_name = 'v_selector'  # else 'v_selector'
    model = Recorder(model, watch_module=CrossAttention, submodule_name=submodule_name)

    model.eval()

    # load visual and audio streams
    # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
    rgb, audio, meta = get_video_and_audio(vid_path, get_meta=True)

    # TODO: check if the offset and start are zeros and print this
    # making an item (dict) to apply transformations
    item = {
        'video': rgb, 'audio': audio, 'meta': meta, 'path': vid_path, 'split': 'test',
        'targets': {
            # setting the start of the visual crop and the offset size.
            # For instance, if the model is trained on 5sec clips, the provided video is 9sec, and `v_start_i_sec=1.3`
            # the transform will crop out a 5sec-clip from 1.3 to 6.3 seconds and shift the start of the audio
            # track by `offset_sec` seconds. It means that if `offset_sec` > 0, the audio will
            # start `offset_sec` earlier than the rgb track.
            # It is a good idea to use something in [-`max_off_sec`, `max_off_sec`] (see `grid`)
            'v_start_i_sec': v_start_i_sec,
            'offset_sec': offset_sec,
            # dummy values -- don't mind them
            'vggsound_target': 0,
            'vggsound_label': 'PLACEHOLDER',
        },
    }

    # making the offset class grid similar to the one used in transforms
    max_off_sec = cfg.data.max_off_sec
    grid = make_class_grid(-max_off_sec, max_off_sec, cfg.model.params.transformer.params.num_offset_cls)
    # TODO: maybe?
    # assert min(grid) <= offset_sec <= max(grid)

    # applying the transform
    transforms = get_transforms(cfg)['test']
    item = transforms(item)

    # prepare inputs for inference
    batch = torch.utils.data.default_collate([item])
    aud, vid, targets = prepare_inputs(batch, device)

    # sanity check: we will take the input to the `model` and recontruct make a video from it.
    # Use this check to make sure the input makes sense (audio should be ok but shifted as you specified)
    reconstruct_video_from_input(aud, vid, batch['meta'], vid_path, v_start_i_sec, offset_sec, vfps, afps)

    # forward pass
    _, off_logits, attention = model(vid, aud, targets)

    # simply prints the results of the prediction
    probs = decode_single_video_prediction(off_logits, grid, item)


if __name__ == '__main__':
    exp_name = '23-02-26T22-31-22'
    vid_path = './data/vggsound/h264_video_25fps_256side_16000hz_aac/3qesirWAGt4_20000_30000.mp4'  # dog barking
    device = 'cpu'

    # target values for an input video (the video will be reencoded to match these)
    vfps = 25
    afps = 16000
    input_size = 256

    # you may artificially offset the audio and visual tracks:
    v_start_i_sec = 0.0  # start of the visual track
    offset_sec = 1.6  # how early audio should start than the visual track

    main(exp_name, vid_path, vfps, afps, device, input_size, v_start_i_sec, offset_sec)
