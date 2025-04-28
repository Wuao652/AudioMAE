import os
import numpy as np

import matplotlib.pyplot as plt
import argparse

import torch
import scipy
import cv2

import sys
import ipdb
from IPython import embed as e

# example 0
wav_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_release/train/00241_Animalia_Arthropoda_Insecta_Hemiptera_Cicadidae_Notopsalta_sericea/bd20bebd-6d3b-4c19-bdb2-078c615f20a9.wav"
np_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_new_spec/train/00241_Animalia_Arthropoda_Insecta_Hemiptera_Cicadidae_Notopsalta_sericea/bd20bebd-6d3b-4c19-bdb2-078c615f20a9.npy"
vis_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_new_spec/train_vis/00241_Animalia_Arthropoda_Insecta_Hemiptera_Cicadidae_Notopsalta_sericea/bd20bebd-6d3b-4c19-bdb2-078c615f20a9.jpg"
example_id = 0

# example 1
wav_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_release/train/03526_Animalia_Chordata_Aves_Passeriformes_Parulidae_Setophaga_palmarum/3f598c18-28db-4577-b4d2-f0dd36450b4b.wav"
np_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_new_spec/train/03526_Animalia_Chordata_Aves_Passeriformes_Parulidae_Setophaga_palmarum/3f598c18-28db-4577-b4d2-f0dd36450b4b.npy"
vis_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_new_spec/train_vis/03526_Animalia_Chordata_Aves_Passeriformes_Parulidae_Setophaga_palmarum/3f598c18-28db-4577-b4d2-f0dd36450b4b.jpg"
example_id = 1

# example 2
wav_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_release/train/03526_Animalia_Chordata_Aves_Passeriformes_Parulidae_Setophaga_palmarum/728179d7-59e2-4ad5-8c2f-2396d047ed4b.wav"
np_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_new_spec/train/03526_Animalia_Chordata_Aves_Passeriformes_Parulidae_Setophaga_palmarum/728179d7-59e2-4ad5-8c2f-2396d047ed4b.npy"
vis_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_new_spec/train_vis/03526_Animalia_Chordata_Aves_Passeriformes_Parulidae_Setophaga_palmarum/728179d7-59e2-4ad5-8c2f-2396d047ed4b.jpg"
example_id = 2

# vis_path = "/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_new_spec/train_vis/00241_Animalia_Arthropoda_Insecta_Hemiptera_Cicadidae_Notopsalta_sericea/eb3aa5f4-c6e4-4e9a-bc62-06f31621050b.jpg"

output_dir = "./outputs"
energy_thres = 200
os.makedirs(os.path.join(output_dir, str(example_id)), exist_ok=True)

def read_sound(spec_path):
    print(f"reading sound from {spec_path} ...")
    img = np.load(spec_path)
    img = np.stack([img]*1, axis=-1)    # [H, W] -> [H, W, 1]
    window_len = 512
    test_stride = 256
    time_len = img.shape[-2]

    if time_len < window_len:
        pad = window_len - time_len
        # img = np.pad(img, ((0, 0), (pad//2, pad - pad//2), (0, 0)), constant_values=0)
        img = np.pad(img, ((0, 0), (0, pad), (0, 0)), constant_values=0)
        time_len = img.shape[-2]


    num = (time_len - window_len + test_stride - 1) // test_stride
    pad = test_stride * num - time_len + window_len
    img = np.pad(img, ((0, 0), (0, pad), (0, 0)), constant_values=0)

    img_frames = []
    idx_tuples = []
    # print(num, time_len)
    for i in range(num+1):
        start = i * test_stride
        end = start + window_len
        if end > img.shape[-2]: break
        idx_tuples.append((start, end))

        # print(out_list[-1].shape, start, start+window_len)

        frame = img[..., start : end, :]
        img_frames.append(frame)
        # cur_img = torch.Tensor(cur_img).to(torch.float) / 255.0
        # cur_img = cur_img.permute(2, 0, 1)
        # cur_img = cur_img.unsqueeze(0)
        # cur_img = standard_transforms(cur_img)
        # frames_list.append(cur_img)

    return img, img_frames, idx_tuples



# mapping between (start_spec, end_spec) of spec and 
# Typically, energy = sum of squares of the amplitude values
# For a signal x(t), energy = sum of x(t)^2 over time.
def frame_to_wav(frame_idx, max_length=np.inf, window_length_samples=512, hop_length_samples=128):
    """
    Convert frame indices to waveform indices.
    :param frame_idx: (start_frame, end_frame)
    :param window_length_samples: length of the window in samples
    :param hop_length_samples: hop length in samples
    :return: (start_wav, end_wav)
    """

    # estimated_t = 1 + (waveform.shape[0] - window_length_samples) // hop_length_samples
    wav_idx = (frame_idx - 1) * hop_length_samples + window_length_samples
    return min(wav_idx, max_length)



if __name__ == "__main__":
    print("hello world from energy thresholding ... ")

    # read the sound using librosa
    # y, sr = librosa.load(wav_path, sr=None)
    # print("y: ", y.shape, y.dtype)
    # print(y.min(), y.max())
    # print("sr: ", sr)


    sr, waveform = scipy.io.wavfile.read(wav_path)      # sample rate is 22050
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    print("sr: ", sr)
    print("wavform: ", waveform.shape, waveform.dtype)
    print(waveform.min(), waveform.max())
    
    # peak normalization
    # according to the following wikipedia page:
    # https://en.wikipedia.org/wiki/Audio_normalization
    waveform = waveform / np.max(np.abs(waveform))

    
    
    
    wav_length = waveform.shape[0]

    img_np, img_frames, idx_turples = read_sound(np_path)
    print("img_frames:\n", img_frames[0].shape)
    print("idx_turples:\n ", idx_turples)

    #img_np = np.load(np_path)
    #img_np = np.stack([img_np]*1, axis=-1)  # [H, W] -> [H, W, 1]
    img_keep = np.zeros_like(img_np)
    w_keep = []

    for i, (start, end) in enumerate(idx_turples):
        print("=" * 200)
        print(start, end)
        start_wav = frame_to_wav(start, max_length=wav_length)
        end_wav = frame_to_wav(end, max_length=wav_length)
        print("start_wav: ", start_wav)
        print("end_wav: ", end_wav)
        wav_frame = waveform[start_wav: end_wav]
        print("wav_frame: ", wav_frame.shape)

        wav_frame_energy = np.sum(wav_frame ** 2)
        print("wav_frame_energy: ", wav_frame_energy)

        if wav_frame_energy >= energy_thres:
            print("save the frame ...")
            wav_frame_name = f"{start}_{end}.jpg"
            cv2.imwrite(os.path.join(output_dir, str(example_id), wav_frame_name), 255 - img_frames[i])

            # visualize the img that are kept
            img_keep[..., start:end, :] = img_frames[i]
            w_keep.extend(list(range(start, end)))


    # get the unique values
    w_keep = list(set(w_keep))
    w_keep.sort()
    print("w_keep:\n", w_keep)
    img_filtered = img_np[:, w_keep, :]


    # save the original img
    cv2.imwrite(os.path.join(output_dir, str(example_id), "orig_img.jpg"), 255 - img_np)
    # save the img_keep
    cv2.imwrite(os.path.join(output_dir, str(example_id), "img_keep.jpg"), 255 - img_keep)
    # save the img_filtered
    cv2.imwrite(os.path.join(output_dir, str(example_id), "img_filtered.jpg"), 255 - img_filtered)

# for pretraining, what we do is 
# remove the white noise part of the model and keep with the sliding window things










    # window_length_samples=512
    # hop_length_samples=128

    # spec = np.load(np_path)
    # print("spec: ", spec.shape)
    # estimated_t = 1 + (waveform.shape[0] - window_length_samples) // hop_length_samples
    # print("estimated_t: ", estimated_t)
    
    # print((3 * 22050 - 512) // 128 + 1)

# how to threshold the np_array
# dumpt the results

