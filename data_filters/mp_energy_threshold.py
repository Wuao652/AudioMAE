# A multiprocess version of applying energy threshold to inastsounds data

import os
import os.path as osp
import json
import numpy as np 
import cv2 
import scipy
import matplotlib.pyplot as plt

from multiprocessing import Process

import argparse
import math

import sys
import ipdb

parser = argparse.ArgumentParser(description="Energy thresholding")
parser.add_argument(
    "--dataset_dir", type=str, default="/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_release"
)
parser.add_argument(
    "--mel_dir", type=str, default="/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_new_spec"
)
parser.add_argument(
    "--output_dir", type=str, default="/scratch3/workspace/wuaoliu_umass_edu-inat_sounds/data/inatsounds_energy_threshold"
)
parser.add_argument(
    "--data_split", type=str, default="train", help="train, val, test"
)
parser.add_argument(
    "--threshold", type=int,default=200, help="energy threshold"
)
parser.add_argument(
    "--n_threads", type=int, default=128, help="number of threads to use for multiprocessing"
)
parser.add_argument(
    "--dump_json_only", action="store_true", default=False, help="dump json files only"
)
# parser.set_defaults(dump_json_only=True)
args = parser.parse_args()
# dump_json_only = True
args.json_path = osp.join(args.dataset_dir, f"{args.data_split}.json")
args.output_dir = osp.join(args.output_dir, f"thres_{str(args.threshold)}")




def frame_to_wav(frame_idx, max_length=np.inf, window_length_samples=512, hop_length_samples=128):
    # estimated_t = 1 + (waveform.shape[0] - window_length_samples) // hop_length_samples
    wav_idx = (frame_idx - 1) * hop_length_samples + window_length_samples
    return min(wav_idx, max_length)

def read_sound(spec_path):
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

    for i in range(num+1):
        start = i * test_stride
        end = start + window_len
        if end > img.shape[-2]: break
        idx_tuples.append((start, end))
        frame = img[..., start : end, :]
        img_frames.append(frame)

    return img, img_frames, idx_tuples

def energy_thresholding(wav_path, mel_path, np_path, vis_path, stat_dict_path=None, threshold=200):
    # sample rate is 22050
    sr, waveform = scipy.io.wavfile.read(wav_path)
    wav_length = waveform.shape[0]
    if wav_length <= 0:
        file_name = "/".join(wav_path.split("/")[-3:])
        print(f"Empty audio file found in : {file_name} ...")
        return None
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0   
    # peak normalization
    # according to the following:
    # https://en.wikipedia.org/wiki/Audio_normalization
    # Not divided by zero
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    img_np, img_frames, idx_turples = read_sound(mel_path)
    img_keep = np.zeros_like(img_np)
    w_keep = []
    stat_dict = {}
    stat_dict["file_name"] = "/".join(wav_path.split("/")[-3:])
    stat_dict["threshold"] = threshold
    stat_dict["data"] = []
    
    for i, (start, end) in enumerate(idx_turples):
        start_wav = frame_to_wav(start, max_length=wav_length)
        end_wav = frame_to_wav(end, max_length=wav_length)
        wav_frame = waveform[start_wav:end_wav]
        wav_frame_energy = np.sum(np.square(wav_frame))

        is_window_keep = False
        if wav_frame_energy >= threshold:
            # # saveing the intermediate windows
            # wav_frame_name = f"{start}_{end}.jpg"
            # os.makedirs(osp.dirname(vis_path), exist_ok=True)
            # cv2.imwrite(osp.join(osp.dirname(vis_path), wav_frame_name), 255-img_frames[i])

            is_window_keep = True
            img_keep[..., start:end, :] = img_frames[i]
            w_keep.extend(list(range(start, end)))
        
        stat_dict["data"].append(
            {
                "window_id": int(i),
                "start": int(start),
                "end": int(end),
                "start_wav": int(start_wav),
                "end_wav": int(end_wav),
                "window_energy": float(wav_frame_energy),
                "is_window_keep": is_window_keep,
            }
        )

    # save vis files
    os.makedirs(osp.dirname(vis_path), exist_ok=True)
    img_keep_vis = np.vstack([img_np, img_keep])
    # cv2.imwrite(osp.join(osp.dirname(vis_path), osp.basename(vis_path).split(".")[0]+"_orig.jpg"), 255-img_np)
    cv2.imwrite(osp.join(osp.dirname(vis_path), osp.basename(vis_path).split(".")[0]+"_keep.jpg"), 255-img_keep_vis)

    # save stat_dict
    if stat_dict_path is not None:
        os.makedirs(osp.dirname(stat_dict_path), exist_ok=True)
        with open(stat_dict_path, "w") as f:
            json.dump(stat_dict, f, indent=2)
    
    if len(w_keep) == 0:
        print(f"No clip is found above threshold in : {stat_dict['file_name']}")
        return stat_dict
    
    w_keep = list(set(w_keep))
    w_keep.sort()
    img_filtered = img_np[..., w_keep, :]
    # save npy file
    os.makedirs(osp.dirname(np_path), exist_ok=True)
    np.save(np_path, img_filtered)
    # save filtered img file
    os.makedirs(osp.dirname(vis_path), exist_ok=True)
    cv2.imwrite(vis_path, 255-img_filtered)

    # img_keep_vis = np.vstack([img_np, img_keep])
    # # cv2.imwrite(osp.join(osp.dirname(vis_path), osp.basename(vis_path).split(".")[0]+"_orig.jpg"), 255-img_np)
    # cv2.imwrite(osp.join(osp.dirname(vis_path), osp.basename(vis_path).split(".")[0]+"_keep.jpg"), 255-img_keep_vis)

    return stat_dict


def convert_file_list(args, file_name_list):
    for file_name in file_name_list:
        # print("----------------------")
        # print(file_name)
        wav_path = osp.join(args.dataset_dir, file_name)
        mel_path = osp.join(args.mel_dir, file_name.replace(".wav", ".npy"))
        np_path = osp.join(args.output_dir, file_name.replace(".wav", ".npy"))
        vis_path = osp.join(args.output_dir, file_name.replace(args.data_split, args.data_split+"_vis").replace(".wav", ".jpg"))
        stat_dict_path = osp.join(args.output_dir, file_name.replace(args.data_split, args.data_split+"_stat_dict").replace(".wav", ".json"))
        # print(wav_path)
        # print(mel_path)
        # print(np_path)
        # print(vis_path)
        # print(stat_dict_path)
        energy_thresholding(wav_path,
                            mel_path,
                            np_path,
                            vis_path,
                            stat_dict_path,
                            threshold=args.threshold)


def main():
    print("hello world from energy thresholding ...")
    print("args:\n", vars(args))
    os.makedirs(args.output_dir, exist_ok=True)

    # load the annotation
    with open(args.json_path, "r") as f:
        data = json.load(f)
        categories = data["categories"]
        audio = data["audio"]
        anno = data["annotations"]
    print("number of audio files: ", len(audio))

    all_audio_names = []
    for i, au in enumerate(audio):
        all_audio_names.append(au["file_name"])
    # print(all_audio_names[:5])
    # all_audio_names = all_audio_names[:128]

    # all_audio_names = [
    #     "train/00241_Animalia_Arthropoda_Insecta_Hemiptera_Cicadidae_Notopsalta_sericea/bd20bebd-6d3b-4c19-bdb2-078c615f20a9.wav",
    #     "train/03526_Animalia_Chordata_Aves_Passeriformes_Parulidae_Setophaga_palmarum/3f598c18-28db-4577-b4d2-f0dd36450b4b.wav",
    #     "train/03526_Animalia_Chordata_Aves_Passeriformes_Parulidae_Setophaga_palmarum/728179d7-59e2-4ad5-8c2f-2396d047ed4b.wav",
    #     ]
    # print(all_audio_names)
    # convert_file_list(args, all_audio_names)



    n_threads = args.n_threads
    processes = []
    size_each = math.ceil(len(all_audio_names) / n_threads)
    print(size_each)
    

    for rank in range(n_threads):
        strt = rank * size_each
        end = min((rank + 1) * size_each, len(all_audio_names))
        entries = all_audio_names[strt:end]

        p = Process(target=convert_file_list, args=(args, entries))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def dump_filtered_json():
    print("hello world from dump filtered json ...")
    print("args:\n", vars(args))
    os.makedirs(args.output_dir, exist_ok=True)

    # path to filtered json file
    filtered_json_path = osp.join(args.output_dir, f"{args.data_split}.json")

    with open(args.json_path, "r") as f:
        data = json.load(f)
    print(data.keys())
    filtered_data = dict()
    filtered_data["info"] = data["info"]
    filtered_data["categories"] = data["categories"]
    filtered_data["licenses"] = data["licenses"]
    filtered_data["audio"] = []
    filtered_data["annotations"] = []
    print(len(data["categories"]))
    print(data["categories"][0])

    # info, categories, audio, annotations, licenses
    print("number of audio files: ", len(data["audio"]))
    keep_audio_ids = []
    for au in data["audio"]:
        # print("=" * 40)
        # print(au)
        id = au["id"]
        file_name = au["file_name"]
        # check if the file exists
        np_path = osp.join(args.output_dir, file_name.replace(".wav", ".npy"))
        if osp.exists(np_path):
            keep_audio_ids.append(id)
            filtered_data["audio"].append(au)
    keep_audio_ids.sort()
    print("length of keep audio ids: ", len(keep_audio_ids))
    # print("keep audio ids:\n", keep_audio_ids)
    print("number of keep audio files: ", len(filtered_data["audio"]))
    
    for anno in data["annotations"]:
        audio_id = anno["audio_id"]
        category_id = anno["category_id"]
        if audio_id in keep_audio_ids:
            filtered_data["annotations"].append(anno)
    print("number of annotations: ", len(filtered_data["annotations"]))
    
    # dump the filtered json
    with open(filtered_json_path, "w") as f:
        json.dump(filtered_data, f, indent=2)
    print(f"filtered json file save to {filtered_json_path} ...")


if __name__ == "__main__":
    if args.dump_json_only:
        dump_filtered_json()
    else:
        # Use energy threshold to filter out the high quality data:
        # save the high quality data
        # save the visualization
        # save the windows info as a json file
        main()


    
