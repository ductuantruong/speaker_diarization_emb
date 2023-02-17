#!/usr/bin/python3
import os
import speechbrain as sb
from speechbrain.utils.data_utils import download_file

if __name__ == "__main__":

    rttm_dir = "../data/Train_Ali_far/rttm_groundtruth"
    save_folder = "../data"
    wav_dir = "../data/Train_Ali_far/audio_dir"
    seg_dur = 3.0
    csv_file = "train.csv"

    from alimeeting_prepare import prepare_alimeeting

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    sb.utils.distributed.run_on_main(
        prepare_alimeeting,
        kwargs={
            "rttm_dir": rttm_dir,
            "save_folder": save_folder,
            "wav_dir": wav_dir,
            "seg_dur": 3.0,
            "csv_file": csv_file,
        },
    )
    
