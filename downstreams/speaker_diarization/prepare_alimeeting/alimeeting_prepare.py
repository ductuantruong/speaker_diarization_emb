import numpy as np
import os
import csv

import sys
from diarization import load_rttm, rttm2annotation, exclude_overlaping


def prepare_alimeeting(rttm_dir, save_folder, wav_dir, seg_dur, csv_file, sampling_rate=16000):

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]
    entry = []
    my_sep = "--"

    for rttm_file in os.listdir(rttm_dir):
        rttm_full_path = os.path.join(rttm_dir, rttm_file)
        rttm_content = load_rttm(rttm_full_path)
        annot_rttm = rttm2annotation(rttm_content)

        utt_id = rttm_file.split(".rttm")[0]
        wav_full_path = os.path.join(wav_dir, utt_id + ".wav")

        # skip overlapping speech segments
        annot_nooverlap = exclude_overlaping(annot_rttm)

        min_seg_length = 3.0

        for segment, track, label in annot_nooverlap.itertracks(yield_label=True):
            dur = segment.end - segment.start

            # skip short segments
            if dur < min_seg_length:
                continue

            spk_id = "_".join(label.split("_")[-2:])
            audio_id = my_sep.join([spk_id, utt_id])

            num_chunks =int( dur // seg_dur)
            for idx_chunk in range(num_chunks):

                start_sample = np.round((segment.start + idx_chunk * seg_dur) * sampling_rate)
                stop_sample = np.round((segment.start + idx_chunk * seg_dur + seg_dur) * sampling_rate)

                audio_id = audio_id + "_" + str(start_sample) + "_" + str(stop_sample)

                csv_line = [
                    audio_id,
                    str(dur),
                    wav_full_path,
                    start_sample,
                    stop_sample,
                    spk_id,
                ]
                entry.append(csv_line)

    csv_output = csv_output + entry

    csv_file = os.path.join(save_folder, csv_file)

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Final prints
    msg = "\t%s successfully created!" % (csv_file)
    print(msg)
