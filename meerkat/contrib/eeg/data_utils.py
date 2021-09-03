import h5py
import numpy as np
import os

FREQUENCY = 200


def compute_file_tuples(raw_dataset_dir, dataset_dir, split, clip_len, stride):
    """
    Args:
        dataset_dir (str): location where resampled signals are
        split (str): whether train, dev, test
        clip_len(int): length of each clip in the input eeg segments
        stride (int): how to sample clips from eeg signal

    Returns (file_name, clip_idx, seizure_label) tuples
            for the given split, clip_len, and stride
    The clip_idx indicates which clip (i.e. segment of EEG signal with clip_len seconds)
    The stride determines how to sample clips from the eeg signal
            (e.g. if stride=clip_len we have no overlapping clips)
    """

    # retrieve paths of all edf files in the dataset_dir for given split
    edf_files = []
    edf_fullfiles = []
    for path, _, files in os.walk(os.path.join(raw_dataset_dir, split)):
        for name in files:
            if ".edf" in name:
                edf_fullfiles.append(os.path.join(path, name))
                edf_files.append(name)

    resampled_files = os.listdir(dataset_dir)
    file_tuples = []

    for h5_fn in resampled_files:
        edf_fn = h5_fn.split(".h5")[0] + ".edf"
        if edf_fn not in edf_files:
            continue
        edf_fn_full = [file for file in edf_fullfiles if edf_fn in file]
        if len(edf_fn_full) != 1:
            print(f"{edf_fn} found {len(edf_fn_full)} times!")
            print(edf_fn_full)

        edf_fn_full = edf_fn_full[0]
        seizure_times = get_seizure_times(edf_fn_full.split(".edf")[0])

        h5_fn_full = os.path.join(dataset_dir, h5_fn)
        with h5py.File(h5_fn_full, "r") as hf:
            resampled_sig = hf["resampled_signal"][()]

        num_clips = (resampled_sig.shape[-1] - clip_len * FREQUENCY) // (
            stride * FREQUENCY
        ) + 1

        for i in range(num_clips):
            start_window = i * FREQUENCY * stride
            end_window = np.minimum(
                start_window + FREQUENCY * clip_len, resampled_sig.shape[-1]
            )

            is_seizure = 0
            for t in seizure_times:
                start_t = int(t[0] * FREQUENCY)
                end_t = int(t[1] * FREQUENCY)
                if not ((end_window < start_t) or (start_window > end_t)):
                    is_seizure = 1
                    break
            file_tuples.append((edf_fn, i, is_seizure))

    return file_tuples


def compute_slice_matrix(
    edf_fn,
    clip_idx,
    h5_fn=None,
    time_step_size=1,
    clip_len=60,
    stride=60,
):
    """
    Convert entire EEG sequence into clips of length clip_len
    Args:
        edf_fn: edf/eeghdf file name, full path
        channel_names: list of channel names
        clip_idx: index of current clip/sliding window, int
        h5_fn: file name of resampled signal h5 file (full path)
        time_step_size: length of each time step, in seconds, int
        clip_len: sliding window size or EEG clip length, in seconds, int
        stride: stride size, by how many seconds the sliding window moves, int
    Returns:
        eeg_clip: EEG clip
        seizure_labels: per-time-step seizure labels
        is_seizure: overall label, 1 if at least one seizure in clip
    """

    physical_clip_len = int(FREQUENCY * clip_len)

    start_window = clip_idx * FREQUENCY * stride

    with h5py.File(h5_fn, "r") as f:
        signal_array = f["resampled_signal"][()]
        resampled_freq = f["resample_freq"][()]
        assert resampled_freq == FREQUENCY

    # (num_channels, physical_clip_len)

    end_window = np.minimum(signal_array.shape[-1], start_window + physical_clip_len)
    curr_slc = signal_array[:, start_window:end_window]  # (num_channels, FREQ*clip_len)
    physical_time_step_size = int(FREQUENCY * time_step_size)

    start_time_step = 0
    time_steps = []
    while start_time_step <= curr_slc.shape[1] - physical_time_step_size:
        end_time_step = start_time_step + physical_time_step_size
        # (num_channels, physical_time_step_size)
        curr_time_step = curr_slc[:, start_time_step:end_time_step]

        time_steps.append(curr_time_step)
        start_time_step = end_time_step

    eeg_clip = np.stack(time_steps, axis=0)

    # get seizure times, take min_sz_len into account
    if ".edf" in edf_fn:
        seizure_times = get_seizure_times(edf_fn.split(".edf")[0])
    else:
        raise NotImplementedError

    # get per-time-step seizure labels
    num_time_steps = eeg_clip.shape[0]
    seizure_labels = np.zeros((num_time_steps)).astype(int)
    is_seizure = 0
    for t in seizure_times:
        start_t = int(t[0] * FREQUENCY)
        end_t = int(t[1] * FREQUENCY)
        if not ((end_window < start_t) or (start_window > end_t)):
            is_seizure = 1

            start_t_sec = int(t[0])  # start of seizure in int seconds
            end_t_sec = int(t[1])  # end of seizure in int seconds

            # shift start_t_sec and end_t_sec so that they start at current clip
            start_t_sec = np.maximum(0, start_t_sec - int(start_window / FREQUENCY))
            end_t_sec = np.minimum(clip_len, end_t_sec - int(start_window / FREQUENCY))
            # print("start_t_sec: {}; end_t_sec: {}".format(start_t_sec, end_t_sec))

            # time step size may not be 1-sec
            start_time_step = int(np.floor(start_t_sec / time_step_size))
            end_time_step = int(np.ceil(end_t_sec / time_step_size))

            seizure_labels[start_time_step:end_time_step] = 1

    return eeg_clip, seizure_labels, is_seizure


def get_seizure_times(file_name):
    """
    Args:
        file_name: file name of .edf file etc.
    Returns:
        seizure_times: list of times of seizure onset in seconds
    """
    tse_file = file_name.split(".edf")[0] + ".tse_bi"

    seizure_times = []
    with open(tse_file) as f:
        for line in f.readlines():
            if "seiz" in line:  # if seizure
                # seizure start and end time
                seizure_times.append(
                    [
                        float(line.strip().split(" ")[0]),
                        float(line.strip().split(" ")[1]),
                    ]
                )
    return seizure_times
