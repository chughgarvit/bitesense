import math
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import pandas as pd

def load_imu_csv(file_path, delimiter=",", header=0):
    """
    Load IMU data from a CSV file.
    
    Assumes the CSV has columns for Ax, Ay, Az, Gx, Gy, Gz in order.
    
    Args:
      file_path: Path to the CSV file.
      delimiter: CSV delimiter (default is comma).
      header: Row number to use as column names.
    
    Returns:
      A numpy array of shape (num_samples, 6)
    """
    df = pd.read_csv(file_path, delimiter=delimiter, header=header)
    # If the CSV has extra columns, only the first 6 are used.
    return df.iloc[:, :6].values

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Designs a Butterworth bandpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.1, highcut=2.5, fs=44.0, order=4):
    """
    Apply a Butterworth bandpass filter to the data.
    data: numpy array of shape (num_samples, num_channels)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = signal.filtfilt(b, a, data, axis=0)
    return filtered

def detect_chewing_episodes(data, fs=44.0, threshold_multiplier=1.0, buffer_time=0.5, extension_time=15.0):
    """
    Detect eating episodes based on the gyroscopic energy signal.
    
    Args:
      data: numpy array of shape (num_samples, 6) (filtered data).
      fs: Sampling frequency.
      threshold_multiplier: Multiplier for threshold (threshold = mean + multiplier*std).
      buffer_time: Buffer duration in seconds to extend detected peaks.
      extension_time: If a new chewing event occurs within this time (seconds), merge episodes.
    
    Returns:
      A list of tuples (start_index, end_index) for each detected eating episode.
    """
    # Compute gyroscopic energy signal from Gx, Gy, Gz (columns 3,4,5).
    gyro = data[:, 3:6]
    energy = np.sqrt(np.sum(gyro**2, axis=1))
    
    # Set a dynamic threshold.
    threshold = np.mean(energy) + threshold_multiplier * np.std(energy)
    
    # Find indices where energy exceeds the threshold.
    above_thresh = np.where(energy > threshold)[0]
    if len(above_thresh) == 0:
        return []
    
    # Group contiguous indices, allowing for gaps shorter than buffer_time.
    buffer_samples = int(buffer_time * fs)
    episodes = []
    start = above_thresh[0]
    end = above_thresh[0]
    for idx in above_thresh[1:]:
        if idx - end <= buffer_samples:
            # Extend current group.
            end = idx
        else:
            # Save current group (extend by buffer).
            episodes.append((max(start - buffer_samples, 0), min(end + buffer_samples, len(data))))
            start = idx
            end = idx
    episodes.append((max(start - buffer_samples, 0), min(end + buffer_samples, len(data))))
    
    # Merge episodes that are close together (within extension_time seconds).
    extension_samples = int(extension_time * fs)
    merged_episodes = []
    current_start, current_end = episodes[0]
    for (start, end) in episodes[1:]:
        if start - current_end <= extension_samples:
            # Merge episodes.
            current_end = end
        else:
            merged_episodes.append((current_start, current_end))
            current_start, current_end = start, end
    merged_episodes.append((current_start, current_end))
    
    return merged_episodes

def segment_windows(data, window_size, step_size):
    """
    Segment time-series data into overlapping windows.
    
    data: numpy array of shape (num_samples, num_channels)
    Returns: numpy array of shape (num_windows, window_size, num_channels)
    """
    segments = []
    num_samples = data.shape[0]
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        segments.append(data[start:end, :])
    return np.array(segments)

def extract_features(window, fs=44.0):
    """
    Extract features from a single window of IMU data.
    
    Input:
      window: numpy array of shape (window_size, 6) representing [Ax, Ay, Az, Gx, Gy, Gz]
    
    Returns:
      A feature vector (numpy array) concatenating feature sets f1 to f5.
    """
    features = []
    num_channels = window.shape[1]
    
    # --- f1: Basic Statistical Features for Each Channel ---
    # For each channel, extract mean, std, median, max, min, number of peaks,
    # skewness, kurtosis, energy, RMS, peak frequency, and spectral entropy.
    for ch in range(num_channels):
        sig = window[:, ch]
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        median_val = np.median(sig)
        max_val = np.max(sig)
        min_val = np.min(sig)
        peaks, _ = signal.find_peaks(sig)
        num_peaks = len(peaks)
        skew_val = stats.skew(sig)
        kurt_val = stats.kurtosis(sig)
        energy_val = np.sum(sig ** 2) / len(sig)
        rms_val = np.sqrt(np.mean(sig ** 2))
        fft_vals = np.fft.rfft(sig)
        fft_freq = np.fft.rfftfreq(len(sig), d=1/fs)
        peak_freq = fft_freq[np.argmax(np.abs(fft_vals))]
        psd = np.abs(fft_vals) ** 2
        psd_norm = psd / (np.sum(psd) + 1e-12)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        
        features.extend([mean_val, std_val, median_val, max_val, min_val, num_peaks,
                         skew_val, kurt_val, energy_val, rms_val, peak_freq, spectral_entropy])
    # f1: 12 features per channel × 6 channels = 72 features.
    
    # --- f2: Jaw Motion-Induced Patterns (Gyroscope channels 3-5) ---
    gyro = window[:, 3:6]
    mean_gyro = np.mean(gyro, axis=0)                # 3 features
    cum_disp = np.sum(np.abs(np.diff(gyro, axis=0)), axis=0)  # 3 features
    gyro_deriv = np.diff(gyro, axis=0)
    rot_acc = np.std(gyro_deriv, axis=0)              # 3 features
    features.extend(mean_gyro.tolist() + cum_disp.tolist() + rot_acc.tolist())
    # f2 adds 9 features.
    
    # --- f3: Texture Features (Accelerometer channels 0-2) ---
    accel = window[:, 0:3]
    accel_mag = np.linalg.norm(accel, axis=1)
    peak_chewing_force = np.max(accel_mag)
    force_rate = np.mean(np.abs(np.diff(accel_mag)))
    threshold = np.mean(accel_mag) + np.std(accel_mag)
    duration_high_intensity = np.sum(accel_mag > threshold) / fs  # seconds
    features.extend([peak_chewing_force, force_rate, duration_high_intensity])
    # f3 adds 3 features.
    
    # --- f4: Nutritional Value Features ---
    chewing_intensity = np.sqrt(np.mean(accel_mag ** 2))  # RMS
    fft_vals_acc = np.fft.rfft(accel_mag)
    fft_freq_acc = np.fft.rfftfreq(len(accel_mag), d=1/fs)
    dominant_freq = fft_freq_acc[np.argmax(np.abs(fft_vals_acc))]
    gyro_energy = np.sqrt(np.sum(gyro ** 2, axis=1))
    resistance_mean = np.mean(gyro_energy)
    resistance_var = np.var(gyro_energy)
    features.extend([chewing_intensity, dominant_freq, resistance_mean, resistance_var])
    # f4 adds 4 features.
    
    # --- f5: Cooking Method Features ---
    if len(gyro_energy) > 1:
        autocorr = np.correlate(gyro_energy - np.mean(gyro_energy), 
                                gyro_energy - np.mean(gyro_energy), mode='full')
        autocorr = autocorr[autocorr.size // 2 + 1:]
        signal_regularity = np.max(autocorr) / (np.var(gyro_energy) * len(gyro_energy) + 1e-12)
    else:
        signal_regularity = 0
    smoothness = np.mean(np.std(gyro, axis=0))
    features.extend([signal_regularity, smoothness])
    # f5 adds 2 features.
    
    # Total feature vector length: 72 + 9 + 3 + 4 + 2 = 90.
    return np.array(features)

def process_imu_data(raw_data, fs=44.0, window_duration=3.0, step_duration=1.5,
                     use_threshold=True, threshold_multiplier=1.0, buffer_time=0.5,
                     extension_time=15.0):
    """
    Process raw IMU data: filtering, threshold-based chewing episode detection,
    window segmentation, and feature extraction.
    
    Args:
      raw_data: numpy array of shape (num_samples, 6)
      window_duration: Window length in seconds.
      step_duration: Sliding window step in seconds.
      use_threshold: If True, use threshold-based analysis to extract only eating episodes.
      threshold_multiplier: Multiplier for the dynamic threshold.
      buffer_time: Buffer duration (seconds) around detected peaks.
      extension_time: Time (seconds) within which nearby episodes are merged.
    
    Returns:
      Numpy array of shape (num_windows, num_features)
    """
    # 1. Filter the data.
    filtered_data = bandpass_filter(raw_data, lowcut=0.1, highcut=2.5, fs=fs, order=4)
    
    # 2. If enabled, detect chewing episodes and concatenate them.
    if use_threshold:
        episodes = detect_chewing_episodes(filtered_data, fs=fs,
                                           threshold_multiplier=threshold_multiplier,
                                           buffer_time=buffer_time,
                                           extension_time=extension_time)
        if len(episodes) > 0:
            episode_data = [filtered_data[start:end, :] for (start, end) in episodes]
            filtered_data = np.concatenate(episode_data, axis=0)
        else:
            print("No chewing episodes detected; using entire data.")
    
    # 3. Segment the (possibly reduced) data into windows.
    window_size = int(window_duration * fs)
    step_size = int(step_duration * fs)
    windows = segment_windows(filtered_data, window_size, step_size)
    
    # 4. Extract features from each window.
    features_list = [extract_features(window, fs=fs) for window in windows]
    return np.array(features_list)

if __name__ == "__main__":
    # Demonstration: load data from CSV if available.
    import os
    fs = 44.0
    csv_path = "data/raw/imu_data.csv"  # Adjust this path as needed.
    if os.path.exists(csv_path):
        raw_data = load_imu_csv(csv_path)
        print("Loaded CSV data with shape:", raw_data.shape)
    else:
        print("CSV file not found")
    
    # Process the raw data with threshold-based chewing detection.
    features_seq = process_imu_data(raw_data, fs=fs, window_duration=3.0, step_duration=1.5,
                                    use_threshold=True, threshold_multiplier=1.0,
                                    buffer_time=0.5, extension_time=15.0)
    print("Extracted feature sequence shape:", features_seq.shape)
