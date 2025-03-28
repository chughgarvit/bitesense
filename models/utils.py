import math
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import pandas as pd

def load_imu_csv(file_path, delimiter=",", header=0):
    """
    Load IMU data from a CSV file.
    Assumes the CSV has columns for Ax, Ay, Az, Gx, Gy, Gz.
    Returns a numpy array of shape (num_samples, 6).
    """
    df = pd.read_csv(file_path, delimiter=delimiter, header=header)
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
    Apply a Butterworth bandpass filter to multi-channel data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return signal.filtfilt(b, a, data, axis=0)

def detect_chewing_episodes(data, fs=44.0, threshold_multiplier=1.0,
                            buffer_time=0.5, extension_time=15.0):
    """
    Detect chewing episodes based on the gyroscopic energy signal.
    Returns a list of (start_index, end_index) tuples.
    """
    # Compute energy from gyroscope channels (Gx, Gy, Gz)
    gyro = data[:, 3:6]
    energy = np.sqrt(np.sum(gyro**2, axis=1))
    threshold = np.mean(energy) + threshold_multiplier * np.std(energy)
    
    above_thresh = np.where(energy > threshold)[0]
    if len(above_thresh) == 0:
        return []
    
    buffer_samples = int(buffer_time * fs)
    episodes = []
    start = above_thresh[0]
    end = above_thresh[0]
    for idx in above_thresh[1:]:
        if idx - end <= buffer_samples:
            end = idx
        else:
            episodes.append((max(start - buffer_samples, 0),
                             min(end + buffer_samples, len(data))))
            start = idx
            end = idx
    episodes.append((max(start - buffer_samples, 0),
                     min(end + buffer_samples, len(data))))
    
    # Merge episodes that are close together
    extension_samples = int(extension_time * fs)
    merged_episodes = []
    cur_start, cur_end = episodes[0]
    for (st, en) in episodes[1:]:
        if st - cur_end <= extension_samples:
            cur_end = en
        else:
            merged_episodes.append((cur_start, cur_end))
            cur_start, cur_end = st, en
    merged_episodes.append((cur_start, cur_end))
    
    return merged_episodes

def segment_windows(data, window_size, step_size):
    """
    Segment time-series data into overlapping windows.
    Returns an array of shape (num_windows, window_size, num_channels).
    """
    segments = []
    num_samples = data.shape[0]
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        segments.append(data[start:end, :])
    return np.array(segments)

#########################
# Feature Extraction
#########################

def extract_f1(window, fs=44.0):
    """
    f1: Extracts the basic statistical features and additional facial-muscle features.
    For each sensor channel, computes the following measures as described in the paper:
      - Basic statistical measures: mean, standard deviation, median,
        number of values above the mean, number of peaks, skewness, kurtosis.
      - Frequency-domain measures: peak frequency (from FFT), spectral entropy,
        high-frequency noise, and low-frequency drift (from the power spectral density).
      - Additional features for activated facial muscles: jerk (std of derivative),
        mean absolute value (magnitude), interquartile range, cross-correlation between successive samples, and RMS.
    Returns a 1D feature vector for the window.
    """
    features = []
    for ch in range(window.shape[1]):
        sig = window[:, ch]
        # Basic time-domain features
        mean_val = np.mean(sig)
        std_val = np.std(sig)
        median_val = np.median(sig)
        num_above = np.sum(sig > mean_val)
        peaks, _ = signal.find_peaks(sig)
        num_peaks = len(peaks)
        skew_val = stats.skew(sig)
        kurt_val = stats.kurtosis(sig)
        
        # Frequency-domain features
        fft_vals = np.fft.rfft(sig)
        fft_freq = np.fft.rfftfreq(len(sig), d=1/fs)
        peak_freq = fft_freq[np.argmax(np.abs(fft_vals))]
        psd = np.abs(fft_vals)**2
        psd_norm = psd / (np.sum(psd) + 1e-12)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        # Define thresholds for high-frequency noise and low-frequency drift
        high_freq_thresh = 0.5 * np.max(fft_freq)
        low_freq_thresh = 0.2 * np.max(fft_freq)
        high_freq_noise = np.sum(psd[fft_freq > high_freq_thresh])
        low_freq_drift = np.sum(psd[fft_freq < low_freq_thresh])
        
        # Additional facial-muscle features
        jerk = np.std(np.diff(sig))
        magnitude = np.mean(np.abs(sig))
        iqr = np.percentile(sig, 75) - np.percentile(sig, 25)
        # Compute cross-correlation between successive samples as a simple proxy
        if len(sig) > 1:
            cross_corr = np.corrcoef(sig[:-1], sig[1:])[0, 1]
        else:
            cross_corr = 0.0
        rms_val = np.sqrt(np.mean(sig**2))
        
        features.extend([
            mean_val, std_val, median_val, num_above, num_peaks,
            skew_val, kurt_val, peak_freq, spectral_entropy,
            high_freq_noise, low_freq_drift, jerk, magnitude, iqr,
            cross_corr, rms_val
        ])
    return np.array(features)

def extract_f2(window, fs=44.0):
    """
    f2: Extracts jaw motion–induced features from the gyroscope data.
    For each gyroscope channel, computes:
      - A measure of angular velocity (using the mean absolute value),
      - Cumulative angular displacement (sum of absolute differences),
      - Rotational acceleration (standard deviation of the derivative).
    Returns a feature vector for the gyroscope channels.
    """
    features = []
    # Use gyroscope channels (assumed to be columns 3 to 5)
    gyro = window[:, 3:6]
    for ch in range(gyro.shape[1]):
        sig = gyro[:, ch]
        ang_vel = np.mean(np.abs(sig))
        cum_disp = np.sum(np.abs(np.diff(sig)))
        rot_acc = np.std(np.diff(sig))
        features.extend([ang_vel, cum_disp, rot_acc])
    return np.array(features)

def extract_f3(window, fs=44.0):
    """
    f3: Extracts texture features from the accelerometer data.
    From the acceleration magnitude, computes:
      - Peak chewing force,
      - Chewing force rate (average change in force),
      - Duration of high-intensity jaw movements (fraction of samples above a threshold).
    Returns a feature vector for texture analysis.
    """
    accel = window[:, 0:3]
    accel_mag = np.linalg.norm(accel, axis=1)
    peak_force = np.max(accel_mag)
    force_rate = np.mean(np.abs(np.diff(accel_mag)))
    threshold = np.mean(accel_mag) + np.std(accel_mag)
    duration_high_intensity = np.sum(accel_mag > threshold) / len(accel_mag)
    return np.array([peak_force, force_rate, duration_high_intensity])

def extract_f4(window, fs=44.0):
    """
    f4: Extracts nutritional value features.
    Computes:
      - Chewing intensity as the RMS of the acceleration magnitude,
      - Chewing frequency as the dominant frequency from the FFT of the acceleration magnitude,
      - Resistance to chewing using the mean and variance of the gyroscope energy.
    Returns a feature vector for nutritional value.
    """
    accel = window[:, 0:3]
    accel_mag = np.linalg.norm(accel, axis=1)
    gyro = window[:, 3:6]
    gyro_energy = np.sqrt(np.sum(gyro**2, axis=1))
    
    chewing_intensity = np.sqrt(np.mean(accel_mag**2))
    fft_vals = np.fft.rfft(accel_mag)
    fft_freq = np.fft.rfftfreq(len(accel_mag), d=1/fs)
    chewing_frequency = fft_freq[np.argmax(np.abs(fft_vals))]
    resistance_mean = np.mean(gyro_energy)
    resistance_var = np.var(gyro_energy)
    
    return np.array([chewing_intensity, chewing_frequency, resistance_mean, resistance_var])

def extract_f5(window, fs=44.0):
    """
    f5: Extracts cooking method features.
    Computes:
      - Signal regularity from the autocorrelation of the gyroscope energy,
      - Jaw movement smoothness (using the standard deviation of the gyroscope signals).
    Returns a feature vector for cooking method analysis.
    """
    gyro = window[:, 3:6]
    gyro_energy = np.sqrt(np.sum(gyro**2, axis=1))
    
    # Compute autocorrelation of the gyroscope energy and derive a regularity measure
    if len(gyro_energy) > 1:
        autocorr = np.correlate(gyro_energy - np.mean(gyro_energy),
                                gyro_energy - np.mean(gyro_energy), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        signal_regularity = np.max(autocorr) / (np.var(gyro_energy) * len(gyro_energy) + 1e-12)
    else:
        signal_regularity = 0
    # Compute smoothness as the average standard deviation across gyroscope channels
    smoothness = np.mean(np.std(gyro, axis=0))
    
    return np.array([signal_regularity, smoothness])

def extract_features(window, fs=44.0):
    """
    Extracts the five feature sets (f1 to f5) from a single window.
    Returns a tuple: (f1, f2, f3, f4, f5)
    The feature calculations follow the descriptions provided in the paper.
    """
    f1 = extract_f1(window, fs)
    f2 = extract_f2(window, fs)
    f3 = extract_f3(window, fs)
    f4 = extract_f4(window, fs)
    f5 = extract_f5(window, fs)
    return (f1, f2, f3, f4, f5)

def process_imu_data(raw_data, fs=44.0, window_duration=3.0, step_duration=1.5,
                     use_threshold=True, threshold_multiplier=1.0,
                     buffer_time=0.5, extension_time=15.0):
    """
    Processes raw IMU data:
      1. Filters the data.
      2. Optionally detects and isolates chewing episodes.
      3. Segments the data into fixed-duration windows.
      4. Extracts five feature sets from each window.
    
    Returns a list of five numpy arrays corresponding to f1 through f5.
    """
    filtered_data = bandpass_filter(raw_data, lowcut=0.1, highcut=2.5, fs=fs, order=4)
    
    if use_threshold:
        episodes = detect_chewing_episodes(filtered_data, fs=fs,
                                           threshold_multiplier=threshold_multiplier,
                                           buffer_time=buffer_time,
                                           extension_time=extension_time)
        if len(episodes) > 0:
            filtered_data = np.concatenate([filtered_data[s:e, :] for (s, e) in episodes], axis=0)
        else:
            print("No chewing episodes detected; using entire data.")
    
    window_size = int(window_duration * fs)
    step_size = int(step_duration * fs)
    windows = segment_windows(filtered_data, window_size, step_size)
    
    f1_list, f2_list, f3_list, f4_list, f5_list = [], [], [], [], []
    for w in windows:
        f1, f2, f3, f4, f5 = extract_features(w, fs)
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
        f5_list.append(f5)
    
    return [np.array(f1_list), np.array(f2_list), np.array(f3_list),
            np.array(f4_list), np.array(f5_list)]

if __name__ == "__main__":
    import os
    fs = 44.0
    csv_path = "data/raw/imu_data.csv"  # Adjust path as needed.
    if os.path.exists(csv_path):
        raw_data = load_imu_csv(csv_path)
    else:
        print("CSV file not found")
    
    feature_sets = process_imu_data(raw_data, fs=fs)
    print("Extracted feature set shapes:")
    for i, fset in enumerate(feature_sets, start=1):
        print(f"f{i}:", fset.shape)
