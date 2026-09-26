import sys
from pathlib import Path
import numpy as np
import soundfile as sf

def audit_file(path: Path):
    if not path.exists():
        return None
    data, sr = sf.read(path)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    
    # Chunks RMS
    chunk_size = sr // 2
    chunks = [np.sqrt(np.mean(mono[i:i+chunk_size]**2)) for i in range(0, len(mono), chunk_size) if len(mono[i:i+chunk_size]) == chunk_size]
    min_rms = float(min(chunks)) if chunks else 0.0
    max_rms = float(max(chunks)) if chunks else 0.0
    
    # Spectral analysis
    fft = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1/sr)
    total_energy = np.sum(fft**2) + 1e-12
    
    dc_sub_energy = np.sum(fft[freqs < 60]**2) / total_energy
    mid_energy = np.sum(fft[(freqs >= 300) & (freqs <= 3000)]**2) / total_energy
    hf_energy = np.sum(fft[freqs > 10000]**2) / total_energy
    hf_extreme = np.sum(fft[freqs > 14000]**2) / total_energy
    centroid = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-12))
    
    # Spectral flatness (Wiener entropy) in mid band 300-3000 Hz (detects peak resonance/honk)
    mid_mask = (freqs >= 300) & (freqs <= 3000)
    mid_mag = fft[mid_mask] + 1e-12
    geom_mean = np.exp(np.mean(np.log(mid_mag)))
    arith_mean = np.mean(mid_mag)
    flatness = float(geom_mean / arith_mean)
    
    # Peak-to-average resonance in mids (dB)
    peak_to_avg_mid_db = float(20 * np.log10((np.max(mid_mag) + 1e-9) / (arith_mean + 1e-9)))

    return {
        "path": path.name,
        "peak": peak,
        "rms": rms,
        "min_rms": min_rms,
        "sub_ratio": float(dc_sub_energy),
        "mid_ratio": float(mid_energy),
        "mid_flatness": flatness,
        "mid_peak_db": peak_to_avg_mid_db,
        "hf_10k_ratio": float(hf_energy),
        "hf_14k_ratio": float(hf_extreme),
        "centroid": centroid,
    }

def main():
    tracks = ["piano", "funk", "ambient"]
    v1_dir = Path("output/sample-expertise-pilot/universal-models/final-audios-30s")
    
    print(f"{'Track':<25} | {'RMS':<7} | {'Mid(300-3k)':<11} | {'Mid Reso':<8} | {'HF>10k':<8} | {'Centroid':<8}")
    print("-" * 75)
    for t in tracks:
        for ver, prefix in [("v1", "bonsai_"), ("v2", "bonsai_v2_"), ("v3", "bonsai_v3_")]:
            p = v1_dir / f"{prefix}{t}_30s.wav"
            m = audit_file(p)
            if m:
                print(f"{m['path']:<25} | {m['rms']:<7.4f} | {m['mid_ratio']*100:<10.1f}% | {m['mid_peak_db']:<7.1f}dB | {m['hf_10k_ratio']*100:<7.2f}% | {m['centroid']:<8.1f}Hz")
        print("-" * 75)

if __name__ == '__main__':
    main()
