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
    hf_energy = np.sum(fft[freqs > 14000]**2) / total_energy
    mid_energy = np.sum(fft[(freqs >= 250) & (freqs <= 4000)]**2) / total_energy
    centroid = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-12))
    
    return {
        "path": path.name,
        "peak": peak,
        "rms": rms,
        "min_rms": min_rms,
        "max_rms": max_rms,
        "dc_sub_ratio": float(dc_sub_energy),
        "hf_ratio": float(hf_energy),
        "mid_ratio": float(mid_energy),
        "centroid": centroid,
    }

def main():
    tracks = ["piano", "funk", "ambient"]
    v1_dir = Path("output/sample-expertise-pilot/universal-models/final-audios-30s")
    
    print(f"{'Track':<25} | {'Peak':<6} | {'RMS':<7} | {'MinRMS':<8} | {'Sub<60Hz':<9} | {'HF>14k':<9} | {'Centroid':<8}")
    print("-" * 85)
    for t in tracks:
        v1_p = v1_dir / f"bonsai_{t}_30s.wav"
        v2_p = v1_dir / f"bonsai_v2_{t}_30s.wav"
        
        m1 = audit_file(v1_p)
        if m1:
            print(f"{m1['path']:<25} | {m1['peak']:<6.3f} | {m1['rms']:<7.4f} | {m1['min_rms']:<8.5f} | {m1['dc_sub_ratio']*100:<8.2f}% | {m1['hf_ratio']*100:<8.3f}% | {m1['centroid']:<8.1f}Hz")
        m2 = audit_file(v2_p)
        if m2:
            print(f"{m2['path']:<25} | {m2['peak']:<6.3f} | {m2['rms']:<7.4f} | {m2['min_rms']:<8.5f} | {m2['dc_sub_ratio']*100:<8.2f}% | {m2['hf_ratio']*100:<8.3f}% | {m2['centroid']:<8.1f}Hz")
        print("-" * 85)

if __name__ == '__main__':
    main()
