import sys
import os
from pathlib import Path
import numpy as np

MLX_RUNTIME_ROOT = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
SCRIPTS_DIR = MLX_RUNTIME_ROOT / "scripts"
sys.path = [str(MLX_RUNTIME_ROOT), str(SCRIPTS_DIR)] + [p for p in sys.path if "musicgen" not in p and "abelton" not in p]

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from models.defs import dit_mlx_medium
from models.defs.latent_dataset import PreEncodedLatentDataset

def main():
    model_path = Path("/Users/guillaumegaillard/Documents/perso/abelton/output/sample-expertise-pilot/sftberlin/quantized-models/dit_medium_bonsai_ternary_int2_group64.npz")
    master_path = Path("/Users/guillaumegaillard/Documents/perso/abelton/output/sample-expertise-pilot/sftberlin/quantized-models/dit_medium_bonsai_distilled_master.npz")
    latents_dir = Path("output/sample-expertise-pilot/sftberlin/sft-training-spectral-body-safe-12s/latents-12s")
    group_size = 64
    T_lat = 128

    print("[1] Initializing teacher and quant_dit from master...")
    teacher = dit_mlx_medium.DiT(T_lat=T_lat)
    teacher.load_weights(str(MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz"), strict=False)

    quant_dit = dit_mlx_medium.DiT(T_lat=T_lat)
    quant_dit.load_weights(str(master_path), strict=False)

    def predicate(path: str, layer: nn.Module) -> bool:
        if not isinstance(layer, nn.Linear):
            return False
        return layer.weight.shape[-1] % group_size == 0

    nn.quantize(quant_dit, bits=2, group_size=group_size, mode="affine", class_predicate=predicate)

    # Pack optimal least-squares ternary into transformer layers
    raw_master = mx.load(str(master_path))
    quant_params = dict(tree_flatten(quant_dit.parameters()))

    print("[2] Packing bit-perfect optimal ternary layers...")
    for k, v in raw_master.items():
        if "transformer.layers." in k and k.endswith(".weight") and v.ndim == 2 and v.shape[1] % group_size == 0:
            w = v.astype(mx.float32)
            out_d, in_d = w.shape
            g = w.reshape(out_d, -1, group_size)
            base_scale = mx.mean(mx.abs(g), axis=-1, keepdims=True)

            best_mse = mx.full((out_d, in_d // group_size, 1), 1e9)
            best_q = mx.zeros_like(g)
            best_s = mx.zeros((out_d, in_d // group_size, 1))

            for factor in [0.55, 0.65, 0.75, 0.85]:
                q = mx.where(g > factor * base_scale, 1.0, mx.where(g < -factor * base_scale, -1.0, 0.0))
                s = mx.sum(g * q, axis=-1, keepdims=True) / (mx.sum(q**2, axis=-1, keepdims=True) + 1e-5)
                rec = q * s
                mse = mx.mean((g - rec)**2, axis=-1, keepdims=True)
                better = mse < best_mse
                best_mse = mx.where(better, mse, best_mse)
                best_q = mx.where(better, q, best_q)
                best_s = mx.where(better, s, best_s)

            q_code = mx.where(best_q == 1.0, 0, mx.where(best_q == 0.0, 1, 2)).astype(mx.uint32).reshape(out_d, in_d)
            np_codes = np.array(q_code)
            codes_reshaped = np_codes.reshape(out_d, in_d // 16, 16)
            packed = np.zeros((out_d, in_d // 16), dtype=np.uint32)
            for i in range(16):
                packed |= (codes_reshaped[:, :, i].astype(np.uint32) << (2 * i))

            s_final = best_s.astype(mx.float16)
            scales = -s_final.reshape(out_d, in_d // group_size)
            biases = s_final.reshape(out_d, in_d // group_size)

            prefix = k[:-7]
            quant_params[prefix + ".weight"] = mx.array(packed)
            quant_params[prefix + ".scales"] = scales
            quant_params[prefix + ".biases"] = biases

    quant_dit.load_weights(list(quant_params.items()), strict=True)
    mx.eval(quant_dit.parameters())

    print("[3] Setting up calibration batches...")
    dataset = PreEncodedLatentDataset(str(latents_dir), T_lat, random_crop=False, seed=42)
    timesteps = [1.0, 0.8, 0.5, 0.2]
    
    calib_data = []
    for i, batch in enumerate(dataset):
        if i >= 16:
            break
        raw_lat = batch["latents"]
        if raw_lat.ndim == 2:
            raw_lat = raw_lat[None, :, :T_lat]
        else:
            raw_lat = raw_lat[:, :, :T_lat]
        lat = mx.array(raw_lat)
        for t_val in timesteps:
            noise = mx.random.normal(lat.shape, dtype=lat.dtype)
            noised = lat * (1.0 - t_val) + noise * t_val
            calib_data.append((noised, t_val))
            
    print(f"Collected {len(calib_data)} calibration states.")

    dummy_cross = mx.random.normal((1, 257, 768), dtype=mx.float16)
    dummy_global = mx.random.normal((1, 768), dtype=mx.float16)

    c = teacher.to_cond_embed[0](dummy_cross)
    c = nn.silu(c)
    context = teacher.to_cond_embed[2](c)

    g = teacher.to_global_embed[0](dummy_global)
    g = nn.silu(g)
    global_pre = teacher.to_global_embed[2](g)

    local = mx.zeros((1, T_lat, 257))

    print("\n[4] Closed-form block gate calibration...")
    k_stars = []
    
    teach_h_list = []
    quant_h_list = []
    
    for noised, t_val in calib_data:
        t_tensor = mx.array([t_val], dtype=mx.float16)
        tf = teacher.timestep_features(t_tensor)
        tf = teacher.to_timestep_embed[0](tf)
        tf = nn.silu(tf)
        t_embed = teacher.to_timestep_embed[2](tf)
        global_embed = global_pre + t_embed

        x_lc = noised.transpose(0, 2, 1)
        x_pp = teacher.preprocess_conv(x_lc) + x_lc

        xt = teacher.transformer.project_in(x_pp)
        xq = quant_dit.transformer.project_in(x_pp)

        mem = mx.broadcast_to(teacher.transformer.memory_tokens[None], (1, 64, 1536))
        xt = mx.concatenate([mem, xt], axis=1)
        xq = mx.concatenate([mem, xq], axis=1)

        gt = teacher.transformer.global_cond_embedder[0](global_embed)
        gt = nn.silu(gt)
        g_proj = teacher.transformer.global_cond_embedder[2](gt)
        
        teach_h_list.append((xt, g_proj))
        quant_h_list.append((xq, g_proj))

    for b in range(24):
        lt = teacher.transformer.layers[b]
        lq = quant_dit.transformer.layers[b]

        local_emb = lt.to_local_embed(local)
        pad = mx.zeros((1, 64, 1536), dtype=local_emb.dtype)
        local_padded = mx.concatenate([pad, local_emb], axis=1)

        numer = 0.0
        denom = 0.0

        for idx in range(len(calib_data)):
            xt, g_proj = teach_h_list[idx]
            xq, _ = quant_h_list[idx]

            xt_next = lt(xt, context, g_proj, local_padded)
            xq_unscaled = lq(xq, context, g_proj, local_padded)

            delta_q = xq_unscaled - xq
            y = xt_next - xq

            y32 = y.astype(mx.float32)
            dq32 = delta_q.astype(mx.float32)

            numer += float(mx.sum(y32 * dq32))
            denom += float(mx.sum(dq32**2))

        k_star = numer / (denom + 1e-6)
        k_star = max(0.5, min(1.8, k_star))
        k_stars.append(k_star)

        g_arr = mx.array(lq.to_scale_shift_gate)
        g_arr[3072:4608] = g_arr[3072:4608] * k_star
        g_arr[7680:9216] = g_arr[7680:9216] * k_star
        lq.to_scale_shift_gate = g_arr

        next_teach = []
        next_quant = []
        cos_sims = []
        for idx in range(len(calib_data)):
            xt, g_proj = teach_h_list[idx]
            xq, _ = quant_h_list[idx]

            xt_next = lt(xt, context, g_proj, local_padded)
            xq_next = lq(xq, context, g_proj, local_padded)
            next_teach.append((xt_next, g_proj))
            next_quant.append((xq_next, g_proj))
            
            c = float(mx.sum(xt_next.astype(mx.float32) * xq_next.astype(mx.float32)) / (mx.sqrt(mx.sum(xt_next.astype(mx.float32)**2)) * mx.sqrt(mx.sum(xq_next.astype(mx.float32)**2))))
            cos_sims.append(c)

        teach_h_list = next_teach
        quant_h_list = next_quant
        avg_cos = sum(cos_sims) / len(cos_sims)
        print(f"Block {b:02d}: k* = {k_star:.3f} | Cos Sim = {avg_cos:.4f}")

    print("\n[5] Calibrating project_out gain...")
    numer_p = 0.0
    denom_p = 0.0
    for idx in range(len(calib_data)):
        xt, _ = teach_h_list[idx]
        xq, _ = quant_h_list[idx]

        vt = teacher.transformer.project_out(xt[:, 64:, :])
        vq = quant_dit.transformer.project_out(xq[:, 64:, :])

        vt32 = vt.astype(mx.float32)
        vq32 = vq.astype(mx.float32)

        numer_p += float(mx.sum(vt32 * vq32))
        denom_p += float(mx.sum(vq32**2))

    k_proj = numer_p / (denom_p + 1e-6)
    print(f"project_out optimal gain: {k_proj:.4f}")

    quant_dit.transformer.project_out.scales = quant_dit.transformer.project_out.scales * k_proj
    quant_dit.transformer.project_out.biases = quant_dit.transformer.project_out.biases * k_proj

    print(f"\n[6] Atomically saving calibrated model to {model_path}...")
    final_params = dict(tree_flatten(quant_dit.parameters()))
    tmp_path = model_path.with_name("dit_medium_bonsai_calibrated.tmp.npz")
    mx.savez(str(tmp_path), **final_params)
    os.replace(str(tmp_path), str(model_path))
    print(f"Successfully calibrated and saved: {model_path} ({model_path.stat().st_size / 1024**2:.1f} MB, {len(final_params)} keys)")

if __name__ == "__main__":
    main()
