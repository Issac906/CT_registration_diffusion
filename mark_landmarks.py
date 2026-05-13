import os
import glob
import numpy as np
import nibabel as nib
from collections import defaultdict, deque


# -----------------------------
# (A) 连通块：把“相邻点”归为同一组件（可选）
# -----------------------------
def connected_components_points(pts_vox_int, connectivity=26):
    """
    pts_vox_int: Nx3 int (voxel indices)
    connectivity: 6/18/26 # 3D 邻域定义
    returns: comp_id (N,), comps (list of lists of point indices)
    """
    pts_vox_int = np.asarray(pts_vox_int, dtype=int) # 把输入强制转为 int 的 numpy 数组
    N = pts_vox_int.shape[0] # N 是点的数量

    # 处理同一 voxel 里有重复点
    # 目的：后面做连通块时只在唯一 voxel 集合上跑，最后再映射回原始点。
    vox2indices = defaultdict(list) # 键是voxel坐标(i,j,k)
    for idx, v in enumerate(map(tuple, pts_vox_int)):
        vox2indices[v].append(idx)

    unique_vox = list(vox2indices.keys())
    uv_index = {v: i for i, v in enumerate(unique_vox)}
    uv_set = set(unique_vox)
    M = len(unique_vox)

    # 构造邻居偏移 (step=1)
    deltas = []
    for dx in (-1, 0, 1): # 枚举3×3×3的相对偏移
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0: # 跳过(0,0,0)
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz) # 通过曼哈顿距离筛选邻居
                if connectivity == 6 and manhattan == 1:
                    deltas.append((dx, dy, dz))
                elif connectivity == 18 and manhattan in (1, 2):
                    deltas.append((dx, dy, dz))
                elif connectivity == 26:
                    deltas.append((dx, dy, dz))

    # BFS 找连通组件
    visited = np.zeros(M, dtype=bool) # 标记每个唯一 voxel 是否被访问过
    comps_unique = [] # 存每个组件里的 voxel 坐标

    for i, v in enumerate(unique_vox):
        if visited[i]: # 遍历所有 voxel；如果已经属于某个组件就跳过。
            continue
        q = deque([v]) # 对一个新组件：队列 q 初始化为当前 voxel（BFS 起点）
        visited[i] = True # 标记 visited
        comp = [] # comp 记录该组件内 voxel 列表
        while q: # BFS 主循环：弹出一个 voxel，加入当前组件。
            cx, cy, cz = q.popleft()
            comp.append((cx, cy, cz))
            for dx, dy, dz in deltas: # 扫描该 voxel 的所有邻居偏移
                nb = (cx + dx, cy + dy, cz + dz) # nb 是邻居 voxel 坐标
                if nb in uv_set: # 若这个邻居也在点集合里（nb in uv_set），且未访问，就加入队列并标记 visited
                    j = uv_index[nb]
                    if not visited[j]:
                        visited[j] = True
                        q.append(nb)
        comps_unique.append(comp) # 一个组件 BFS 完成，把它记录下来

    # 从“唯一 voxel”映射回“原始点索引”
    comps = []
    for comp_vox in comps_unique: # comp_vox 是组件里的 voxel 坐标列表。
        idxs = []
        for v in comp_vox:
            idxs.extend(vox2indices[v]) # 每个 voxel 可能对应多个原始点索引（重复点），用 extend 全部加入。
        comps.append(sorted(idxs)) # 排序后存入 comps：每个组件 → 原始点索引列表。

    comp_id = np.full(N, -1, dtype=int) # 长度 N，每个点属于哪个组件（默认 -1）
    for cid, idxs in enumerate(comps): # 遍历每个组件 cid，把组件内点的 comp_id 设为组件编号
        for idx in idxs:
            comp_id[idx] = cid
    
    # comp_id：每个点的组件 id
    # comps：每个组件里有哪些点索引。
    return comp_id, comps

# 给每个连通组件生成一个“基底 label”
# 目的：让不同组件之间的数值间隔很大，便于调试/可视化，不容易互相重叠
def make_large_separated_labels(num_components, start=1000, step=1000, dtype=np.int32):
    return (start + step * np.arange(num_components, dtype=np.int64)).astype(dtype)


# -----------------------------
# (B) IO / 基础函数
# -----------------------------
# 读 NIfTI
def load_volume(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return img, data

# 读点文件 txt
def read_points_txt(path):
    pts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(pts, dtype=np.float64)

# 按参考图保存 NIfTI
def save_like(ref_img, data, out_path, dtype=None):
    if dtype is not None:
        data = data.astype(dtype)
    out_img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
    nib.save(out_img, out_path)


# -----------------------------
# (C) 关键融合：每个“白点小块”内部体素赋不同值
# -----------------------------
def points_to_unique_voxel_values_mask(
    shape, 
    points_ijk, 
    radius=1,
    comp_id=None,
    comp_base_labels=None,
    base_start=3000,
    marker_stride=None,
    include_component_base=True,
    dtype=np.int32,
):
    """
    生成一个 overlay mask（不要改原CT），其中：
      - 每个点对应一个 (2r+1)^3 小块
      - 小块内每个体素值都不同（由 t=0..block_size-1）
      - 若 include_component_base=True 且提供 comp_id/comp_base_labels：
            每个体素值 = comp_base_labels[comp_id[n]] + (n_in_comp * marker_stride) + t
        否则：
            每个体素值 = base_start + (n * marker_stride) + t

    参数：
      marker_stride: 每个 marker 预留数值跨度；默认自动 = block_size + 1
      include_component_base: True 时会把组件的大间隔label作为更高层基值（更好调试）
    """

    # 初始化 mask 和点坐标
    mask = np.zeros(shape, dtype=dtype)
    pts = np.rint(points_ijk).astype(int)

    # 计算 block 尺寸与 stride
    k = 2 * radius + 1
    block_size = k * k * k
    if marker_stride is None:
        marker_stride = block_size + 1

    # 若要按组件编码，先为每个组件内的点编号（0..m-1）
    idx_in_comp = None
    if include_component_base and (comp_id is not None) and (comp_base_labels is not None):
        comp_id = np.asarray(comp_id, dtype=int)
        idx_in_comp = np.zeros(len(comp_id), dtype=int)
        counters = defaultdict(int)
        for n in range(len(comp_id)):
            cid = int(comp_id[n])
            idx_in_comp[n] = counters[cid]
            counters[cid] += 1
    
    # 遍历每个点，写入它的块
    X, Y, Z = shape
    for n, (i, j, k0) in enumerate(pts):
        if i < 0 or j < 0 or k0 < 0 or i >= X or j >= Y or k0 >= Z: # 点在体积外就跳过（安全检查）
            continue

        # 计算该点块的 base 值
        if include_component_base and (idx_in_comp is not None):
            cid = int(comp_id[n])
            base = int(comp_base_labels[cid]) + int(idx_in_comp[n]) * int(marker_stride)
        else:
            base = int(base_start) + int(n) * int(marker_stride)

        # 在 (2r+1)^3 小块内填值（base+t）
        t = 0
        for dz in range(-radius, radius + 1):
            kk = k0 + dz
            if kk < 0 or kk >= Z:
                t += (2 * radius + 1) * (2 * radius + 1)
                continue
            for dy in range(-radius, radius + 1):
                jj = j + dy
                if jj < 0 or jj >= Y:
                    t += (2 * radius + 1)
                    continue
                for dx in range(-radius, radius + 1):
                    ii = i + dx
                    val = base + t
                    t += 1
                    if ii < 0 or ii >= X:
                        continue
                    mask[ii, jj, kk] = val

    return mask


# verify_unique_inside_blocks函数用于抽样检查：每个点的小块里是否每个 voxel 都被赋值且都唯一（无 0、无重复）
def verify_unique_inside_blocks(overlay_mask, points_ijk, radius=1, samples=20, seed=0):
    pts = np.rint(points_ijk).astype(int)
    X, Y, Z = overlay_mask.shape
    k = 2 * radius + 1
    full = k**3

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(pts))
    if len(idxs) > samples:
        idxs = rng.choice(idxs, size=samples, replace=False)

    bad = []
    for n in idxs:
        i, j, k0 = pts[n]
        i0, i1 = max(0, i - radius), min(X, i + radius + 1)
        j0, j1 = max(0, j - radius), min(Y, j + radius + 1)
        k0a, k1a = max(0, k0 - radius), min(Z, k0 + radius + 1)

        block = overlay_mask[i0:i1, j0:j1, k0a:k1a]
        vals = block[block != 0].ravel()
        uniq = np.unique(vals)

        expected = (i1 - i0) * (j1 - j0) * (k1a - k0a)

        if len(vals) != expected:
            bad.append((n, "has_zeros_inside_block", expected, len(vals)))
        elif uniq.size != expected:
            bad.append((n, "duplicate_values", expected, uniq.size))

    print(f"checked {len(idxs)} points (radius={radius}), full_block={full}")
    if not bad:
        print("OK: all checked blocks have unique values per voxel.")
    else:
        print("FAILED examples (point_index, reason, expected, got):")
        for item in bad[:10]:
            print(item)
    return bad


# -----------------------------
# (D) CT 路径匹配：兼容三种命名（_s / -ssm / 无后缀）
# -----------------------------
def find_ct_path(ct_root, case_idx, ph):
    case_dir = os.path.join(ct_root, f"case{case_idx}")
    p = os.path.join(case_dir, f"case{case_idx}_T{ph}.nii.gz")
    if os.path.exists(p):
        return p
    hits = sorted(glob.glob(os.path.join(case_dir, f"*T{ph}*.nii.gz")))
    return hits[0] if hits else None

# 这里用来找点文件 txt，兼容数据里“Pack/Deploy、大小写、不同命名”。
def find_pts_path_global(root, case_idx, ph):
    # 先精确尝试（兼容大小写与 Pack/Deploy）
    candidates = [
        os.path.join(root, f"Case{case_idx}Pack", "extremePhases", f"case{case_idx}_dirLab300_T{ph}_xyz.txt"),
        os.path.join(root, f"Case{case_idx}Pack", "ExtremePhases", f"case{case_idx}_dirLab300_T{ph}_xyz.txt"),
        os.path.join(root, f"Case{case_idx}Deploy", "extremePhases", f"case{case_idx}_dirLab300_T{ph}_xyz.txt"),
        os.path.join(root, f"Case{case_idx}Deploy", "ExtremePhases", f"case{case_idx}_dirLab300_T{ph}_xyz.txt"),
        os.path.join(root, f"Case{case_idx}Pack", "ExtremePhases", f"Case{case_idx}_300_T{ph}_xyz.txt"),
        os.path.join(root, f"Case{case_idx}Pack", "extremePhases", f"Case{case_idx}_300_T{ph}_xyz.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # 兜底：在 ROOT 下递归搜（兼容各种子目录名）
    hits = sorted(glob.glob(os.path.join(root, "**", f"*case{case_idx}*T{ph}*xyz*.txt"), recursive=True))
    if not hits:
        hits = sorted(glob.glob(os.path.join(root, "**", f"*Case{case_idx}*T{ph}*xyz*.txt"), recursive=True))
    return hits[0] if hits else None
# -----------------------------
# (E) main
# -----------------------------
if __name__ == "__main__":
    CT_ROOT  = "/host/d/data/4DCT/DIR_LAB_original/nii.gz"
    PTS_ROOT = "/host/d/data/4DCT/DIR_LAB_original/img"
    OUT_DIR  = os.path.join(CT_ROOT, "markers_out_all")
    os.makedirs(OUT_DIR, exist_ok=True)

    phases = ["00", "50"]
    radius = 1
    marker_stride = 200  # 每个点预留 200 个数值空间，避免不同点的块编号区间重叠

    for case_idx in range(1, 11): # case1 到 case10
        case_name = f"Case{case_idx}"

        # 每个 Case 一个输出子目录
        case_out_dir = os.path.join(OUT_DIR, case_name)
        os.makedirs(case_out_dir, exist_ok=True)

        # 找 CT 文件路径、点文件路径
        for ph in phases:
            vol_path = find_ct_path(CT_ROOT, case_idx, ph) 
            pts_path = find_pts_path_global(PTS_ROOT, case_idx, ph)

            # 找不到就跳过，并打印诊断信息
            if (vol_path is None) or (pts_path is None): 
                print("[SKIP missing]", case_name, "T" + ph)
                print("  vol:", vol_path, "exists:", (vol_path is not None and os.path.exists(vol_path)))
                print("  pts:", pts_path, "exists:", (pts_path is not None and os.path.exists(pts_path)))
                continue

            # 处理一个 (case, phase)
            print("\nProcessing", case_name, "T" + ph)
            img, vol = load_volume(vol_path)
            pts_ijk = read_points_txt(pts_path) 

            # 对每个 (case, phase) 独立生成编码（保证该相位内唯一）
            # 把点四舍五入为整数 voxel，然后按 26 邻域做连通组件
            comp_id, comps = connected_components_points(np.rint(pts_ijk).astype(int), connectivity=26)
            # 每个组件分配一个基底（3000、5000、7000...）
            comp_base = make_large_separated_labels(len(comps), start=3000, step=2000, dtype=np.int32)

            # 生成 overlay mask：背景是 0；每个点附近 3×3×3 的体素被编码为不同整数值
            overlay_mask = points_to_unique_voxel_values_mask(
                shape=vol.shape,
                points_ijk=pts_ijk,
                radius=radius,
                comp_id=comp_id,
                comp_base_labels=comp_base,
                base_start=3000,
                marker_stride=marker_stride,
                include_component_base=True,
                dtype=np.int32,
            )

            # 把 mask 写入 CT（覆盖原体素）
            out_ct = vol.astype(np.int64, copy=True)
            m = overlay_mask != 0
            out_ct[m] = overlay_mask[m].astype(out_ct.dtype, copy=False)

            # 保存输出
            out_path = os.path.join(case_out_dir, f"{case_name}_T{ph}_ct_with_markers.nii.gz")
            save_like(img, out_ct, out_path, dtype=np.int64)
            print("saved:", out_path)

            # 随机抽样验证：每个点的小块内体素值是否全唯一且无 0
            bad = verify_unique_inside_blocks(
                overlay_mask=overlay_mask,
                points_ijk=pts_ijk,
                radius=radius,
                samples=50, 
                seed=0,
            )
            if bad:
                raise RuntimeError(f"{case_name} T{ph} verify failed, examples: {bad[:3]}")