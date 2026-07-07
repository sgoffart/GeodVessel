"""
DPC walk — exact paper implementation.

D(Ak) = -||Ak - pm||                       Eq.6  (standard)
       = -min_{pt in CLj} ||Ak - pt||      Eq.11 (type3/branch)
P(Ak) = CFC(features)  → PN via min-max   Eq.9
C(Ak) = cos(ok,o-1) + cos(ok,o-2)         Eq.8
DPC   = D + ω·PN + C  if cos(o-1,o-2)<=0.5  else  D + ω·PN   Eq.10

Filters:
  - cos(ok, o-1) < 0  → angle > 90° → discard
  - already visited   → discard
"""

import numpy as np
from itertools import product
from skimage.measure import block_reduce
from collections import deque
from tqdm import tqdm


class DPC:

    def __init__(self, volume, seg, model, omega=5.0, divergence_penalty=None):
        self.volume = volume.astype(np.float32)
        self.seg    = seg
        self.model  = model
        self.omega  = omega
        self._cache = {}
        self._off23 = self._build_offsets({2, 3})
        self._off12 = self._build_offsets({1, 2})
        print(f"[DPC] offsets Cheb2-3:{len(self._off23)}  Cheb1-2:{len(self._off12)}")

    # ── offsets ──────────────────────────────────────────────────
    @staticmethod
    def _build_offsets(cheb_set):
        offs = []
        r = max(cheb_set)
        for dx,dy,dz in product(range(-r,r+1), repeat=3):
            if max(abs(dx),abs(dy),abs(dz)) in cheb_set:
                offs.append((dx,dy,dz))
        return np.array(offs, dtype=np.int32)

    # ── features (Eq.7) ──────────────────────────────────────────
    def _patch(self, coord, size):
        x,y,z = coord; r = size//2
        s = self.volume.shape
        x0,x1 = max(0,x-r), min(s[0],x+r+1)
        y0,y1 = max(0,y-r), min(s[1],y+r+1)
        z0,z1 = max(0,z-r), min(s[2],z+r+1)
        p = self.volume[x0:x1,y0:y1,z0:z1]
        pad = [(x0-(x-r),  (x+r+1)-x1),
               (y0-(y-r),  (y+r+1)-y1),
               (z0-(z-r),  (z+r+1)-z1)]
        p = np.pad(p, pad, mode='constant')
        return p[:size,:size,:size].astype(np.float32)

    @staticmethod
    def _znorm(p):
        return (p - p.mean()) / (p.std() + 1e-6)

    def features(self, coord):
        if coord in self._cache:
            return self._cache[coord]
        # large: 15³ → maxpool → 7³
        pl = self._znorm(self._patch(coord, 15))
        pl = np.pad(pl, [(0,1)]*3, mode='edge')
        pl = block_reduce(pl, (2,2,2), np.max)[:7,:7,:7]
        # small: 7³
        ps = self._znorm(self._patch(coord, 7))
        feat = np.concatenate([pl.flatten(), ps.flatten()]).astype(np.float32)
        self._cache[coord] = feat
        return feat

    # ── neighbors ────────────────────────────────────────────────
    def _neighbors(self, center, use_small=False):
        offs = self._off12 if use_small else self._off23
        pts  = offs + np.array(center, int)
        sh   = self.volume.shape
        ok   = np.all((pts >= 0) & (pts < sh), axis=1)
        return [tuple(p) for p in pts[ok]]

    # ── batch P ──────────────────────────────────────────────────
    def _P_batch(self, coords):
        if not coords:
            return {}
        feats  = np.array([self.features(c) for c in coords], np.float32)
        probas = self.model.predict_proba(feats)
        return {c: float(probas[i,1]) for i,c in enumerate(coords)}

    @staticmethod
    def _minmax(d):
        if not d: return {}
        vals = np.array(list(d.values()))
        lo,hi = vals.min(), vals.max()
        if hi-lo < 1e-8:
            return {k:0.5 for k in d}
        return {k:(v-lo)/(hi-lo) for k,v in d.items()}

    # ── cosine ───────────────────────────────────────────────────
    @staticmethod
    def _cos(a, b):
        na,nb = np.linalg.norm(a), np.linalg.norm(b)
        if na<1e-8 or nb<1e-8: return 0.0
        return float(np.dot(a,b)/(na*nb))

    # ── main walk ────────────────────────────────────────────────
    def walk(self, start, end,
             max_steps=6000,
             stay_in_seg=False,
             convergence_thr=3.0,
             max_stuck_steps=300,
             branch_skel_pts=None):   # for type3 Eq.11
        """
        Parameters
        ----------
        branch_skel_pts : np.ndarray (M,3) or None
            If given, D(Ak) = -min_{pt in branch} ||Ak-pt||  (Eq.11, type3).
            Otherwise  D(Ak) = -||Ak - end||                 (Eq.6).
        """
        start = tuple(int(x) for x in start)
        end   = tuple(int(x) for x in end)

        # Use smaller neighborhood when gap distance is small
        init_dist  = float(np.linalg.norm(np.array(start)-np.array(end)))
        use_small  = init_dist < 30.0

        # For type3: precompute branch pts array once
        branch_pts = (np.array(branch_skel_pts, float)
                      if branch_skel_pts is not None else None)

        path    = [start]
        visited = {start}
        best_d  = init_dist
        stuck   = 0

        # History: last two offset vectors
        o1 = o2 = None

        pbar = tqdm(total=max_steps, desc="DPC", ncols=100,
                    bar_format='{l_bar}{bar}| {n}/{total} [{elapsed}]')

        for step in range(max_steps):
            cur = path[-1]
            d_cur = float(np.linalg.norm(np.array(cur)-np.array(end)))

            pbar.set_postfix(dist=f"{d_cur:.1f}", path=len(path))
            pbar.update(1)

            # ── convergence ──────────────────────────────────────
            if d_cur <= convergence_thr:
                pbar.close()
                print(f"\n✅ step={step}  dist={d_cur:.2f}")
                if path[-1] != end:
                    path.append(end)
                break

            # ── stuck check ──────────────────────────────────────
            if d_cur < best_d:
                best_d = d_cur; stuck = 0
            else:
                stuck += 1
            if stuck >= max_stuck_steps:
                pbar.close()
                print(f"\n⚠️  stuck at step {step}, best_dist={best_d:.2f}")
                break

            # ── get neighbors ────────────────────────────────────
            cands = self._neighbors(cur, use_small=use_small)

            if stay_in_seg:
                cands = [c for c in cands if self.seg[c] > 0]

            # Filter already visited
            cands = [c for c in cands if c not in visited]

            if not cands:
                pbar.close()
                print(f"\n⚠️  no candidates at step {step}")
                break

            # ── angle filter: cos(ok, o1) >= 0  (≤ 90°) ─────────
            # Paper: "nodes with angles exceeding 90° relative to
            #         historical directions are filtered out"
            if o1 is not None:
                filtered = [
                    c for c in cands
                    if self._cos(np.array(c,float)-np.array(cur,float), o1) >= 0.0
                ]
                # Fall back if everything filtered
                cands = filtered if filtered else cands

            if not cands:
                pbar.close()
                print(f"\n⚠️  all filtered at step {step}")
                break

            # ── batch P → PN ─────────────────────────────────────
            P_raw = self._P_batch(cands)
            PN    = self._minmax(P_raw)

            # ── C guard: cos(o1, o2) ─────────────────────────────
            # Paper Eq.10: if cos(o1,o2) > 0.5, drop C term
            use_C = True
            if o1 is not None and o2 is not None:
                if self._cos(o1, o2) > 0.5:
                    use_C = False

            # ── score each candidate ─────────────────────────────
            best_score = -np.inf
            best_pt    = None
            
            # Precompute current distances once outside the loop
            cur_arr = np.array(cur, float)
            end_arr = np.array(end, float)
            d_cur_end = float(np.linalg.norm(cur_arr - end_arr))

            if branch_pts is not None:
                diffs_cur = branch_pts - cur_arr
                d_cur_branch = float(np.min(np.linalg.norm(diffs_cur, axis=1)))

            for ak in cands:
                ok_vec = np.array(ak, float) - cur_arr

                # D term (Eq.6 or Eq.11) — progress toward target
                if branch_pts is not None:
                    # Eq.11: D = progress toward nearest branch point
                    diffs_ak = branch_pts - np.array(ak, float)
                    d_ak_branch = float(np.min(np.linalg.norm(diffs_ak, axis=1)))
                    D = d_cur_branch - d_ak_branch
                else:
                    # Eq.6: D = progress toward end point
                    d_ak_end = float(np.linalg.norm(np.array(ak, float) - end_arr))
                    D = d_cur_end - d_ak_end

                # P term
                Pn = PN.get(ak, 0.5)

                # C term (Eq.8)
                C = 0.0
                if use_C:
                    if o1 is not None:
                        C += self._cos(ok_vec, o1)
                    if o2 is not None:
                        C += self._cos(ok_vec, o2)

                score = D + self.omega * Pn + C
                if score > best_score:
                    best_score = score
                    best_pt    = ak

            if best_pt is None:
                pbar.close(); break

            # ── update history ────────────────────────────────────
            o2 = o1
            o1 = np.array(best_pt,float) - np.array(cur,float)

            path.append(best_pt)
            visited.add(best_pt)

        else:
            pbar.close()
            print(f"\n⚠️  max_steps={max_steps} reached")

        final_d = float(np.linalg.norm(np.array(path[-1])-np.array(end)))
        print(f"Path={len(path)}  final_dist={final_d:.2f}  "
              f"cache={len(self._cache)}")
        return path
