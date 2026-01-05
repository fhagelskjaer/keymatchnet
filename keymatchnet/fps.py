import numpy as np

def farthest_point_sampler(pc, n, random_start=False):
    m = len(pc)
    assert pc.shape == (m, 3)

    if random_start:
        idx = np.random.randint(0, m)
        p = pc[idx : idx + 1]
        fps_idx = [idx]
    else:
        p = pc.mean(axis=0, keepdims=True)
        fps_idx = []

    for i in range(n):
        if i == n - 1 and random_start:
            break

        dists = np.linalg.norm(p[:, None] - pc[None], axis=2)
        # choose the point in pc which is furthest away from all points in p
        idx = dists.min(axis=0).argmax()
        fps_idx.append(idx)
        p = np.concatenate([p, pc[idx : idx + 1]])
        
        if i == 0 and not random_start: # discard mean point
            p = p[1:]

    return np.array(fps_idx, int)
