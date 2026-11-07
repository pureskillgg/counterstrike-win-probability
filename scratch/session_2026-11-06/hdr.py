import numpy as np, os
E = '/home/sagemaker-user/adam/builder/final/export_clean_v2'
for f in sorted(os.listdir(E)):
    p = os.path.join(E, f)
    if f.endswith('.npy'):
        a = np.load(p, mmap_mode='r')
        print(f"{f:22s} shape={a.shape} dtype={a.dtype} fortran={a.flags['F_CONTIGUOUS'] and not a.flags['C_CONTIGUOUS']} bytes={os.path.getsize(p)}")
    else:
        print(f"{f:22s} (non-npy) bytes={os.path.getsize(p)}")
