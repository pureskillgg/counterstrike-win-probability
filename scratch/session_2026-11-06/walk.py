import os, sys, json, time
RP = 0x400
def walk(root):
    files = dirs = nbytes = reparse = errs = 0
    rp_list = []
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            it = os.scandir(cur)
        except OSError as e:
            errs += 1
            continue
        with it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    errs += 1
                    continue
                if st.st_file_attributes & RP:
                    reparse += 1
                    if len(rp_list) < 5000:
                        rp_list.append(e.path)
                    continue
                if e.is_dir(follow_symlinks=False):
                    dirs += 1
                    stack.append(e.path)
                else:
                    files += 1
                    nbytes += st.st_size
    return dict(root=root, files=files, dirs=dirs, bytes=nbytes, reparse_skipped=reparse, errors=errs)
if __name__ == '__main__':
    out = []
    for r in sys.argv[2:]:
        t = time.time()
        res = walk(r)
        res['secs'] = round(time.time() - t, 1)
        out.append(res)
        print(json.dumps(res), flush=True)
    with open(sys.argv[1], 'w') as f:
        json.dump(out, f, indent=1)
