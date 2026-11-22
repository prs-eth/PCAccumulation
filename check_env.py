"""Check the Python environment of the reference model (V0) and of the export / viewer tools.

Run after `bash setup_env.sh`, on the laptop or on the Jetson (if V0 is profiled there):

    source venv_pcaccumulation/bin/activate
    python check_env.py                 # all checks, including one real frame if data/ exists
    python check_env.py --no-frame      # skip the forward pass

Each check prints ok / FAIL / warn. Exit code 1 if any required check fails.
Required: torch + CUDA, torch_scatter, torchsparse, numba, the model + checkpoint (this also
JIT-compiles chamfer_distance/ on first use), and one forward pass when frames are available.
Optional (warn only): onnx / onnxruntime (export/), open3d + matplotlib (viewers).
"""
import argparse, os, sys, time, traceback

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

FAILED = []


def check(name, fn, required=True):
    t = time.perf_counter()
    try:
        info = fn()
        print(f'  ok    {name:<22} {info or ""}  ({(time.perf_counter() - t) * 1e3:.0f} ms)')
        return True
    except Exception as e:
        tag = 'FAIL' if required else 'warn'
        print(f'  {tag:<5} {name:<22} {type(e).__name__}: {e}')
        if required:
            FAILED.append(name)
            if os.environ.get('CHECK_ENV_TRACE'):
                traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/waymo/waymo.yaml')
    ap.add_argument('--ckpt', default='checkpoints/waymo.pth')
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--frame-list', default='data/waymo/selected_frames.txt')
    ap.add_argument('--no-frame', action='store_true', help='skip the forward pass on a real frame')
    args = ap.parse_args()
    os.chdir(REPO)
    print(f'python {sys.version.split()[0]}  ({sys.executable})')

    print('\nGPU stack')
    import torch

    def cuda():
        assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
        p = torch.cuda.get_device_properties(0)
        x = torch.randn(256, 256, device='cuda')
        float((x @ x).sum())
        return (f'torch {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, '
                f'{p.name} (sm_{p.major}{p.minor}, {p.total_memory / 2**30:.1f} GB)')
    if not check('torch + CUDA', cuda):
        print('\nno usable GPU: the remaining checks need CUDA')
        sys.exit(1)

    def scatter():
        import torch_scatter
        src = torch.tensor([[1.], [-3.], [2.], [-1.]], device='cuda')
        idx = torch.tensor([0, 0, 1, 1], device='cuda')
        out = torch_scatter.scatter(src, idx, dim=0, reduce='max')
        assert out.flatten().tolist() == [1., 2.], out.flatten().tolist()
        return f'torch_scatter {torch_scatter.__version__}, scatter max on GPU'
    check('torch_scatter', scatter)

    def sparse():
        import numpy as np
        import torchsparse
        from torchsparse.utils.quantize import sparse_quantize
        coords = np.array([[0, 0, 0], [0, 0, 0], [1, 2, 3]], dtype=np.int32)
        _, sel, inv = sparse_quantize(coords, return_index=True, return_inverse=True)
        assert len(sel) == 2 and len(inv) == 3, (sel, inv)
        torchsparse.SparseTensor(coords=torch.randint(0, 10, (100, 4), dtype=torch.int32, device='cuda'),
                                 feats=torch.rand(100, 32, device='cuda'))
        return f'torchsparse {torchsparse.__version__}, sparse_quantize + SparseTensor'
    check('torchsparse', sparse)

    def jit():
        import numba
        import numpy as np

        @numba.njit
        def f(a):
            s = 0.0
            for v in a:
                s += v
            return s
        assert f(np.arange(10.0)) == 45.0
        return f'numba {numba.__version__}, njit compile + run'
    check('numba', jit)

    print('\nTools (optional)')

    def onnx_ok():
        import onnx, onnxruntime
        return f'onnx {onnx.__version__}, onnxruntime {onnxruntime.__version__}'
    check('onnx / onnxruntime', onnx_ok, required=False)

    def viewers():
        import open3d, matplotlib
        return f'open3d {open3d.__version__}, matplotlib {matplotlib.__version__}'
    check('open3d / matplotlib', viewers, required=False)

    print('\nModel')
    state = {}

    def model():
        from common.reference import build_config, load_model
        assert os.path.isfile(args.ckpt), f'{args.ckpt} not found'
        state['cfg'] = build_config(args.config, 'test')
        state['model'] = load_model(state['cfg'], args.ckpt)
        n = sum(p.numel() for p in state['model'].parameters())
        return f'MotionNet + {args.ckpt}, {n / 1e6:.2f} M parameters'
    check('model + checkpoint', model)

    def chamfer():
        from chamfer_distance.chamfer_distance import ChamferDistance
        a = torch.rand(1, 50, 3, device='cuda')
        d1, d2 = ChamferDistance()(a, a)
        assert float(d1.max()) < 1e-6
        return 'chamfer_distance CUDA extension (JIT) built and runs'
    check('chamfer_distance', chamfer)

    if args.no_frame:
        print('\nreal frame: skipped (--no-frame)')
    elif not os.path.isfile(args.frame_list):
        print(f'\nreal frame: skipped ({args.frame_list} not found; copy the selected frames to {args.data_root}/)')
    elif 'model' in state:
        print('\nReal frame')

        def forward():
            from common.reference import read_frame_list, load_frame, make_preprocessor, prepare_batch
            rel = read_frame_list(args.frame_list)[0]
            batch = prepare_batch(make_preprocessor(state['cfg']), load_frame(args.data_root, rel))
            torch.manual_seed(0)
            with torch.no_grad():
                out = state['model'](batch)
            torch.cuda.synchronize()
            n = batch['input_points'].shape[0]
            moving = int((out['mos_est'].argmax(1) == 1).sum())
            n_obj = len(set(out['inst_labels_est'].cpu().tolist()) - {0})
            err = out.get('ego_trans_error')
            err = f', ego error vs GT {float(torch.as_tensor(err).float().mean()):.3f} m' if err is not None else ''
            return f'{rel}: {n} points, {moving} moving, {n_obj} objects{err}'
        check('forward pass', forward)

    print()
    if FAILED:
        print(f'FAILED: {", ".join(FAILED)}   (set CHECK_ENV_TRACE=1 for tracebacks)')
        sys.exit(1)
    print('environment ok')


if __name__ == '__main__':
    main()
