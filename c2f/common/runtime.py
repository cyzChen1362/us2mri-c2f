import os, random, signal, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .paths import compute_only
STOP = False

def setup(seed):
    compute_only()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.device('cuda', local)

def rank():
    return dist.get_rank() if dist.is_initialized() else 0

def world():
    return dist.get_world_size() if dist.is_initialized() else 1

def gather(value):
    if world() == 1:
        return [value]
    result = [None] * world()
    dist.all_gather_object(result, value)
    return result

def sync_bool(value, device):
    flag = torch.tensor(int(value), device=device)
    if world() > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())

def barrier():
    if world() > 1:
        dist.barrier()

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())

def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state(s['cuda'])

def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    with temporary.open('rb') as f:
        os.fsync(f.fileno())
    if path.name == 'last.pt' and path.exists():
        previous = path.with_name('previous.pt')
        path.replace(previous)
    temporary.replace(path)

def unwrap(model):
    return model.module if hasattr(model, 'module') else model

def stop_reason(run):
    job = os.environ.get('SLURM_JOB_ID', '')
    if (Path(run) / f'.cancel_{job}').exists():
        return 'manual'
    if (Path(run) / f'.deadline_{job}').exists():
        return 'deadline'
    end = float(os.environ.get('REMIND_END_TIME', 'inf'))
    if time.time() >= end - 300:
        return 'deadline'
    return None
