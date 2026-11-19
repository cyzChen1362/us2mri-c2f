"""Native MRI LPS, millimetres and XYZ Euler degrees. Arrays are ZYX."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

def wrap(x):
    return (x + 180) % 360 - 180

def rotation(a):
    x, y, z = torch.deg2rad(a.float()).unbind(-1)
    cx, cy, cz = (x.cos(), y.cos(), z.cos())
    sx, sy, sz = (x.sin(), y.sin(), z.sin())
    return torch.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx, sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx, -sy, cy * sx, cy * cx], -1).reshape(-1, 3, 3)

def physical_rotation(p, basis=None):
    r = rotation(p[:, 3:])
    return r if basis is None else basis.float() @ r

def physical_pose(p, basis):
    r = physical_rotation(p, basis)
    a = torch.stack([torch.atan2(r[:, 2, 1], r[:, 2, 2]), torch.atan2(-r[:, 2, 0], torch.sqrt(r[:, 0, 0] ** 2 + r[:, 1, 0] ** 2)), torch.atan2(r[:, 1, 0], r[:, 0, 0])], -1)
    return torch.cat([p[:, :3], torch.rad2deg(a)], -1)

def points(p, q, basis=None):
    return torch.einsum('bij,bhwj->bhwi', physical_rotation(p, basis), q) + p[:, None, None, :3]

def localize(initial, target):
    """Reference is the observable initial orientation, never the target."""
    basis = Rotation.from_euler('xyz', initial[:, 3:], degrees=True).as_matrix()
    rt = Rotation.from_euler('xyz', target[:, 3:], degrees=True).as_matrix()
    a = initial.copy()
    a[:, 3:] = 0
    b = target.copy()
    b[:, 3:] = Rotation.from_matrix(basis.transpose(0, 2, 1) @ rt).as_euler('xyz', degrees=True)
    return (a, b, basis.astype('float32'))

def render(v, p, q, origin, basis=None):
    world = points(p.float(), q.float(), basis)
    size = torch.tensor(v.shape[-3:][::-1], device=p.device)
    grid = 2 * (world - origin) / (size - 1) - 1
    return F.grid_sample(v.expand(len(p), -1, -1, -1, -1), grid[:, None], mode='bilinear', padding_mode='zeros', align_corners=True)[:, :, 0] * 2 - 1

def grid(params, device):
    params = torch.as_tensor(params, device=device, dtype=torch.float32)
    y, x = torch.meshgrid(torch.arange(256, device=device), torch.arange(256, device=device), indexing='ij')
    xy = torch.stack([x, y], -1)[None] * params[:, None, None, 2:3] + params[:, None, None, :2]
    return F.pad(xy, (0, 1))

def sample_initial(target, emp, rng):
    k = rng.integers(len(emp['translation_local_mm']), size=len(target))
    er = emp['rotation_local'][k]
    d = emp['translation_local_mm'][k]
    extra = np.zeros((len(k), 3))
    axis = rng.normal(size=(len(k), 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    noise = Rotation.from_rotvec(axis * np.deg2rad(rng.uniform(0, 5, len(k)))[:, None]).as_matrix()
    r = Rotation.from_euler('xyz', target[:, 3:], degrees=True).as_matrix()
    p = target[:, :3] + np.einsum('bij,bj->bi', r, d + np.einsum('bij,bj->bi', er, extra))
    angles = Rotation.from_matrix(r @ er @ noise).as_euler('xyz', degrees=True)
    return np.c_[p, angles].astype('float32')

def references(initial, target, rng):
    d = np.c_[target[:, :3] - initial[:, :3], wrap(target[:, 3:] - initial[:, 3:])]
    sigma = np.c_[np.repeat(np.minimum(1, 0.1 * np.linalg.norm(d[:, :3], axis=1))[:, None], 3, 1), np.repeat(np.minimum(1, 0.1 * np.linalg.norm(d[:, 3:], axis=1))[:, None], 3, 1)]
    refs = []
    for k in range(1, 6):
        a = initial + d * k / 5
        if k < 5:
            a = a + rng.normal(size=a.shape) * sigma * np.sin(np.pi * k / 5)
        else:
            a = target.copy()
        a[:, 3:] = wrap(a[:, 3:])
        refs.append(a)
    return np.stack(refs, 1).astype('float32')
