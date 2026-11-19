"""CPU physical reslicing. 2D matrices use pixel edges: centers are (x+.5,y+.5)."""
import itertools, json
import numpy as np
import SimpleITK as sitk
from .paths import DATA
from .geometry import roi_transform
from .preprocess import prepare_image, nonzero_region

def support_vertices(mask):
    """Row extrema give exact affine extrema of all foreground pixel cells."""
    ys = np.flatnonzero(mask.any(1))
    if not len(ys):
        raise ValueError('empty_support')
    rows = mask[ys]
    lo = rows.argmax(1)
    hi = mask.shape[1] - rows[:, ::-1].argmax(1)
    return np.concatenate([np.c_[lo, ys], np.c_[hi, ys], np.c_[lo, ys + 1], np.c_[hi, ys + 1]])

def transform_points(matrix, points):
    return np.c_[points, np.ones(len(points))] @ matrix[:2, :].T

def support_box(points, matrix, size=256):
    q = transform_points(matrix, points)
    lo = q.min(0)
    hi = q.max(0)
    return np.r_[(lo + hi) / 2, hi - lo].astype(np.float32) / size

def rotation(axis, degrees):
    axis = np.asarray(axis, float).copy()
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    a = np.deg2rad(degrees)
    return np.eye(3) * np.cos(a) + (1 - np.cos(a)) * np.outer(axis, axis) + np.sin(a) * skew

def physical_corners(image, transform=None):
    points = [image.TransformContinuousIndexToPhysicalPoint(v) for v in itertools.product(*[(-0.5, n - 0.5) for n in image.GetSize()])]
    return np.array([transform.TransformPoint(p) for p in points] if transform else points)

def load_source(modality, case):
    path = DATA / modality / case
    meta = json.loads((path / 'complete.json').read_text())
    us = sitk.ReadImage(meta['source_us'])
    us.SetDirection(meta['direction'])
    us.SetSpacing(meta['spacing'])
    arr = sitk.GetArrayFromImage(us).astype(np.float32)
    us = sitk.Cast(us, sitk.sitkFloat32)
    mr = sitk.ReadImage(meta['source_mri'])
    values = sitk.GetArrayFromImage(mr).astype(np.float32)
    values -= values.min()
    values = values.astype(np.int16).astype(np.float32)
    shifted = sitk.GetImageFromArray(values)
    shifted.CopyInformation(mr)
    pull = sitk.ReadTransform(str(path / 'us_to_mri_pull.h5'))
    bounds = np.concatenate([physical_corners(mr, pull.GetInverse()), physical_corners(us)])
    anchors = {}
    for row in meta['slice_records']:
        k = row['k']
        ys, xs = np.where(arr[k] != 0)
        cx, cy = (xs.mean(), ys.mean())
        i = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
        anchors[k] = (float(xs[i]), float(ys[i]))
    mask = sitk.GetImageFromArray((arr != 0).astype(np.uint8))
    mask.CopyInformation(us)
    return dict(us=us, us_support=mask, mr=shifted, pull=pull, bounds=bounds, anchors=anchors, meta=meta)

def slice_planes(source, k, tilt=(0.0, 0.0), jitter=0.0, reference_grid=False):
    us = source['us']
    D = np.array(us.GetDirection()).reshape(3, 3)
    if reference_grid:
        meta = source['meta']
        grid = sitk.Image([*meta['shape_xyz'][:2], 1], sitk.sitkFloat32)
        grid.SetDirection(meta['direction'])
        grid.SetSpacing(meta['spacing'])
        grid.SetOrigin(np.array(meta['origin']) + D[:, 2] * meta['spacing'][2] * k)
        B = D
        pixel_basis = B[:, :2] * np.array(meta['spacing'][:2])[None, :]
    else:
        B = rotation(D[:, 1], tilt[1]) @ rotation(D[:, 0], tilt[0]) @ D
        x, y = source['anchors'][k]
        z = float(np.clip(k + jitter, 0, us.GetSize()[2] - 1))
        center = np.array(us.TransformContinuousIndexToPhysicalPoint((x, y, z)))
        uv = (source['bounds'] - center) @ np.linalg.inv(B).T
        lo = np.floor(uv[:, :2].min(0) / 0.5) - 2
        hi = np.ceil(uv[:, :2].max(0) / 0.5) + 3
        dims = (hi - lo).astype(int)
        grid = sitk.Image([int(dims[0]), int(dims[1]), 1], sitk.sitkFloat32)
        grid.SetDirection(B.flatten())
        grid.SetSpacing((0.5, 0.5, 1.0))
        grid.SetOrigin(center + B @ np.r_[lo * 0.5, 0.0])
        pixel_basis = B[:, :2] * 0.5
    planes = [sitk.GetArrayFromImage(sitk.Resample(im, grid, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32))[0] for im, tx in [(us, sitk.Transform()), (source['mr'], source['pull'])]]
    support = sitk.GetArrayFromImage(sitk.Resample(source['us_support'], grid, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8))[0] > 0
    planes[0][~support] = 0
    return (planes, pixel_basis, dict(origin=list(grid.GetOrigin()), direction=list(grid.GetDirection()), spacing=list(grid.GetSpacing()), shape=list(grid.GetSize())))

def normalized_plane(image):
    valid, bbox = nonzero_region(image)
    lo, hi = np.percentile(image[valid], [0.5, 99.5])
    out = np.zeros(image.shape, np.float32)
    out[valid] = np.clip((image[valid] - lo) / max(float(hi - lo), 1e-06), 0, 1)
    return (out, valid, bbox)

def canvas_matrix(bbox, linear, zoom=1.0, offset_fraction=(0.0, 0.0), size=256):
    y0, y1, x0, x1 = bbox
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    extent = np.ptp((corners - center) @ linear.T, axis=0)
    scale = size / max(extent) * zoom
    matrix = np.eye(3)
    matrix[:2, :2] = linear * scale
    margin = np.maximum((size - extent * scale) / 2, 0)
    delta = np.clip(np.array(offset_fraction) * size, -margin, margin)
    matrix[:2, 2] = size / 2 + delta - matrix[:2, :2] @ center
    return matrix

def warp(image, matrix, nearest=False, size=256):
    inv = np.linalg.inv(matrix)
    tx = sitk.AffineTransform(2)
    tx.SetMatrix(inv[:2, :2].flatten())
    tx.SetTranslation(inv[:2, 2] + inv[:2, :2] @ np.array([0.5, 0.5]) - 0.5)
    source = sitk.GetImageFromArray(np.asarray(image, np.float32))
    result = sitk.Resample(source, [size, size], tx, sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear, [0.0, 0.0], [1.0, 1.0], [1.0, 0.0, 0.0, 1.0], 0.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(result)

def augment(planes, rng, cfg, override=None):
    aug = cfg['augmentation']
    params = {}
    matrices = []
    images = []
    masks = []
    vertices = support_vertices(np.isfinite(planes[0]) & (planes[0] != 0))
    if not aug['enabled']:
        for plane in planes:
            matrix, _ = roi_transform(plane)
            x, mask = prepare_image(plane, return_mask=True)
            images.append(x.numpy()[0] * 2 - 1)
            masks.append(mask.numpy()[0])
            matrices.append(matrix)
    else:
        params = dict(angle=float(rng.uniform(-aug['rotation_degrees'], aug['rotation_degrees'])), flip_x=bool(rng.random() < aug['horizontal_flip_probability']), flip_y=bool(rng.random() < aug['vertical_flip_probability']), stretch=rng.uniform(*aug['stretch_range'], 2).tolist(), mri_zoom=float(rng.uniform(*aug['mri_canvas_scale_range'])), offset=rng.uniform(-aug['mri_translation_fraction'], aug['mri_translation_fraction'], 2).tolist())
        if override:
            params.update(override)
        a = np.deg2rad(params['angle'])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        linear = R @ np.diag(np.array(params['stretch']) * np.array([-1 if params['flip_x'] else 1, -1 if params['flip_y'] else 1]))
        for i, plane in enumerate(planes):
            norm, valid, bbox = normalized_plane(plane)
            matrix = canvas_matrix(bbox, linear, 1.0 if i == 0 else params['mri_zoom'], (0, 0) if i == 0 else params['offset'])
            mask = warp(valid, matrix, True) > 0.5
            if not mask.any():
                raise ValueError('empty_augmented_support')
            image = np.clip(warp(norm, matrix), 0, 1)
            image[~mask] = 0
            images.append(image * 2 - 1)
            masks.append(mask)
            matrices.append(matrix)
    box = support_box(vertices, matrices[1])
    return dict(images=np.stack(images), masks=np.stack(masks), box=box, mr_to_original=np.linalg.inv(matrices[1]).astype(np.float32), correspondence=matrices[1] @ np.linalg.inv(matrices[0]), transforms=np.stack(matrices), augmentation=params)

def generate(source, row, epoch, cfg, index, identity=False):
    settings = cfg['sampling']
    base = [settings['seed'], int(epoch), int(index), 7319]
    attempts = 1 if identity else settings['retry_count'] + 1
    for attempt in range(attempts):
        rng = np.random.default_rng(np.random.SeedSequence(base + [attempt]))
        fallback = not identity and attempt == settings['retry_count']
        tilt = np.zeros(2) if identity or fallback else rng.uniform(-settings['tilt_degrees'], settings['tilt_degrees'], 2)
        jitter = 0.0 if identity or fallback else float(rng.uniform(-settings['normal_jitter_layers'], settings['normal_jitter_layers']))
        planes, basis, geometry = slice_planes(source, row['k'], tilt, jitter, reference_grid=identity or fallback)
        if not all((np.any(np.isfinite(p) & (p != 0)) for p in planes)):
            if identity or fallback:
                raise ValueError(f"Invalid retained anchor {row['case']} k={row['k']}")
            continue
        try:
            sample = augment(planes, rng, cfg)
        except ValueError:
            if identity or fallback:
                raise
            continue
        sample.update(pixel_basis=basis.astype(np.float32), pixel_mm=np.linalg.norm(basis, axis=0).astype(np.float32), audit=dict(case=row['case'], k=row['k'], epoch=int(epoch), index=int(index), attempt=attempt, fallback=fallback, tilt=tilt.tolist(), jitter=jitter, geometry=geometry, augmentation=sample['augmentation']))
        return sample
    raise RuntimeError('No plane generated')
