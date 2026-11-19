"""MRIFullVision fixed inputs and trajectory-anchored dynamic inputs.

All image coordinates use pixel edges. Physical grid origins use pixel centers.
The MRI head mask determines the sole crop, after any geometric augmentation.
"""
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from .paths import RAW, case_cache_path
from .geometry import roi_transform
from .preprocess import prepare_image
from .head_roi import prepare_final_mri
from .reslicing import support_vertices, support_box, physical_corners, rotation, normalized_plane, canvas_matrix, warp

def read(path):
    return json.loads(Path(path).read_text())

def selected_units(cfg):
    root = Path(cfg['data']['dataset_root'])
    split = read(cfg['data']['split_file'])
    partitions = {case: part for part in ('train', 'validation', 'test') for case in split[part]}
    assert len(partitions) == sum((len(split[p]) for p in ('train', 'validation', 'test')))
    units = [dict(u, partition=partitions[u['case']]) for u in read(root / 'Processing_Metadata/units.json') if u['branch'] == 'coregistered/' + cfg['data']['branch'] and u['case'] in partitions]
    assert len(units) == len(partitions)
    return sorted(units, key=lambda u: u['case'])

def case_info(cfg, unit):
    root = Path(cfg['data']['dataset_root'])
    meta = root / 'Processing_Metadata/units' / unit['id']
    suffix = 'cet1' if cfg['modality'] == 'ceT1' else 'flair'
    reg = next((r for r in read(meta / 'registration.json') if r['modality'] == suffix))
    return dict(unit=unit, root=str(root), meta=str(meta), suffix=suffix, source_us=str(RAW.parent / unit['US']), source_mri=reg['native'], pull=str(meta / f'{suffix}_US_to_native_MRI.h5'), registration=reg)

def grid(matrix, shape_hw):
    matrix = np.asarray(matrix)
    spacing = np.linalg.norm(matrix[:3, :3], axis=0)
    im = sitk.Image([int(shape_hw[1]), int(shape_hw[0]), 1], sitk.sitkFloat32)
    im.SetOrigin(matrix[:3, 3])
    im.SetSpacing(spacing)
    im.SetDirection((matrix[:3, :3] / spacing).ravel())
    return im

def sample_plane(image, reference, pull, nearest=False):
    return sitk.GetArrayFromImage(sitk.Resample(image, reference, pull, sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear, 0, sitk.sitkUInt8 if nearest else sitk.sitkFloat32))[0]

def triangle_head(native, settings):
    filt = sitk.TriangleThresholdImageFilter()
    filt.SetNumberOfHistogramBins(settings['histogram_bins'])
    filt.SetInsideValue(0)
    filt.SetOutsideValue(1)
    mask = filt.Execute(sitk.Cast(native, sitk.sitkFloat32))
    mask = sitk.BinaryMorphologicalOpening(mask, [settings['opening_radius_voxels']] * 4)
    mask = sitk.RelabelComponent(sitk.ConnectedComponent(mask)) == 1
    mask = sitk.BinaryMorphologicalClosing(mask, [settings['closing_radius_voxels']] * 3)
    return (sitk.BinaryFillhole(mask), float(filt.GetThreshold()))

def shifted_mri(native):
    arr = sitk.GetArrayFromImage(native).astype(np.float32)
    arr -= arr.min()
    shifted = sitk.GetImageFromArray(arr.astype(np.int16).astype(np.float32))
    shifted.CopyInformation(native)
    return shifted

def finish_input(us, mr, support, head, basis, cfg, linear=None, offset=(0.0, 0.0)):
    size = cfg['data']['size']
    head_method = 'native_3d_Triangle'
    if not head.any():
        filt = sitk.TriangleThresholdImageFilter()
        filt.SetNumberOfHistogramBins(cfg['data']['MRI_preprocessing']['histogram_bins'])
        filt.SetInsideValue(0)
        filt.SetOutsideValue(1)
        mask = filt.Execute(sitk.GetImageFromArray(np.asarray(mr, np.float32)))
        mask = sitk.RelabelComponent(sitk.ConnectedComponent(mask)) == 1
        head = sitk.GetArrayFromImage(sitk.BinaryFillhole(mask)) > 0
        head_method = 'slice_2d_Triangle_empty_3d_intersection'
    if linear is None:
        linear = np.eye(2)
        ui, um = prepare_image(us, size=size, return_mask=True)
        ui, um = (ui.numpy()[0], um.numpy()[0])
        ua, _ = roi_transform(us, size)
    else:
        norm, valid, bbox = normalized_plane(us)
        ua = canvas_matrix(bbox, linear, size=size)
        ui = np.clip(warp(norm, ua, size=size), 0, 1)
        um = warp(valid, ua, True, size=size) > 0.5
        ui[~um] = 0
    mi, mm, ma, crop = prepare_final_mri(mr, head, linear, np.linalg.norm(basis, axis=0), cfg['data']['MRI_preprocessing']['margin_mm'], size)
    crop['head_method'] = head_method
    points = support_vertices(support) + np.asarray(offset)
    box = support_box(points, ma, size)
    translation = np.eye(4)
    translation[:2, 2] = offset
    return dict(images=np.stack([ui, mi]).astype(np.float32) * 2 - 1, masks=np.stack([um, mm]), box=box, mr_to_original=np.linalg.inv(ma).astype(np.float32), pixel_basis=np.asarray(basis, np.float32), pixel_mm=np.linalg.norm(basis, axis=0).astype(np.float32), US_to_input=ua, MRI_to_input=ma, correspondence=ma @ translation @ np.linalg.inv(ua), crop=crop)

def fixed_sample(cfg, info, row, head, pull):
    root = Path(info['root'])
    us_entry = next((x for x in row['outputs'] if x['kind'] == 'US'))
    mr_entry = next((x for x in row['outputs'] if x['kind'] == 'MRI' and x['source'].endswith('-' + info['suffix'] + '.nii.gz')))
    us = np.load(root / us_entry['path'])
    mr = np.load(root / mr_entry['path'])
    matrix = np.asarray(mr_entry['pixel_to_LPS'])
    head2d = sample_plane(head, grid(matrix, mr.shape), pull, True) > 0
    support = np.load(root / row['US_support']).astype(bool)
    return finish_input(us, mr, support, head2d, matrix[:3, :2], cfg, offset=mr_entry['US_to_MRI_offset_xy'])

def load_source(cfg, case):
    folder = case_cache_path(cfg, case)
    info = read(folder / 'source.json')
    us = sitk.ReadImage(info['source_us'], sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(us)
    labels, _ = ndi.label(arr != 0, structure=np.ones((3, 4, 3)))
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    support = labels == counts.argmax()
    arr[~support] = 0
    cleaned = sitk.GetImageFromArray(arr)
    cleaned.CopyInformation(us)
    mask = sitk.GetImageFromArray(support.astype(np.uint8))
    mask.CopyInformation(us)
    native = shifted_mri(sitk.ReadImage(info['source_mri']))
    pull = sitk.ReadTransform(info['pull'])
    poses = dict(np.load(Path(info['meta']) / 'poses.npz'))
    head = sitk.ReadImage(str(folder / 'Triangle_mask.nii.gz'), sitk.sitkUInt8)
    head.CopyInformation(native)
    return dict(us=cleaned, support=mask, mr=native, pull=pull, poses=poses, info=info, head=head, bounds=np.concatenate([physical_corners(us), physical_corners(native, pull.GetInverse())]))

def dynamic_planes(source, k, tilt, jitter_mm, pixel_mm, reference_row=None):
    if reference_row is not None:
        mr = next((x for x in reference_row['outputs'] if x['kind'] == 'MRI' and x['source'].endswith('-' + source['info']['suffix'] + '.nii.gz')))
        matrix = np.asarray(mr['pixel_to_LPS'])
        reference = grid(matrix, mr['shape_hw'])
    else:
        base = source['poses']['frames_uvn_LPS'][k].T
        basis = rotation(base[:, 1], tilt[1]) @ rotation(base[:, 0], tilt[0]) @ base
        center = source['poses']['centers_LPS'][k] + basis[:, 2] * jitter_mm
        uv = (source['bounds'] - center) @ basis
        low = np.floor(uv[:, :2].min(0) / pixel_mm) - 2
        high = np.ceil(uv[:, :2].max(0) / pixel_mm) + 2
        matrix = np.eye(4)
        matrix[:3, :2] = basis[:, :2] * pixel_mm
        matrix[:3, 2] = basis[:, 2]
        matrix[:3, 3] = center + basis @ np.r_[low * pixel_mm, 0.0]
        reference = grid(matrix, (high - low).astype(int)[::-1])
    us = sample_plane(source['us'], reference, sitk.Transform())
    support = sample_plane(source['support'], reference, sitk.Transform(), True) > 0
    us[~support] = 0
    mr = sample_plane(source['mr'], reference, source['pull'])
    head = sample_plane(source['head'], reference, source['pull'], True) > 0
    return (us, mr, support, head, matrix)

def generate(source, row, epoch, cfg, index):
    sampling, aug = (cfg['sampling'], cfg['augmentation'])
    for attempt in range(sampling['retry_count'] + 1):
        rng = np.random.default_rng(np.random.SeedSequence([sampling['seed'], int(epoch), int(index), 6458, attempt]))
        fallback = attempt == sampling['retry_count']
        tilt = np.zeros(2) if fallback else rng.uniform(-sampling['tilt_degrees'], sampling['tilt_degrees'], 2)
        jitter = 0.0 if fallback else float(rng.uniform(-sampling['normal_jitter_layers'], sampling['normal_jitter_layers']) * source['us'].GetSpacing()[2])
        us, mr, support, head, matrix = dynamic_planes(source, row['k'], tilt, jitter, sampling['pixel_mm'])
        if not support.any() or not np.any(us) or (not np.any(mr)):
            if fallback:
                raise ValueError(f"Empty trajectory anchor {row['case']} / {row['k']}")
            continue
        params = {}
        linear = None
        if aug['enabled']:
            params = dict(angle=float(rng.uniform(-aug['rotation_degrees'], aug['rotation_degrees'])), flip_x=bool(rng.random() < aug['horizontal_flip_probability']), flip_y=bool(rng.random() < aug['vertical_flip_probability']), stretch=rng.uniform(*aug['stretch_range'], 2).tolist())
            angle = np.deg2rad(params['angle'])
            linear = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ np.diag(np.array(params['stretch']) * [-1 if params['flip_x'] else 1, -1 if params['flip_y'] else 1])
        sample = finish_input(us, mr, support, head, matrix[:3, :2], cfg, linear)
        sample['audit'] = dict(case=row['case'], k=row['k'], epoch=int(epoch), index=int(index), attempt=attempt, fallback=fallback, tilt=tilt.tolist(), jitter_mm=jitter, pixel_to_LPS=matrix.tolist(), augmentation=params, crop=sample['crop'])
        return sample
    raise RuntimeError('No trajectory plane generated')
