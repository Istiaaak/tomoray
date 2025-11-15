import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch



def _to_numpy(x: torch.Tensor):
    """Convert tensor to CPU numpy array (float if needed)."""
    if torch.is_floating_point(x):
        x = x.detach().cpu().float()
    else:
        x = x.detach().cpu()
    return x.numpy()


def _to_zyx(vol):
    """
    Ensure a 3D numpy array (Z, Y, X) from common shapes:
      - (Z, Y, X)
      - (1, Z, Y, X)
      - (Z, Y, X, 1)
    """
    arr = np.asarray(vol)
    arr = np.squeeze(arr)  # remove size-1 dimensions

    if arr.ndim == 4:
        if arr.shape[0] == 1:      # (1, Z, Y, X) -> (Z, Y, X)
            arr = arr[0]
        elif arr.shape[-1] == 1:   # (Z, Y, X, 1) -> (Z, Y, X)
            arr = arr[..., 0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D (Z,Y,X), got {arr.shape}")

    return arr


def pick_slices(vol_zyx: np.ndarray, z=None, y=None, x=None):
    """
    Pick central (or specified) orthogonal slices from a 3D volume.

    Args:
        vol_zyx: (Z, Y, X)
        z, y, x: optional indices for axial / coronal / sagittal slices.

    Returns:
        axial, coronal, sagittal: 2D numpy arrays.
    """
    Z, Y, X = vol_zyx.shape
    z = Z // 2 if z is None else max(0, min(Z - 1, z))
    y = Y // 2 if y is None else max(0, min(Y - 1, y))
    x = X // 2 if x is None else max(0, min(X - 1, x))
    axial   = vol_zyx[z, :, :]
    coronal = vol_zyx[:, y, :]
    sagitt  = vol_zyx[:, :, x]
    return axial, coronal, sagitt


def show_triptych(pred_zyx, gt_zyx=None, title_prefix="Epoch ?"):
    """
    Display axial / coronal / sagittal slices in a single horizontal strip.

    If gt_zyx is provided, each slice is shown as [pred; gt] stacked vertically.
    """
    pred_zyx = _to_zyx(pred_zyx)
    vmin, vmax = pred_zyx.min(), pred_zyx.max()

    axial, coronal, sagitt = pick_slices(pred_zyx)
    panels = [axial, coronal, sagitt]

    if gt_zyx is not None:
        gt_zyx = _to_zyx(gt_zyx)
        vmin = min(vmin, gt_zyx.min())
        vmax = max(vmax, gt_zyx.max())
        a2, c2, s2 = pick_slices(gt_zyx)
        axial   = np.concatenate([axial,  a2], axis=0)
        coronal = np.concatenate([coronal, c2], axis=0)
        sagitt  = np.concatenate([sagitt, s2], axis=0)
        panels = [axial, coronal, sagitt]

    strip = np.concatenate(panels, axis=1)
    plt.figure(figsize=(12, 4))
    plt.imshow(strip, cmap="gray", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.title(
        f"{title_prefix} — Axial | Coronal | Sagittal"
        + (" (top=pred, bottom=GT)" if gt_zyx is not None else "")
    )
    plt.show()


def save_nifti(vol_zyx: np.ndarray, out_path: str, spacing=(1.0, 1.0, 1.0)):
    """
    Save a (Z, Y, X) volume as NIfTI, with isotropic spacing by default.
    """
    img = sitk.GetImageFromArray(vol_zyx.astype(np.float32))  # ITK expects (Z, Y, X)
    img.SetSpacing(tuple(map(float, spacing)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(img, out_path)


def viz_fixed_sample(model, ds, device, idx=0, title="Epoch ?"):
    """
    Pick a fixed sample from a dataset and visualize
    the predicted vs. ground-truth volume.
    """
    model.eval()
    with torch.no_grad():
        sample = ds[idx]
        views  = sample["views"].unsqueeze(0).to(device)   # (1, V, 1, H, W)
        angles = sample["angles"].unsqueeze(0).to(device)  # (1, V)
        ct     = sample["ct"].unsqueeze(0).to(device)      # (1, 1, D, H, W)

        pred = model(views, angles, chunk=16384)           # (1, 1, D, H, W)

    pred_np = pred[0, 0].cpu().numpy()
    ct_np   = ct[0, 0].cpu().numpy()
    show_triptych(pred_np, ct_np, title_prefix=title)
    model.train()