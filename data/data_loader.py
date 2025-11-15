import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import os, glob, math
from typing import Tuple, List, Dict


def read_nii(path:str):

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    return sitk.ReadImage(path)

def img_to_arr(img:sitk.Image):

    return sitk.GetArrayFromImage(img) 


def resample_projection(img:sitk.Image, out_size=(128, 128), inter= sitk.sitkLinear):
    h_old, w_old = img.GetSize()
    sx_old, sy_old = img.GetSpacing()
    
    h, w = out_size
    sx = sx_old * (w_old / w)
    sy = sy_old * (h_old / h)

    out = sitk.Resample(
        img,
        [w, h],
        sitk.Transform(),
        inter,
        img.GetOrigin(),
        [sx, sy],
        img.GetDirection(),
        0.0,
        img.GetPixelIDValue()
    )
    return out

def resample_ct(img: sitk.Image, out_dhw=(128,128,128), interp=sitk.sitkLinear):

    D_new, H_new, W_new = out_dhw
    W_old, H_old, D_old = img.GetSize() 
    sx_old, sy_old, sz_old = img.GetSpacing()
    sx_new = sx_old * (W_old / W_new)
    sy_new = sy_old * (H_old / H_new)
    sz_new = sz_old * (D_old / D_new)
    
    return sitk.Resample(
        img,
        [W_new, H_new, D_new],
        sitk.Transform(),
        interp,
        img.GetOrigin(),
        [sx_new, sy_new, sz_new],
        img.GetDirection(),
        0.0,
        img.GetPixelID()
    )

def normalize_array(arr: np.ndarray):
    arr = arr.astype(np.float32)
    min, max = float(arr.min()), float(arr.max())
    return (arr - min) / (max - min + 1e-6) 


class XRays_CT_Dataset(Dataset):

    def __init__(self, root:str, drr_size, ct_size, angles_deg):
        super().__init__()
        self.root = root
        self.drr_size = drr_size
        self.ct_size = ct_size
        self.angles_deg = angles_deg
        self.cases = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        if not self.cases:
            raise RuntimeError(f"No case found in {root}")
        
        self.case_to_index = {case_id: i for i, case_id in enumerate(self.cases)}
    
    def __len__(self):
        return len(self.cases)


    def get_index_by_case(self, case_or_path:str):

        name = os.path.basename(case_or_path.rstrip("/"))
        name_lower = name.lower()

        # match exact (case-insensitive)
        exact = [cid for cid in self.cases if cid.lower() == name_lower]
        if exact:
            return self.case_to_index[exact[0]]

        # sinon, autoriser un préfixe unique (ex: "verse01")
        pref = [cid for cid in self.cases if cid.lower().startswith(name_lower)]
        if not pref:
            raise KeyError(f"Case '{name}' not found under {self.root}.")
        if len(pref) > 1:
            raise ValueError(f"Ambiguous prefix '{name}': matches {pref}. Please provide the full case ID.")
        return self.case_to_index[pref[0]]
    
    def get_by_case(self, case_or_path: str):
        idx = self.get_index_by_case(case_or_path)
        return self.__getitem__(idx)



    def _find_ct(self, dir:str):
        for stem in ("ct", "CT"):
            for ext in (".nii.gz", ".nii"):
                path = os.path.join(dir, stem + ext)
                if os.path.exists(path):
                    return path
        
    
    def _find_drr_angle(self, dir:str, deg:float):

        tag_int = f"angle{int(round(deg))%360:03d}"

        patterns = [
            os.path.join(dir, f"verse*_drr_{tag_int}.nii*"),
            os.path.join(dir, f"*drr*_{tag_int}.nii*"),
            os.path.join(dir, f"drr_{tag_int}.nii*"),
            os.path.join(dir, f"*{tag_int}.nii*"),
        ]
        for pat in patterns:
            gl = sorted(glob.glob(pat))
            if gl:
                return gl[0]


    def __getitem__(self, index:int):

        case_id = self.cases[index]
        dir = os.path.join(self.root, case_id)
        
        # CT 
        ct_img = read_nii(self._find_ct(dir))
        ct_img = resample_ct(ct_img, self.ct_size)
        
        ct_arr = img_to_arr(ct_img)                # (D, H, W)
        ct_arr = normalize_array(ct_arr)

        ct = torch.from_numpy(ct_arr).unsqueeze(0).unsqueeze(0) # (1, D, H, W)


        # DRRs
        
        views = []
        angles_rad = []
        for deg in self.angles_deg:
            drr = self._find_drr_angle(dir, deg)

            img = read_nii(drr)
            img = resample_projection(img, self.drr_size)

            drr_arr = img_to_arr(img)                       # (H, W)
            drr_arr = normalize_array(drr_arr)

            drr = torch.from_numpy(drr_arr).unsqueeze(0)    # (1, H, W)

            views.append(drr)
            angles_rad.append(math.radians(deg))

        views = torch.stack(views,dim=0)
        angles_rad = torch.tensor(angles_rad, dtype=torch.float32)

        return {
            "views": views,
            "angles": angles_rad,
            "ct": ct,
            "id": case_id
        }


def make_loader(root:str, angles_deg, drr_size: Tuple[int, int], ct_size:Tuple[int, int, int], batch_size:int, shuffle: bool = True):
    ds = XRays_CT_Dataset(
        root=root, 
        angles_deg=angles_deg,
        drr_size=drr_size,
        ct_size=ct_size
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
