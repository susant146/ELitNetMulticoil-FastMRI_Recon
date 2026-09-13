
import json
from pathlib import Path
import h5py
from astropy.io import fits


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        fname = fname.split('/')[-1]
        fits.writeto(str(out_dir/fname), recons.squeeze(0), overwrite=True)

def save_rec_res(reconstructions, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        fname = fname.split('/')[-1]
        fits.writeto(str(out_dir/fname), recons.squeeze(0), overwrite=True)

def save_ZF(ZF_image, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for fname, image in ZF_image.items():
        fname = fname.split('/')[-1]
        fits.writeto(str(out_dir/fname), image.squeeze(0), overwrite=True)


def save_target(target, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    for fname, image in target.items():
        fname = fname.split('/')[-1]
        fits.writeto(str(out_dir/fname), image.squeeze(0), overwrite=True)


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.

    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]
