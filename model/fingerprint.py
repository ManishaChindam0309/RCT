
import hashlib
import torch


def tensor_fingerprint(x):
    """
    Creates a SHA-256 fingerprint of an exact tensor.

    Includes shape, dtype and tensor contents.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("Expected a PyTorch tensor")

    tensor = x.detach().contiguous().cpu()

    digest = hashlib.sha256()

    digest.update(str(tensor.shape).encode())
    digest.update(str(tensor.dtype).encode())

    raw_bytes = tensor.view(torch.uint8).numpy().tobytes()

    digest.update(raw_bytes)

    return digest.hexdigest()


def computation_fingerprint(x, module_version):
    """
    Combines the input fingerprint with a version
    identifying the computation's parameters/configuration.
    """

    digest = hashlib.sha256()

    digest.update(tensor_fingerprint(x).encode())
    digest.update(str(module_version).encode())

    return digest.hexdigest()
