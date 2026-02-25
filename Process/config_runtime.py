import os
import agd

def configure_agd():
    path = os.environ.get("FILEHFM_BINARY_DIR")

    if path is None:
        raise RuntimeError(
            "FILEHFM_BINARY_DIR not set.\n"
            "Please set it in your environment."
        )

    agd.Eikonal.LibraryCall.binary_dir["FileHFM"] = path