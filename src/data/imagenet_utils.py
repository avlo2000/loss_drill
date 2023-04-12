import json


def load_uid_to_idx(path_to_file: str) -> dict:
    """
        Loads mapping for UID to Index for ImageNet labels
    Returns:
        uid_to_idx: dict UID to int index
    """    
    with open(path_to_file, 'r') as f:
        idx_to_uid_class = json.load(f)
    uid_to_idx = {uid: int(idx) for idx, (uid, _) in idx_to_uid_class.items()}
    return uid_to_idx