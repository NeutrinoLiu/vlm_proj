import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml


class InvalidQAContext(Exception):
    def __init__(self, func_name):
        super().__init__(f"Invalid context for function {func_name}.")

def anno_of_obj_from_frame(frame, obj):
    token = obj["instance_token"]
    for anno in frame["annos"]:
        if anno["instance_token"] == token:
            return anno
    return None

# --------------------------- physicals calculation -------------------------- #

dist = lambda pt1, pt2: ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2) ** 0.5

class Captioner:
    def __init__(self, cap_file):
        with open(cap_file, 'r') as f:
            self.caps = yaml.safe_load(f)
    def obj_desc_fn(self, idx):
        def obj_desc(ctx):
            objs = ctx['objs']
            scene = ctx['scene'].name
            assert len(objs) > idx, f"query object index {idx} out of range, only {len(objs)} objects"
            obj = objs[idx]
            cap = self.caps[scene][obj["instance_token"]]
            return cap
        return obj_desc
    def caption_ready(self, scene, inst_token):
        return scene in self.caps and inst_token in self.caps[scene]

def obj_desc_fn(idx):
    def obj_desc(ctx):
        objs = ctx['objs']
        assert len(objs) > idx, f"query object index {idx} out of range, only {len(objs)} objects"
        return objs[idx]["category_name"].split(".")[-1]
    return obj_desc

def obj_cam_dist_fn(idx):
    def obj_cam_dist(ctx):
        objs = ctx['objs']
        frames = ctx['frames']
        assert len(objs) > idx, f"query object index {idx} out of range, only {len(objs)} objects"
        assert len(frames) == 1, f"obj_cam_dist should only have one frame, but got {len(frames)}"
        frame = frames[0]
        obj = objs[idx]

        anno = anno_of_obj_from_frame(frame, obj)
        assert anno is not None, f"Object {obj['instance_token']} not found in frame {frame['sample_data_token']}"

        cam_t = frame["cam_t"]
        obj_t = anno["box_t"]
        cam_dist = dist(cam_t, obj_t)
        return f"{cam_dist:.2f}"
    return obj_cam_dist

def obj_dist_between(ctx):
    objs = ctx['objs']
    frames = ctx['frames']

    assert len(objs) == 2, f"obj_dist_between should only have two objects, but got {len(objs)}"
    assert len(frames) == 1, f"obj_dist_between should only have one frame, but got {len(frames)}"
    frame = frames[0]
    obj1 = objs[0]
    obj2 = objs[1]

    anno1 = anno_of_obj_from_frame(frame, obj1)
    anno2 = anno_of_obj_from_frame(frame, obj2)
    assert anno1 is not None, f"Object {obj1['instance_token']} not found in frame {frame['sample_data_token']}"
    assert anno2 is not None, f"Object {obj2['instance_token']} not found in frame {frame['sample_data_token']}"

    obj1_t = anno1["box_t"]
    obj2_t = anno2["box_t"]
    between_dist = dist(obj1_t, obj2_t)
    return f"{between_dist:.2f}"

def obj_cam_dist_minmax(ctx):
    objs = ctx['objs']
    frames = ctx['frames']

    assert len(objs) == 1, f"obj_cam_dist_range should only have one object, but got {len(objs)}"
    assert len(frames) > 1, f"obj_cam_dist_range should have at least two frames, but got {len(frames)}"
    obj = objs[0]
    if ctx['minmax'] == 'min':
        fn = min
    elif ctx['minmax'] == 'max':
        fn = max
    else:
        raise ValueError(f"minmax should be 'min' or 'max', but got {ctx['minmax']}")
    cam_dists = []
    for frame in frames:
        anno = anno_of_obj_from_frame(frame, obj)
        if anno is None: continue
        cam_t = frame["cam_t"]
        obj_t = anno["box_t"]
        cam_dists.append(
            dist(cam_t, obj_t)
        )

    return f"{fn(cam_dists):.2f}"

def obj_dist_between_minmax(ctx):
    objs = ctx['objs']
    frames = ctx['frames']

    assert len(objs) == 2, f"obj_dist_between_range should only have two objects, but got {len(objs)}"
    assert len(frames) > 1, f"obj_dist_between_range should have at least two frames, but got {len(frames)}"
    obj1 = objs[0]
    obj2 = objs[1]
    between_dists = []
    if ctx['minmax'] == 'min':
        fn = min
    elif ctx['minmax'] == 'max':
        fn = max
    else:
        raise ValueError(f"minmax should be 'min' or 'max', but got {ctx['minmax']}")
    for frame in frames:
        anno1 = anno_of_obj_from_frame(frame, obj1)
        anno2 = anno_of_obj_from_frame(frame, obj2)
        if anno1 is None or anno2 is None: continue
        obj1_t = anno1["box_t"]
        obj2_t = anno2["box_t"]
        between_dists.append(
            dist(obj1_t, obj2_t)
        )
    
    if len(between_dists) == 0:
        raise InvalidQAContext("obj_dist_between_minmax")

    return f"{fn(between_dists):.2f}"


def local_coords(anno, frame):
    cam_quat = np.array(frame["cam_r"]) # w, x, y, z
    cam_pose = np.array(frame["cam_t"])
    obj_pos = np.array(anno["box_t"])

    rel_pos = obj_pos - cam_pose

    r = R.from_quat(cam_quat, scalar_first=True)
    r_inv = r.inv()

    ret = r_inv.apply(rel_pos)
    x = - ret[1] # x axis is ->right
    y = ret[0]  # y axis is  ->forward
    return x, y

def obj_local_coords_fn(idx):
    def obj_local_coords(ctx):
        objs = ctx['objs']
        frames = ctx['frames']
        assert len(objs) > idx, f"query object index {idx} out of range, only {len(objs)} objects"
        assert len(frames) == 1, f"obj_local_coords should only have one frame, but got {len(frames)}"
        frame = frames[0]
        obj = objs[idx]

        anno = anno_of_obj_from_frame(frame, obj)
        assert anno is not None, f"Object {obj['instance_token']} not found in frame {frame['sample_data_token']}"

        x, y = local_coords(anno, frame)
        return f"{x:.2f} {y:.2f}"

    return obj_local_coords

# ---------------------------- indexing functions ---------------------------- #

def index_of_minmax_dist(ctx):
    objs = ctx['objs']
    frames = ctx['frames']
    if ctx['minmax'] == 'min':
        fn = min
        place_holder = float("inf")
    elif ctx['minmax'] == 'max':
        fn = max
        place_holder = float("-inf")
    else:
        raise ValueError(f"minmax should be 'min' or 'max', but got {ctx['minmax']}")

    assert len(objs) > 1, f"index_of_minmax_dist should have multiple objects, but got {len(objs)}"
    assert len(frames) == 1, f"index_of_minmax_dist should only have one frame, but got {len(frames)}"

    frame = frames[0]
    cam_dists = []
    for obj in objs:
        anno = anno_of_obj_from_frame(frame, obj)
        if anno is None: 
            cam_dists.append(place_holder)
            continue
        cam_t = frame["cam_t"]
        obj_t = anno["box_t"]
        cam_dists.append(
            dist(cam_t, obj_t)
        )
    
    idx_of_minmax = cam_dists.index(fn(cam_dists))
    if cam_dists[idx_of_minmax] == place_holder:
        raise InvalidQAContext("index_of_minmax_dist")
    return idx_of_minmax, cam_dists

# ------------------------------- frame related ------------------------------ #

def frame_idx_fn(idx, from_zero=False):
    def frame_ts(ctx):
        frames = ctx['frames']
        assert len(frames) > idx, f"query frame index {idx} out of range, only {len(frames)} frames"
        return str(frames[idx]["timestamp_idx"]) if not from_zero else "0"
    return frame_ts

def frame_range_fn(from_zero=False):
    def frame_range(ctx):
        frames = ctx['frames']

        assert len(frames) > 0, "frame_range should have at least one frame"

        if from_zero:
            return f"0 ~ {len(frames) - 1}"
        return f"{frames[0]['timestamp_idx']} ~ {frames[-1]['timestamp_idx']}" 
    return frame_range

# ------------------------------ str generation ------------------------------ #

def minmax(min_prompt="minimal", max_prompt="maximal"):
    minmax = random.choice(['min', 'max'])
    def minmax_fn(ctx):
        ctx['minmax'] = minmax
        return f"{min_prompt if minmax == 'min' else max_prompt}"
    return minmax_fn