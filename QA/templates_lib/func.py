def anno_of_obj_from_frame(frame, obj):
    token = obj["instance_token"]
    for anno in frame["annos"]:
        if anno["instance_token"] == token:
            return anno
    return None

def obj_desc_fn(idx):
    def obj_desc(frames, objs):
        assert len(objs) > idx, f"query object index {idx} out of range, only {len(objs)} objects"
        return objs[idx]["category_name"].split(".")[-1]
    return obj_desc

def obj_cam_dist(frames, objs):
    assert len(objs) == 1, f"obj_cam_dist should only have one object, but got {len(objs)}"
    assert len(frames) == 1, f"obj_cam_dist should only have one frame, but got {len(frames)}"
    frame = frames[0]
    obj = objs[0]

    anno = anno_of_obj_from_frame(frame, obj)
    assert anno is not None, f"Object {obj['instance_token']} not found in frame {frame['sample_data_token']}"

    cam_t = frame["cam_t"]
    obj_t = anno["box_t"]
    cam_dist = ((cam_t[0] - obj_t[0]) ** 2 + (cam_t[1] - obj_t[1]) ** 2 + (cam_t[2] - obj_t[2]) ** 2) ** 0.5
    return f"{cam_dist:.2f}"

def obj_dist_between(frames, objs):
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
    dist = ((obj1_t[0] - obj2_t[0]) ** 2 + (obj1_t[1] - obj2_t[1]) ** 2 + (obj1_t[2] - obj2_t[2]) ** 2) ** 0.5
    return f"{dist:.2f}"

def frame_ts_fn(idx):
    def frame_ts(frames, objs):
        assert len(frames) > idx, f"query frame index {idx} out of range, only {len(frames)} frames"
        return str(frames[idx]["timestamp_idx"])
    return frame_ts