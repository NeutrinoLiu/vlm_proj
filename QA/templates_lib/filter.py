def filter_area(anno, thres=10_000):
    return anno['2d_crop']["area"] > thres

def filter_visiblity(anno, thres=100):
    vis = anno['visibility'].split("-")[-1]
    return int(vis) >= thres

def filter_multiple_fn(filters:list):
    def filter_fn(anno):
        for f in filters:
            if not f(anno):
                return False
        return True
    return filter_fn