import numpy as np
from pybaseutils import file_utils, image_utils
from collections import defaultdict


if __name__ == "__main__":
    inp_tensor = np.random.random(size=(10, 3, 224, 224))
    out_tensor = defaultdict(list)  # TODO  CPU模式逐个推理，比批量推理快
    for i in range(len(inp_tensor)):
        out = [np.mean()]
        for k in range(len(out)):
            out_tensor[k].append(out[k])
    out_tensor = [np.concatenate(v, axis=0) for k, v in out_tensor.items()]