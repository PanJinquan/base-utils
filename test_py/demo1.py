import numpy as np
from pybaseutils import file_utils, image_utils, yaml_utils
from collections import defaultdict

if __name__ == "__main__":
    data={"content": "123"}
    print(data.pop("content"))
    print(data.pop("content","DAta"))

