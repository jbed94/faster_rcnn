# Faster-R-CNN

## How to use

1. Clone, or add submodule to your project,
2. Import from faster_rcnn package,
3. Create instance of FasterRCNN class.

```
from .faster_rcnn import FasterRCNN

model = FasterRCNN(**FasterRCNN.std_spec(10, False))
output_dict = model(...)
```
