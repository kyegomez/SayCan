[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# SayCan
Pytorch implementation of the model from "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances",

## Installation

You can install the package using pip

```bash
pip install saycan
```

------

## Usage
```
import torch
from saycan.model import SayCan

model = SayCan().cuda()

x = torch.randint(0, 256, (1, 1024)).cuda()

model(x) # (1, 1024, 20000)

```
----

# License
MIT

----

# Citation
```latex
@misc{2204.01691,
Author = {Michael Ahn and Anthony Brohan and Noah Brown and Yevgen Chebotar and Omar Cortes and Byron David and Chelsea Finn and Chuyuan Fu and Keerthana Gopalakrishnan and Karol Hausman and Alex Herzog and Daniel Ho and Jasmine Hsu and Julian Ibarz and Brian Ichter and Alex Irpan and Eric Jang and Rosario Jauregui Ruano and Kyle Jeffrey and Sally Jesmonth and Nikhil J Joshi and Ryan Julian and Dmitry Kalashnikov and Yuheng Kuang and Kuang-Huei Lee and Sergey Levine and Yao Lu and Linda Luu and Carolina Parada and Peter Pastor and Jornell Quiambao and Kanishka Rao and Jarek Rettinghouse and Diego Reyes and Pierre Sermanet and Nicolas Sievers and Clayton Tan and Alexander Toshev and Vincent Vanhoucke and Fei Xia and Ted Xiao and Peng Xu and Sichun Xu and Mengyuan Yan and Andy Zeng},
Title = {Do As I Can, Not As I Say: Grounding Language in Robotic Affordances},
Year = {2022},
Eprint = {arXiv:2204.01691},
}
```