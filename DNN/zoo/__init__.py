import os.path as op
import sys

from .cornet_r import CORnet_R as cornet_r
from .cornet_rt import CORnet_RT as cornet_rt
from .cornet_rt_hw3 import CORnet_RT as cornet_rt_hw3
from .cornet_rt_output_avpool import CORnet_RT as cornet_rt_output_avpool
from .cornet_s import CORnet_S as cornet_s
from .cornet_s_V1 import CORnet_S as cornet_s_V1
from .cornet_s_V1_v2 import CORnet_S as cornet_s_V1_v2
from .cornet_s_V1_v3 import CORnet_S as cornet_s_V1_v3
from .cornet_s_V1_v4 import CORnet_S as cornet_s_V1_v4
from .cornet_s_V1_v5 import CORnet_S as cornet_s_V1_v5
from .cornet_s_V1_v6 import CORnet_S as cornet_s_V1_v6
from .cornet_s_unshared import CORnet_S as cornet_s_unshared
from .cornet_s_output_avpool import CORnet_S as cornet_s_output_avpool
from .cornet_s_cont import CORnet_S_cont as cornet_s_cont
from .cornet_s_custom import CORnet_S_custom as cornet_s_custom
from .cornet_st import CORnet_ST as cornet_st
from .cornet_flab import CORnet_FLaB as cornet_flab
from .cornet_z import CORnet_Z as cornet_z
from .cornet_s_hw7 import CORnet_S as cornet_s_hw7
from .cornet_s_hw3 import CORnet_S as cornet_s_hw3
from .cornet_s_hd2_hw3 import CORnet_S as cornet_s_hd2_hw3
#from .cornet_s_custom_predify import *
from .locCon1HL import *
from .cognet.cognet import CogNet as cognet
from .cognet.cognet_v2 import CogNet as cognet_v2
from .cognet.cognet_v3 import CogNet as cognet_v3
from .cognet.cognet_v4 import CogNet as cognet_v4
from .cognet.cognet_v5 import CogNet as cognet_v5
from .cognet.cognet_v6 import CogNet as cognet_v6
from .cognet.cognet_v7 import CogNet as cognet_v7
from .cognet.cognet_v8 import CogNet as cognet_v8
from .cognet.cognet_v9 import CogNet as cognet_v9
from .cognet.cognet_v10 import CogNet as cognet_v10
from .cognet.cognet_v11 import CogNet as cognet_v11
from .cognet.cognet_v12 import CogNet as cognet_v12
from .cognet.cognet_v13 import CogNet as cognet_v13
from .cognet.cognet_v14 import CogNet as cognet_v14
from .GaborFilterBank import GaborFilterBank
from .alexnet_V1 import AlexNet as alexnet_V1
#from . import segmentation
#from . import detection
#from . import video
#from . import quantization

sys.path.append(op.expanduser('~/david/repos/PredNet_pytorch'))
from prednet import PredNet as prednet
