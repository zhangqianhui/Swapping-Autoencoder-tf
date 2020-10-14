from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from Dataset import CelebA
from SwapAutoEncoderAdaIN import SAE
from config.train_options import TrainOptions
import setproctitle
setproctitle.setproctitle("SAE")

opt = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":

    dataset = CelebA(opt)
    sae = SAE(dataset, opt)
    sae.build_model()
    sae.train()