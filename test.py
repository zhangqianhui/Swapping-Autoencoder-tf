from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from Dataset import CelebA
from SwapAutoEncoderAdaIN import SAE
from config.test_options import TestOptions

opt = TestOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":

    dataset = CelebA(opt)
    gaze_gan = SAE(dataset, opt)
    gaze_gan.build_test_model()
    gaze_gan.test()
