from .options import BaseOptions

class TestOptions(BaseOptions):

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--pos_number', type=int, default=4, help='position')
        parser.add_argument('--use_sp', action='store_true', help='use spetral normalization')
        parser.add_argument('--test_num', type=int, default=300, help='the number of test samples')
        parser.add_argument('--n_att', type=float, default=4, help='number of attribute')
        parser.add_argument('--crop_n', type=int, default=8, help='numbers for crops')
        
        self.isTrain = False
        return parser

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain
        self.print_options(opt)
        self.opt = opt

        return self.opt