from .options import BaseOptions

class TrainOptions(BaseOptions):

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_model_freq', type=int, default=10000, help='frequency of saving checkpoints')
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=100000, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=50000, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_d', type=float, default=2e-3, help='initial learning rate for Adam in d')
        parser.add_argument('--lr_g', type=float, default=2e-3, help='initial learning rate for Adam in g')
        parser.add_argument('--lr_co', type=float, default=1e-4, help='initial learning rate from adam in co-discriminator')
        parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam')
        parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for adam')
        parser.add_argument('--loss_type', type=str, default='softplus',
                            choices=['gan', 'hinge', 'wgan_gp', 'lsgan', 'softplus'], help='using type of gan loss')
        parser.add_argument('--loss_type2', type=str, default='lsgan',
                            choices=['gan', 'hinge', 'wgan_gp', 'lsgan', 'softplus'], help='using type of gan loss')
        parser.add_argument('--gp_type', type=str, default='R1_regu', choices=['Dirac', 'wgan-gp', 'R1_regu'], help='gp type')
        parser.add_argument('--lam_gp_d', type=float, default=10.0, help='weight for gradient penalty of d')
        parser.add_argument('--lam_gp_co', type=float, default=1.0, help='wegiht for gradient penalty of co d')
        parser.add_argument('--pos_number', type=int, default=4, help='position')
        parser.add_argument('--test_num', type=int, default=300, help='the number of test samples')
        parser.add_argument('--crop_n', type=int, default=8, help='numbers for crops')
        parser.add_argument('--d_reg_every', type=int, default=1, help='l1 reg optimization every d')
        parser.add_argument('--g_reg_every', type=int, default=1, help='l1 reg optimization every g')

        self.isTrain = True
        return parser

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain
        self.print_options(opt)
        self.opt = opt

        return self.opt