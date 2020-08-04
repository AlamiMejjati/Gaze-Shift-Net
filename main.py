"""Code for Attention-Guided Image editing"""
import os
from datetime import datetime
import argparse
import glob
from trainer import *
from models import *
from dataloader import *
from utils import print_options, str2bool
from get_data import *


def trainer(args, parser):
    log_dir = args.logdir
    checkpoint_dir = args.checkpoint_dir
    LAMBDAsal = args.LAMBDAsal
    LAMBDAFM = args.LAMBDAFM
    LAMBDAD = args.LAMBDAD
    LAMBDA_p = args.LAMBDA_p
    LAMBDA_s = args.LAMBDA_s
    LAMBDA_r = args.LAMBDA_r
    base_lr = args.lr
    max_step = args.maxstep
    salmodelpath = args.salmodelpath
    batch_size = args.batch_size
    random_s = args.random_s
    nb_blocks = args.nb_blocks

    if not args.checkpoint_dir:
        if 'withstyle' in args.trainer:
            log_dir = os.path.join(log_dir, args.D, args.dataloader, args.trainer, args.G,
                                   'Lsal_%.2f_Lp_%.3f_Lr_%.3f_ndf_%d_ngf_%d_lrg_%.5f_lrd_%.5f_donormG_%s_donormD_%s'
                                   % (LAMBDAsal, LAMBDA_p, LAMBDA_r, args.ndf, args.ngf, args.lr, args.lrd, str(args.donormG), str(args.donormD)),
                                   datetime.now().strftime("%Y%m%d-%H%M%S"))
        elif 'Ploss' in args.trainer:
            log_dir = os.path.join(log_dir, args.D, args.dataloader, args.trainer, args.G,
                                   'Lsal_%.2f_Lp_%.3f_ndf_%d_ngf_%d_lrg_%.5f_lrd_%.5f_donormG_%s_donormD_%s'
                                   % (LAMBDAsal, LAMBDA_p, args.ndf, args.ngf, args.lr, args.lrd, str(args.donormG), str(args.donormD)),
                                   datetime.now().strftime("%Y%m%d-%H%M%S"))
        elif 'neurons' in args.trainer:
            log_dir = os.path.join(log_dir, args.D, args.dataloader, args.trainer, args.G,
                                   'Lsal_%.2f_ndf_%d_ngf_%d_lrg_%.5f_lrd_%.5f_donormG_%s_donormD_%s_fcdim_%d_nbneurons_%d'
                                   %(LAMBDAsal, args.ndf, args.ngf, args.lr, args.lrd, str(args.donormG), str(args.donormD), args.fc_dim, args.nb_neurons),
                                   datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            log_dir = os.path.join(log_dir, args.D, args.dataloader, args.trainer, args.G,
                                   'Lsal_%.2f_ndf_%d_ngf_%d_lrg_%.5f_lrd_%.5f_donormG_%s_donormD_%s_fcdim_%d'
                                   %(LAMBDAsal, args.ndf, args.ngf, args.lr, args.lrd, str(args.donormG), str(args.donormD), args.fc_dim),
                                   datetime.now().strftime("%Y%m%d-%H%M%S"))

        if (not os.path.exists(log_dir)):
            os.makedirs(log_dir)
    else:
        log_dir = args.checkpoint_dir
    to_restore = checkpoint_dir
    print_freq = args.print_freq
    print_options(parser, args, log_dir, True)
    get_data_G = getattr(sys.modules[__name__], args.get_data_G)

    cocoiterator, max_images_coco = get_data_G(args.dataloader, os.path.join(args.data, 'maskdir'),
                                               args.path, True, batch_size, random_s, args.shape1, args.shape2,
                                               args.drop_remainder)

    Gmodel = getattr(sys.modules[__name__], args.G)
    Dmodel = getattr(sys.modules[__name__], args.D)
    model_trainer = getattr(sys.modules[__name__], args.trainer)
    model_trainer = model_trainer(Gmodel, Dmodel, salmodelpath, cocoiterator,
                                  print_freq, log_dir, to_restore, base_lr,
                                  max_step, checkpoint_dir, max_images_coco, batch_size, args)
    # model_trainer = model_trainer(Gmodel, Dmodel, salmodelpath, cocoiterator, print_freq, log_dir, to_restore, base_lr,
    #                               max_step, checkpoint_dir, max_images_coco, batch_size, args)
    if args.resave:
        model_trainer.resave()
    else:
        model_trainer.train()

    return log_dir

def tester(args, log_dir, test_id=0):
    checkpoint_dir = log_dir
    base_lr = args.lr
    max_step = args.maxstep
    salmodelpath = args.salmodelpath
    batch_size = args.batch_size
    n_scale = args.n_scale
    n_dis = args.n_dis
    ngf = args.ngf
    ndf = args.ndf
    nb_blocks = args.nb_blocks
    to_restore = log_dir
    print_freq = args.print_freq
    random_s = args.random_s

    get_data_G = getattr(sys.modules[__name__], args.get_data_G)

    cocoiterator, max_images_coco = get_data_G(args.dataloader, os.path.join(args.data, 'maskdir_val'),
                                               args.path, False, batch_size, random_s, args.shape1,
                                               args.shape2, args.drop_remainder, 0)


    Gmodel = getattr(sys.modules[__name__], args.G)
    Dmodel = getattr(sys.modules[__name__], args.D)

    model_trainer = getattr(sys.modules[__name__], args.trainer)
    model_trainer = model_trainer(Gmodel, Dmodel, salmodelpath, cocoiterator,
                                  print_freq, log_dir, to_restore, base_lr,
                                  max_step, checkpoint_dir, max_images_coco, batch_size, args)

    model_trainer.test()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data', required=True, help='directory containing mask annotation')
    parser.add_argument(
        '--path', required=True,
        help='the path that contains raw coco JPEG images')
    parser.add_argument('--traintest', type=int, default=1)
    parser.add_argument('--logdir', required=False, default='./logs')
    parser.add_argument('--checkpoint_dir', required=False, default=False)
    parser.add_argument('--salmodelpath', required=True)
    parser.add_argument('--LAMBDAsal', type=float, required=False, default=25000, help='lambda parameter for saliency loss')
    parser.add_argument('--LAMBDA_r', type=float, required=False, default=10, help='z reconstruction loss')
    parser.add_argument('--LAMBDAFM', type=float, required=False, default=10, help='lambda parameter for FM loss')
    parser.add_argument('--LAMBDAD', type=float, required=False, default=10, help='lambda parameter for critic loss')
    parser.add_argument('--LAMBDA_p', type=float, required=False, default=0.1, help='lambda parameter for perceptual loss')
    parser.add_argument('--LAMBDA_s', type=float, required=False, default=0.1, help='lambda parameter for sparse loss')
    parser.add_argument('--LAMBDA_tv', type=float, required=False, default=10, help='lambda parameter for TV regularization')
    parser.add_argument('--lr', type=float, required=False, default=0.00001)
    parser.add_argument('--lrd', type=float, required=False, default=0.00001)
    parser.add_argument('--maxstep', type=int, required=False, default=400)
    parser.add_argument('--nb_its', type=int, required=False, default=150000)
    parser.add_argument('--batch_size', type=int, required=False, default=4)
    parser.add_argument('--print_freq', type=int, required=False, default=500)
    parser.add_argument('--n_dis', type=int, default=3, help='number of discriminator layers')
    parser.add_argument('--n_scale', type=int, default=3, help='number of scales for discriminator')
    parser.add_argument('--ngf', type=int, default=64, help='number of channels in the generator')
    parser.add_argument('--ndf', type=int, default=64, help='number of channels in the discriminator')
    parser.add_argument('--nb_blocks', type=int, default=9, help='number of residual blocks for the generator')
    parser.add_argument('--G', default='increase', help='which generator to use')
    parser.add_argument('--D', default='MSD_global',  help='which discriminator to use')
    parser.add_argument('--D_dir', default='MSD_global',  help='which discriminator to use')
    parser.add_argument('--trainer', default='adv_increase_sftmx', help='which trainer to use')
    parser.add_argument('--get_data_G', default='get_data_coco', help='which trainer to use')
    parser.add_argument('--get_data_D', default='get_data_mix', help='which trainer to use')
    parser.add_argument('--dataset', default='coco', help='name of dataset used')
    parser.add_argument('--dataloader', default='CoCoLoader_rectangle', help='which dataloader to use')
    parser.add_argument('--dataloader_mix', default='Adobe5kCoCoLoader', help='which dataloader to use for D')
    parser.add_argument('--random_s', type=int, default=1, help='random s values or fixed')
    parser.add_argument('--hinge_lb', type=float, default=0.02, help='lower bound on hinge loss')
    parser.add_argument('--hinge_ub', type=float, default=0.1, help='upper bound on hinge loss')
    parser.add_argument('--interpolate', type=int, default=0, help='upper bound on hinge loss')
    parser.add_argument('--drop_remainder', type=int, default=1, help='drop remainder or not')
    parser.add_argument('--shape1', type=int, default=240, help='width of image')
    parser.add_argument('--shape2', type=int, default=320, help='height of image')
    parser.add_argument('--nb_gpu', type=int, default=1, help='number of gpus')
    parser.add_argument('--donormG', type=str2bool, default=True, help="Apply instance normalization in the generator")
    parser.add_argument('--donormD', type=str2bool, default=False, help="Apply instance normalization in the discriminator")
    parser.add_argument('--sl', type=float, default=0.3, help='tradeoff hyper-parameter')
    parser.add_argument('--zdim', type=int, default=10, help='length of latent variable z')
    parser.add_argument('--shuffle', type=int, default=0, help='shuffle or not')
    parser.add_argument('--video', type=int, default=0, help='use for video inference')
    parser.add_argument('--resave', type=str2bool, default=False, help='re-save the model')
    parser.add_argument('--zbinsize', type=int, default=11, help='the dimension of zbin')
    parser.add_argument('--startdecay', type=int, default=50, help='start decay at x% of total number of iterations')
    parser.add_argument('--nb_neurons', type=int, default=100, help='number of neurons to predict')
    parser.add_argument('--fc_dim', type=int, default=128, help='number of neurons in dense hidden layers')
    args = parser.parse_args()
    if args.traintest == 1:
        print('starting training')
        training_log = trainer(args, parser)
        print('DONE TRAINING')
        args.batch_size = 1
        args.nb_gpu = 1

        args.dataloader = 'CoCoLoader_rectangle_HR'
        args.get_data_G = 'get_data_coco'
        if args.resave ==False:
            tf.keras.backend.clear_session()
            print('START TESTING')
            tester(args, training_log)
            print('DONE TESTING')
    elif args.traintest == 0:
        print('starting training')
        trainer(args, parser)
        print('DONE TRAINING')
    elif args.traintest == 2:
        tf.keras.backend.clear_session()
        print('START TESTING')
        tester(args, args.checkpoint_dir)
        print('DONE TESTING')
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
