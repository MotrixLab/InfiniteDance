import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader  
    parser.add_argument('--dataname', type=str, default='all', help='dataset directory')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--window_size', type=int, default=128, help='training motion length')

    ## optimization
    parser.add_argument('--total-iter', default=300000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm_up_iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default= 0.0002, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[200000], nargs="+", type=int, help="learning rate schedule (iterations)")    
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=1, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss_vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l2', help='reconstruction loss')
    
    ## vqvae arch
    parser.add_argument("--code_dim", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=1024, help="nb of embedding")#change 512——>1024
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=1024, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation_growth_rate", type=int, default=4, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=1024, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq_norm', type=str, default=None, help='dataset directory')
    
    parser.add_argument('--num_quantizers', type=int, default=1, help='num_quantizers')
    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume_pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume_gpt", type=str, default=None, help='resume pth for GPT')
    parser.add_argument("--nb_joints", type=int, default=22, help='')

    ##weights
    parser.add_argument("--w_motion", type=float, default=1, help='')
    parser.add_argument("--w_joints", type=float, default=0.02, help='')
    parser.add_argument("--w_vel", type=float, default=0.02, help='')
    parser.add_argument("--w_acc", type=float, default=0.02, help='')
    parser.add_argument("--w_foot", type=float, default=0.02, help='')

    
    ## output directory 
    parser.add_argument('--out_dir', type=str, default='DanceVQVAE/output', help='output directory')

    parser.add_argument('--exp_name', type=str, default='exp20250131', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print_iter', default=500, type=int, help='print frequency')
    parser.add_argument('--eval_iter', default=500, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis_gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb_vis', default=20, type=int, help='nb of visualizations')
    # Whether to train an additional model for velocity decoder
    # If True, an additional velocity decoder model will be trained.
    # If False, the velocity decoder is not trained.
    # Note: These two options (vel_decoder and rot_decoder) cannot be True at the same time.
    parser.add_argument('--vel_decoder', default=False)
    parser.add_argument('--rot_decoder', default=False)

    
    
    #-----------------------------------------------InfiniteDanceData-----------------------------------------------
    parser.add_argument("--split_file_train", type=str, default="InfiniteDanceData/partition/All_train.txt")
    parser.add_argument("--split_file_eval", type=str, default="InfiniteDanceData/partition/All_eval.txt")
    parser.add_argument("--data_root", type=str, default="All_mofea/FineDance")
    parser.add_argument("--meta_dir", type=str, default="All_mofea/FineDance")
    #-----------------------------------------------finedance-----------------------------------------------
    



    #-----------------------------------------------finedance-----------------------------------------------
    # parser.add_argument("--split_file_train", type=str, default="InfiniteDanceData/partition/finedance_train.txt")
    # parser.add_argument("--split_file_eval", type=str, default="InfiniteDanceData/partition/finedance_eval.txt")
    # parser.add_argument("--data_root", type=str, default="All_mofea/FineDance")
    #-----------------------------------------------finedance-----------------------------------------------
    
    
    return parser.parse_args()

