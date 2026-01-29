import os
import sys
import argparse
from Transformer.exp.exp_pred import Exp_pred
from Transformer.exp.exp_mae import Exp_mae

from utils.data.stock_data_handle import Stock_Data
from utils.config import fix_seed
from utils.config import TRANSFORMER_PARAMS_DEFAULT
from utils.config import TRANSFORMER_PARAMS_PRED_SHORT, TRANSFORMER_PARAMS_PRED_LONG, TRANSFORMER_PARAMS_MAE
import utils.tools as tools

import time
import pdb
import random
import torch
import numpy as np

# 创建一个简单的类来模拟 argparse 命名空间
class Args:
    def __init__(self, param_dict):
        for key, value in param_dict.items():
            setattr(self, key, value)

working_path = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到模块搜索路径
# sys.path.insert(0, working_path)

if __name__ == '__main__':
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    #TRANSFORMER_PARAMS_MAE,TRANSFORMER_PARAMS_PRED_SHORT, TRANSFORMER_PARAMS_PRED_LONG
    TRANSFORMER_PARAMS_TARGET = TRANSFORMER_PARAMS_MAE
    # 先用TRANSFORMER_PARAMS_DEFAULT赋默认值，然后用TRANSFORMER_PARAMS_TARGET覆盖相应的值
    args = Args({**TRANSFORMER_PARAMS_DEFAULT, **TRANSFORMER_PARAMS_TARGET})
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print(args)

    exp_dict = {'pred': Exp_pred, 'mae': Exp_mae}
    data_type_dict = {'stock': Stock_Data}
    Exp = exp_dict[args.exp_type]
    data =  data_type_dict[args.data_type](
            root_path=args.root_path,
            dataset_name=args.data_name,
            full_stock_path=args.full_stock_path,
            size=[args.seq_len, args.label_len, args.pred_len],
            prediction_len=[args.short_term_len, args.long_term_len]
            )

    # pdb.set_trace()

    for ii in range(args.itr):
        id = tools.generate_id()
        setting = '{}_{}_{}_alpha{}_sl{}_pl{}_enc{}_cout{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}_dt{}_id{}'.format(args.exp_type, args.project_name, args.data_name, str(args.rank_alpha).replace('.','_'),
                    args.seq_len, args.pred_len, args.enc_in, args.c_out,
                    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.des, ii, args.data_name, id)

        exp = Exp(args, data, id)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        print('Task id: ',id)
        start = time.time()
        exp.train(setting)
        end = time.time()
        print("Training Time:",end-start)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)