import os
import sys
import time
import gc
import torch
import argparse
from Transformer.exp.exp_pred import Exp_pred
from Transformer.exp.exp_mae import Exp_mae

from utils.data.stock_data_handle import Stock_Data
from utils import config
from utils.config import TRANSFORMER_PARAMS_DEFAULT
from utils.config import TRANSFORMER_PARAMS_PRED_SHORT, TRANSFORMER_PARAMS_PRED_LONG, TRANSFORMER_PARAMS_MAE
import utils.tools as tools


# 创建一个简单的类来模拟 argparse 命名空间
class Args:
    def __init__(self, param_dict):
        for key, value in param_dict.items():
            setattr(self, key, value)
    
    def __str__(self):
        attrs = []
        for key, value in self.__dict__.items():
            attrs.append(f"{key}={repr(value)}")
        return f"Args({', '.join(attrs)})"
    
    def __repr__(self):
        return self.__str__()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose transformer parameters')
    parser.add_argument('--params_type', type=int, default=2, choices=[0, 1, 2], 
                        help='Type of transformer parameters: 0 for PRED_SHORT, 1 for PRED_LONG, 2 for MAE')
    args_parsed = parser.parse_args()

    config.set_seed()  # 使用 config 中的 set_seed 函数设置随机种子，确保全局一致性

    # 根据命令行参数选择对应的参数配置
    if args_parsed.params_type == 0:
        TRANSFORMER_PARAMS_TARGET = TRANSFORMER_PARAMS_PRED_SHORT
    elif args_parsed.params_type == 1:
        TRANSFORMER_PARAMS_TARGET = TRANSFORMER_PARAMS_PRED_LONG
    else:  # 默认为2，即MAE
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
    
    print(f"Loading {args.exp_type} dataset...")
    data =  data_type_dict[args.data_type](
            full_stock_path=args.full_stock_path,
            temporal_len=args.seq_len,
            prediction_len=[args.short_term_len, args.long_term_len],
            exp_type=args.exp_type,
            a=config.SCALE_A
            )
    
    # Transformer 训练不需要 full_df，手动释放以腾出系统内存
    if hasattr(data, 'full_df'):
        del data.full_df
    
    gc.collect()

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

        # 迭代结束，显式清理显存和对象，防止内存持续上涨
        del exp
        if args.use_gpu:
            torch.cuda.empty_cache()
        gc.collect()