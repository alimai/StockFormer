from cgi import test
from torch.utils.data.dataset import Dataset
from utils.data.stock_data_handle import Stock_Data, DatasetStock_PRED
from Transformer.exp.exp_basic import Exp_Basic
from Transformer.models.transformer import Transformer_base as Transformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, ranking_loss
import utils.tools as utils
import utils.metrics_object as metrics_object

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
#import pdb

import os
import time

dataset_dict = {
    'stock': DatasetStock_PRED,
}

class Exp_pred(Exp_Basic):
    def __init__(self, args, data_all, id):
        super(Exp_pred, self).__init__(args)
        log_dir = os.path.join('log', 'pred_'+args.project_name+'_'+str(args.rank_alpha)+'_'+id)
        print(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.data_all = data_all
    
    def _build_model(self):
        model_dict = {
            'Transformer':Transformer,
        }

        if self.args.model=='Transformer':
            model = model_dict[self.args.model](
                # self.args
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.activation
                # self.device
            )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model.float()

    def _get_data(self, flag):
        args = self.args

        if flag == 'train':
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size
        else:
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
      
        dataset = dataset_dict[self.args.data_type](self.data_all, type=flag, pred_type=self.args.pred_type)
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, metric_builders, stage='test'):
        self.model.eval()
        total_loss = []
        metric_objs = [builder(stage) for builder in metric_builders]
        
        # 平衡模式：提升子批次大小到 6000 (约 20 天数据)
        sub_batch_size = 6000

        with torch.no_grad():
            for i, (batch_x1, batch_x2, batch_y) in enumerate(vali_loader):
                bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
                # 数据保持在 CPU
                batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float()
                batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float()
                batch_y_gpu = batch_y.float().to(self.device)
                
                outputs = []
                num_samples = batch_x1.shape[0]
                for start_idx in range(0, num_samples, sub_batch_size):
                    end_idx = min(start_idx + sub_batch_size, num_samples)
                    sub_x1 = batch_x1[start_idx:end_idx].to(self.device)
                    sub_x2 = batch_x2[start_idx:end_idx].to(self.device)
                    
                    with torch.amp.autocast('cuda'):
                        _, _, sub_out = self.model(sub_x1, sub_x2)
                    outputs.append(sub_out.detach().cpu())
                    del sub_x1, sub_x2, sub_out
                
                output = torch.cat(outputs, dim=0).to(self.device).reshape(bs, stock_num).float()
                loss = criterion(output, batch_y_gpu) + self.args.rank_alpha * ranking_loss(output, batch_y_gpu)

                total_loss.append(loss.item())

                for metric in metric_objs:
                    metric.update(output, batch_y_gpu)
                
                del batch_x1, batch_x2, batch_y_gpu, output, outputs

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, metric_objs
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'valid')
        test_data, test_loader = self._get_data(flag = 'test')

        metrics_builders = [
            metrics_object.MIRRTop1,
        ]

        path = os.path.join('./checkpoints/',setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        scaler = torch.amp.GradScaler('cuda')
        metric_objs = [builder('train') for builder in metrics_builders]

        valid_loss_global = np.inf
        best_model_index = -1

        print(f"Starting training loop... ({train_steps} steps per epoch)")
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_accum = []
            
            self.model.train()
            for i, (batch_x1, batch_x2, batch_y) in enumerate(train_loader):
                iter_count += 1
                bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
                
                # 初始数据在 CPU
                batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float()
                batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float()
                batch_y_gpu = batch_y.float().to(self.device)

                model_optim.zero_grad()
                
                total_iter_loss = 0
                
                # 性能优化：每 16 天并行处理一次 (针对显存富余的平衡点)
                accum_days = 16
                for day_idx in range(0, bs, accum_days):
                    end_day = min(day_idx + accum_days, bs)
                    num_days = end_day - day_idx
                    
                    day_x1 = batch_x1[day_idx*stock_num : end_day*stock_num].to(self.device)
                    day_x2 = batch_x2[day_idx*stock_num : end_day*stock_num].to(self.device)
                    day_y = batch_y_gpu[day_idx:end_day] # [num_days, stock_num]
                    
                    with torch.amp.autocast('cuda'):
                        _, _, day_out = self.model(day_x1, day_x2)
                        day_out = day_out.reshape(num_days, stock_num).float()
                        
                        # 按比例缩放 Loss
                        loss_group = (criterion(day_out, day_y) + self.args.rank_alpha * ranking_loss(day_out, day_y)) * (num_days / bs)
                        
                    scaler.scale(loss_group).backward()
                    total_iter_loss += loss_group.item()
                    
                    del day_x1, day_x2, day_out
                
                scaler.step(model_optim)
                scaler.update()
                train_loss_accum.append(total_iter_loss)
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, total_iter_loss))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                    # 仅在日志输出时回收缓存，减少同步频率
                    torch.cuda.empty_cache()

                del batch_x1, batch_x2, batch_y_gpu

            train_loss = np.average(train_loss_accum)
            valid_loss, valid_metrics = self.vali(vali_data, vali_loader, criterion, metrics_builders, stage='valid')
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion, metrics_builders, stage='test')

            self.writer.add_scalar('Train/loss', train_loss, epoch)
            self.writer.add_scalar('Valid/loss', valid_loss, epoch)
            self.writer.add_scalar('Test/loss', test_loss, epoch)

            all_logs = {
                metric.name: metric.value for metric in valid_metrics + test_metrics
            }
            for name, value in all_logs.items():
                self.writer.add_scalar(name, value.mean(), global_step=epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Valid Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss, test_loss))
            
            torch.save(self.model.state_dict(), path+'/'+'checkpoint_{0}.pth'.format(epoch+1))

            if valid_loss.item() < valid_loss_global:
                best_model_index = epoch+1
                valid_loss_global = valid_loss.item()

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint_{0}.pth'.format(best_model_index)
        self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print('best model index: ', best_model_index)
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        
        # 子批次大小提升到 6000
        sub_batch_size = 6000

        metrics_builders = [
            metrics_object.MIRRTop1,
            metrics_object.RankIC
        ]
        metric_objs = [builder('test') for builder in metrics_builders]
        
        with torch.no_grad():
            for i, (batch_x1, batch_x2, batch_y) in enumerate(test_loader):
                bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
                batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float()
                batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float()
                batch_y_gpu = batch_y.float().to(self.device)

                outputs = []
                num_samples = batch_x1.shape[0]
                for start_idx in range(0, num_samples, sub_batch_size):
                    end_idx = min(start_idx + sub_batch_size, num_samples)
                    sub_x1 = batch_x1[start_idx:end_idx].to(self.device)
                    sub_x2 = batch_x2[start_idx:end_idx].to(self.device)
                    
                    with torch.amp.autocast('cuda'):
                        _, _, sub_out = self.model(sub_x1, sub_x2)
                    outputs.append(sub_out.detach().cpu())
                    del sub_x1, sub_x2, sub_out

                output = torch.cat(outputs, dim=0).to(self.device).reshape(bs, stock_num).float()

                for metric in metric_objs:
                    metric.update(output, batch_y_gpu)
                
                del batch_x1, batch_x2, batch_y_gpu, output, outputs

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        all_logs = {
                metric.name: metric.value for metric in metric_objs
            }
        for name, value in all_logs.items():
            print(name, value.mean())

        return 0
