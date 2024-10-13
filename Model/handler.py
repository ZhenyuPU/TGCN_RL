import numpy as np
import pandas as pd
from Utils.HIES_options import HIES_Options
import os
import json
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import time
from Utils.prediction_utils import Args
from Utils.data_loader.forecast_dataloader import ForecastDataset, de_normalized
import matplotlib.pyplot as plt
from Utils.prediction_utils import evaluate
from datetime import datetime


def plot_(arr, result_file):
    plt.plot(arr, 'k-')
    save_path = os.path.join(result_file, 'fig')
    os.makedirs(save_path, exist_ok=True)               # 确保目录存在
    plt.savefig(os.path.join(save_path, 'training_loss.png'))
    plt.close()                                         # 关闭图形以释放内存

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)

def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


# Sliding window translation prediction within interval
def slide_pred_within_interval(model, step, horizon, window_size, inputs, forecast_steps):
    while step < horizon:
        forecast_result = model(inputs)
        len_model_output = forecast_result.size()[1]    # 获取时间步数
        # 检查是否获得有效的输出，没有就提出异常情况
        if len_model_output == 0:
            raise Exception('Get blank inference result')
        # 输出模型在执行预测之前的几步，该步数将随着预测的前进而不断向前滚动，相当于用于预测的数据，但是要一直向前滚动
        inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                            :].clone() 
        # 预测的数据，保证结果都在window_size尺度内，平移inputs
        inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
        # 将预测结果存入forecast_steps直到达到指定的horizon
        forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
            forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
        step += min(horizon - step, len_model_output)

        return forecast_steps

# 模型推理得到预测结果
def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()   # 将模型设置为评估模式，关闭Dropout和BatchNorm等与训练相关的特性，确保推理时的行为一致
    with torch.no_grad():
        for data in dataloader:
            if len(data) == 1:
                inputs = data.to(device)
                step = 0
                forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=float)
                forecast_steps = slide_pred_within_interval(model, step, horizon, window_size, inputs, forecast_steps)
                return forecast_steps, None
            else:
                inputs, target = data
                inputs = inputs.to(device)
                target = target.to(device)
                step = 0
                forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=float)   # 形成空数组，[batch size, horizon, node_cnt]
                # 多步预测，每次预测一部分未来，但是如果出现预测的步数少于要求预测的步数horizon，那么就进行结合预测值的滚动预测
                # 其实是当预测步长小于horizon的时候这么干，因为我要让预测出来的长度与horizon一样，不一样的话需要反复用同一组(inputs, forecast_result)去迭代获取直到预测长度=horizon   滑动窗口平移预测，与dataloader无关，这个迭代是让预测的所有内容都是根据interval间隔的，这个预测所有内容要算上所有batch size的，上一个只是一个窗口
                forecast_steps = slide_pred_within_interval(model, step, horizon, window_size, inputs, forecast_steps)
                forecast_set.append(forecast_steps)
                target_set.append(target.detach().cpu().numpy())
            # print(np.array(forecast_set).shape)             # (32, 3, 140) * 31(The last batch isn't full)
            # print(np.concatenate(forecast_set, axis=0).shape)    # (986, 3, 140)
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)



def validate(model, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon,
             result_file=None):
    start = datetime.now()
    # 返回归一化后的预测值和目标值
    forecast_norm, target_norm = inference(model, dataloader, device,
                                           node_cnt, window_size, horizon)
    # 判断是否需要反归一化
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic)
        target = de_normalized(target_norm, normalize_method, statistic)
    else:
        forecast, target = forecast_norm, target_norm
    score = evaluate(target, forecast)                              # 评估
    score_by_node = evaluate(target, forecast, by_node=True)        # 评估每一个结点的，如果适用
    end = datetime.now()

    score_norm = evaluate(target_norm, forecast_norm)
    print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    # 如果result_file路径存在，则检查是否存在该目录，不存在则创建
    # if result_file:
    #     if not os.path.exists(result_file):
    #         os.makedirs(result_file)
    #     step_to_print = 0                                 # 选择要保存的时间步，这里选择第一个时间步
    #     forcasting_2d = forecast[:, step_to_print, :]
    #     forcasting_2d_target = target[:, step_to_print, :]
    #     # 将预测结果和目标结果保存并求出误差
    #     np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
    #     np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
    #     np.savetxt(f'{result_file}/predict_abs_error.csv',
    #                np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
    #     np.savetxt(f'{result_file}/predict_ape.csv',
    #                np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

    return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                rmse=score[2], rmse_node=score_by_node[2])






def prediction_train(train_data, valid_data, model, result_file, args):
    node_cnt = train_data.shape[1]
    model.to(args.device)
    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')
    # 选择归一化方法
    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    # save statistic information as a json file
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)
    # Select optimizer
    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    # Construct training set and validating set
    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                            normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                        num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)
    # Calculate the total number of trainable parameters
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")

    # Start traning
    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    loss_arr = []
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            model.zero_grad()
            forecast = model(inputs)
            # print(forecast.shape, target.shape)
            loss = forecast_loss(forecast, target)
            cnt += 1
            loss.backward()
            my_optim.step()      # Optimizer
            loss_total += float(loss)
        loss_arr.append(loss_total)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt))
        # save_model(model, result_file, epoch)
        # 按照预定步长进行学习率衰减
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()                              # 调用学习率调度器调整学习率
        # 调用学习率调度器调整学习率
        # 按照制定频率在验证集进行验证，validate each epoch 
        # 目的在于防止性能下降
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            # 调用validate函数进行验证
            performance_metrics = \
                validate(model, valid_loader, args.device, args.norm_method, normalize_statistic,
                        node_cnt, args.window_size, args.horizon,
                        result_file=result_file)
            # 将当前性能与历史最佳验证性能进行比较
            if best_validate_mae > performance_metrics['mae']:
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
            # if best for now, save model
            if is_best_for_now:
                save_model(model, result_file)
        # early stop 当验证集性能不再提高且连续次数达到指定次数的时候，触发早停机制，提前结束训练，防止在训练集上过拟合导致在验证集上性能下降太多
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    
    # plot_(loss_arr, result_file)
    print('-------------Finish Training--------------------')
    return performance_metrics, normalize_statistic




def test(test_data, args, result_train_file, result_test_file):
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)
    model = load_model(result_train_file)
    node_cnt = test_data.shape[1]
    test_set = ForecastDataset(test_data, window_size=args.window_size, horizon=args.horizon,
                               normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    performance_metrics = validate(model, test_loader, args.device, args.norm_method, normalize_statistic,
                      node_cnt, args.window_size, args.horizon,
                      result_file=result_test_file)
    mae, mape, rmse, mae_node, mape_node, rmse_node = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse'], performance_metrics['mae_node'], performance_metrics['mape_node'], performance_metrics['rmse_node']
    print('Performance on test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))
    node    =   ['Electricity demand', 'Heating demand', 'Cooling demand', 'PV power', 'Electricity price']
    perf_metrics    =   ['MAE', 'MAPE', 'RMSE']
    with open(f'{result_test_file}/performance.txt', 'w') as file:
        file.write(f'The total performance metrics:\n')
        file.write(f'MAE: {mae}\n')
        file.write(f'MAPE: {mape}\n')
        file.write(f'RMSE: {rmse}\n')

        for i, node_name in enumerate(node):
            file.write(f'       {node_name}    ')
        file.write('\n')
        file.write(f'MAE:  ')
        for value in mae_node:
            file.write(f'{value}   ')  
        file.write('\n') 

        file.write(f'MAPE: ')
        for value in mape_node:
            file.write(f'{value}   ')  
        file.write('\n')  

        file.write(f'RMSE: ')
        for value in rmse_node:
            file.write(f'{value}   ')  
        file.write('\n')   


# 获取结果数据集，model(train) + model(test)，利用inference函数
def get_results(data, args, result_train_file, result_forecast_file, batch_size, horizon):
    # load models
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)
    model = load_model(result_train_file)
    # load data
    node_cnt    =   data.shape[1]
    data_set    =   ForecastDataset(data, window_size=args.window_size, horizon=args.horizon,
                                    normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    data_loader =   torch_data.DataLoader(data_set, batch_size=batch_size, drop_last=False,
                                          shuffle=False, num_workers=0)

    # inference
    forecast_norm, _ =   inference(model, dataloader=data_loader, device=args.device, node_cnt=node_cnt,
                              window_size=args.window_size, horizon=horizon)
    # 判断是否需要反归一化
    if args.norm_method and normalize_statistic:
        forecast = de_normalized(forecast_norm, args.norm_method, normalize_statistic)
    else:
        forecast = forecast_norm
    step_to_print = 0
    forecasting_2d = forecast if data.shape[0] == args.window_size else forecast[:, step_to_print, :]
    # np.savetxt()
    if data.shape[0] > args.window_size:
        os.makedirs(result_forecast_file, exist_ok=True)               # 确保目录存在
        np.savetxt(os.path.join(result_forecast_file, 'forecast.csv'), forecasting_2d, delimiter=',')
    else:
        # Return forecasting data in the prediction horizon
        return forecasting_2d
