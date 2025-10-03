import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM

from data_provider.failure_data_factory import data_provider
import time
import random
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM for Failure Prediction')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear, TimeLLM]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='google_failure_data.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S: univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='failure', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=12, help='start token length')
parser.add_argument('--pred_len', type=int, default=3, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='BCE', help='loss function: MSE, BCE (Binary Cross Entropy)')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--use_deepspeed', action='store_true', help='enable DeepSpeed (default: off for local runs)')

# output saving
parser.add_argument('--save_preds_csv', action='store_true', help='save predictions to CSV')
parser.add_argument('--save_plot_png', action='store_true', help='save an example plot to PNG')
parser.add_argument('--output_dir', type=str, default='./outputs-failure', help='directory to save predictions/plots/metrics')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary classification')

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json') if args.use_deepspeed else None
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    # Loss function 선택
    if args.loss == 'MSE':
        criterion = nn.MSELoss()
    elif args.loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits
    else:
        criterion = nn.MSELoss()  # 기본값
    
    # 평가 지표
    mae_metric = nn.L1Loss()

    # Only prepare model/optimizer/scheduler to avoid Accelerate auto-moving
    # DataLoaders to device (which can push float64 to MPS and crash).
    model, model_optim, scheduler = accelerator.prepare(
        model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    
                    # NaN 값 처리
                    outputs = torch.nan_to_num(outputs, nan=0.0)
                    batch_y = torch.nan_to_num(batch_y, nan=0.0)
                    
                    loss = criterion(outputs, batch_y)
                    
                    # NaN 손실이 발생하면 작은 값으로 대체
                    if torch.isnan(loss):
                        print("NaN 손실 발견, 기본값으로 대체")
                        loss = torch.tensor(1e-5, device=accelerator.device)
                    
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                
                # NaN 값 처리
                outputs = torch.nan_to_num(outputs, nan=0.0)
                batch_y = torch.nan_to_num(batch_y, nan=0.0)
                
                loss = criterion(outputs, batch_y)
                
                # NaN 손실이 발생하면 작은 값으로 대체
                if torch.isnan(loss):
                    print("NaN 손실 발견, 기본값으로 대체")
                    loss = torch.tensor(1e-5, device=accelerator.device)
                
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            try:
                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()
            except Exception as e:
                print(f"역전파 중 오류 발생: {e}")
                # 오류 발생 시 이 배치 건너뛰기

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    # after training, optionally run a test pass to save predictions/plots
    if args.save_preds_csv or args.save_plot_png:
        all_preds = []
        all_trues = []
        first_batch_true_hist = None
        first_batch_pred = None
        first_batch_true_future = None

        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(accelerator.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    accelerator.device)

                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y_future = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

                # gather across processes
                outputs, batch_y_future = accelerator.gather_for_metrics((outputs, batch_y_future))

                if accelerator.is_local_main_process:
                    all_preds.append(outputs.detach().cpu().numpy())
                    all_trues.append(batch_y_future.detach().cpu().numpy())

                    if first_batch_pred is None:
                        # also keep history for plotting from the current local batch before gather
                        # rebuild a local copy for the first local batch example
                        first_batch_true_hist = batch_y[:1, :args.label_len, f_dim:].detach().cpu().numpy()
                        first_batch_true_future = batch_y_future[:1].detach().cpu().numpy()
                        first_batch_pred = outputs[:1].detach().cpu().numpy()

        if accelerator.is_local_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            file_prefix = f"{args.model_id}_{args.model_comment}"

            # save metrics and predictions CSV
            if len(all_preds) > 0:
                preds = np.concatenate(all_preds, axis=0)
                trues = np.concatenate(all_trues, axis=0)

                # 이진 분류 지표 계산
                if args.loss == 'BCE':
                    # Sigmoid 적용 (BCEWithLogitsLoss를 사용했기 때문에)
                    preds_prob = 1 / (1 + np.exp(-preds))
                    preds_binary = (preds_prob > args.threshold).astype(int)
                    trues_binary = (trues > args.threshold).astype(int)
                    
                    accuracy = accuracy_score(trues_binary.flatten(), preds_binary.flatten())
                    precision = precision_score(trues_binary.flatten(), preds_binary.flatten(), zero_division=0)
                    recall = recall_score(trues_binary.flatten(), preds_binary.flatten(), zero_division=0)
                    f1 = f1_score(trues_binary.flatten(), preds_binary.flatten(), zero_division=0)
                    
                    try:
                        auc = roc_auc_score(trues_binary.flatten(), preds_prob.flatten())
                    except:
                        auc = 0
                    
                    conf_matrix = confusion_matrix(trues_binary.flatten(), preds_binary.flatten())
                    
                    # 결과 저장
                    metrics_path = os.path.join(args.output_dir, f"metrics_{file_prefix}.csv")
                    pd.DataFrame([
                        {"metric": "Accuracy", "value": accuracy},
                        {"metric": "Precision", "value": precision},
                        {"metric": "Recall", "value": recall},
                        {"metric": "F1", "value": f1},
                        {"metric": "AUC", "value": auc},
                    ]).to_csv(metrics_path, index=False)
                    
                    # 혼동 행렬 저장
                    confusion_path = os.path.join(args.output_dir, f"confusion_{file_prefix}.csv")
                    pd.DataFrame(conf_matrix).to_csv(confusion_path, index=False)
                    
                    print(f"분류 성능 지표: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                    print(f"혼동 행렬:\n{conf_matrix}")
                else:
                    # 회귀 지표
                    mse = float(np.mean((preds - trues) ** 2))
                    mae = float(np.mean(np.abs(preds - trues)))
                    metrics_path = os.path.join(args.output_dir, f"metrics_{file_prefix}.csv")
                    pd.DataFrame([
                        {"metric": "MSE", "value": mse},
                        {"metric": "MAE", "value": mae},
                    ]).to_csv(metrics_path, index=False)

                if args.save_preds_csv:
                    # flatten to long format: sample, step, pred, true
                    num_samples, horizon, num_channels = preds.shape
                    rows = []
                    for s in range(num_samples):
                        for t in range(horizon):
                            # for MS, num_channels==1; for others, we serialize channel 0
                            if args.loss == 'BCE':
                                rows.append({
                                    "sample": s,
                                    "step": t,
                                    "pred_prob": float(preds_prob[s, t, 0]),
                                    "pred_binary": int(preds_binary[s, t, 0]),
                                    "true_binary": int(trues_binary[s, t, 0])
                                })
                            else:
                                rows.append({
                                    "sample": s,
                                    "step": t,
                                    "pred": float(preds[s, t, 0]),
                                    "true": float(trues[s, t, 0])
                                })
                    preds_path = os.path.join(args.output_dir, f"predictions_{file_prefix}.csv")
                    pd.DataFrame(rows).to_csv(preds_path, index=False)

            # save a quick plot for the first example
            if args.save_plot_png and first_batch_pred is not None:
                plt.figure(figsize=(9, 4))
                # history then future truth
                hist = first_batch_true_hist[0, :, 0] if first_batch_true_hist is not None else None
                fut_true = first_batch_true_future[0, :, 0]
                
                if args.loss == 'BCE':
                    # Sigmoid 적용
                    fut_pred_prob = 1 / (1 + np.exp(-first_batch_pred[0, :, 0]))
                    
                    if hist is not None:
                        plt.plot(range(len(hist)), hist, label='history (label_len)', color='#888888')
                        offset = len(hist)
                    else:
                        offset = 0
                    
                    plt.plot(range(offset, offset + len(fut_true)), fut_true, label='true (future)', color='#1f77b4')
                    plt.plot(range(offset, offset + len(fut_pred_prob)), fut_pred_prob, label='pred probability', color='#d62728')
                    plt.axhline(y=args.threshold, color='green', linestyle='--', linewidth=0.8, label='threshold')
                else:
                    if hist is not None:
                        plt.plot(range(len(hist)), hist, label='history (label_len)', color='#888888')
                        offset = len(hist)
                    else:
                        offset = 0
                    
                    plt.plot(range(offset, offset + len(fut_true)), fut_true, label='true (future)', color='#1f77b4')
                    plt.plot(range(offset, offset + len(first_batch_pred[0, :, 0])), first_batch_pred[0, :, 0], label='pred', color='#d62728')
                
                plt.axvline(x=offset - 1, color='k', linestyle='--', linewidth=0.8)
                plt.title(f"{args.data} {args.features} | {args.model} | Failure Prediction")
                plt.legend()
                plt.tight_layout()
                plot_path = os.path.join(args.output_dir, f"plot_{file_prefix}.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'  # unique checkpoint saving path
    del_files(path)  # delete checkpoint files
    accelerator.print('success delete checkpoints')
