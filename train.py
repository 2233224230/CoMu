
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from models.CoMu import CoMu
from utils.data_loader import ConvECorpus
from utils.data_util import load_data
import os
import datetime

def parse_args():
    config_args = {
        'lr': 0.0005,
        'cuda': 0,
        'epochs': 2000,
        'weight_decay': 0,
        'seed': 2025,
        'model': 'CoMu',
        'dim': 200,
        'r_dim': 200,
        'dataset': 'DB15K',
        'pre_trained': 0,
        'image_features': 1,
        'text_features': 1,
        'eval_freq': 100,
        'gamma': 1.0,
        'neg_num': 2,
        'batch_size': 1024,
        'save': 1,
        'lamda_l':1e-5,
        'lamda_g':1e-4,
        'warmup_epochs':50,
        'emb_dir': '',
        'emb_dataset': '',
    }
    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val, type=type(val))
    args = parser.parse_args()
    return args

args = parse_args()
print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
torch.cuda.set_device(args.cuda)
for k, v in list(vars(args).items()):
    print(str(k) + ':' + str(v))

entity2id, relation2id, img_features, text_features, train_data, val_data, test_data = load_data(
    args.dataset
)
print("Training data {:04d}".format(len(train_data[0])))
print("img_features_dim: ", img_features.shape)
print("text_features_dim: ", text_features.shape)


corpus = ConvECorpus(args, train_data, val_data, test_data, entity2id, relation2id)

if args.image_features:
    args.img = F.normalize(torch.Tensor(img_features), p=2, dim=1)
if args.text_features:
    args.desp = F.normalize(torch.Tensor(text_features), p=2, dim=1)
args.entity2id = entity2id
args.relation2id = relation2id

model_name = {'CoMu': CoMu}
time.sleep(5)

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)  
            if m.bias is not None:
                nn.init.zeros_(m.bias) 
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def train_decoder(args):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model = model_name[args.model](args)
    
    init_weights(model)
    
    args.img_dim = model.img_dim
    args.txt_dim = model.txt_dim
    
    lamda_l = args.lamda_l
    lamda_g = args.lamda_g
    
    
    
    print(str(model))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)
    corpus.batch_size = args.batch_size
    corpus.neg_num = args.neg_num


    t_total = time.time()
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = model.init_metric_dict()
    training_range = tqdm(range(args.epochs))
    
    warmup_epochs = args.warmup_epochs
    
    
    for epoch in training_range:
        model.alpha = min(1.0, epoch / float(warmup_epochs))
        model.train()
        epoch_loss = []
        epoch_loss_s = []
        epoch_loss_i = []
        epoch_loss_t = []
        epoch_loss_mm = []
        epoch_ce_loss = []
        epoch_cl_loss = []
        epoch_cf_loss = []


      
        t = time.time()
        corpus.shuffle()
        for batch_num in range(corpus.max_batch_num):
            optimizer.zero_grad()
            train_indices, train_values = corpus.get_batch(batch_num)
            _,train_values_new = corpus.get_batch(batch_num)
            train_indices = torch.LongTensor(train_indices)
            if args.cuda is not None and int(args.cuda) >= 0:
                train_indices = train_indices.to(args.device)
                train_values = train_values.to(args.device)
                train_values_new = train_values_new.to(args.device)
            output, attn_cf = model.forward(train_indices)
            loss_s, loss_i, loss_t, loss_mm, cl_loss, cf_loss = model.loss_func(output, train_values)
            loss =  loss_s + loss_i + loss_t + loss_mm + lamda_l * cl_loss + lamda_g * cf_loss
            loss_ce = loss_s + loss_i + loss_t + loss_mm

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            epoch_loss.append(loss.data.item())
            
            epoch_loss_s.append(loss_s.data.item())
            epoch_loss_i.append(loss_i.data.item())
            epoch_loss_t.append(loss_t.data.item())
            epoch_loss_mm.append(loss_mm.data.item())
            
            epoch_cl_loss.append(cl_loss.data.item())
            epoch_cf_loss.append(cf_loss.data.item())
            
            epoch_ce_loss.append(loss_ce.data.item())

 

        training_range.set_postfix(loss_ce=f"{np.sum(epoch_ce_loss):.5f}", cl_loss=f"{np.sum(epoch_cl_loss):.5f}", cf_loss=f"{np.sum(epoch_cf_loss):.5f}", loss_s=f"{np.sum(epoch_loss_s):.5f}", loss_i=f"{np.sum(epoch_loss_i):.5f}", loss_t=f"{np.sum(epoch_loss_t):.5f}", loss_mm=f"{np.sum(epoch_loss_mm):.5f}")
        
        lr_scheduler.step()

        if (epoch + 1) % args.eval_freq == 0:
            print(f"Epoch {epoch + 1}: Evaluating on Test Set ({args.dataset})...")
            model.eval()
            with torch.no_grad():
                val_metrics, _  = corpus.get_validation_pred(model, 'test')  
                val_metrics_s, _ = corpus.get_validation_pred_signle(model, 'test', 0)
                val_metrics_i, _ = corpus.get_validation_pred_signle(model, 'test', 1)
                val_metrics_t, _ = corpus.get_validation_pred_signle(model, 'test', 2)
                val_metrics_mm, _ = corpus.get_validation_pred_signle(model, 'test', 3)   
            if val_metrics['MRR'] > best_test_metrics['MRR']:
                best_test_metrics['MRR'] = val_metrics['MRR']
            if val_metrics['MR'] < best_test_metrics['MR']:
                best_test_metrics['MR'] = val_metrics['MR']
            if val_metrics['Hits@1'] > best_test_metrics['Hits@1']:
                best_test_metrics['Hits@1'] = val_metrics['Hits@1']
            if val_metrics['Hits@3'] > best_test_metrics['Hits@3']:
                best_test_metrics['Hits@3'] = val_metrics['Hits@3']
            if val_metrics['Hits@10'] > best_test_metrics['Hits@10']:
                best_test_metrics['Hits@10'] = val_metrics['Hits@10']
            if val_metrics['Hits@100'] > best_test_metrics['Hits@100']:
                best_test_metrics['Hits@100'] = val_metrics['Hits@100']      
            print('\n'.join(['Epoch: {:04d}, Overall'.format(epoch + 1), model.format_metrics(val_metrics, 'test')]))
            print('\n\n'.join(['Epoch: {:04d}, Structure: '.format(epoch + 1), model.format_metrics(val_metrics_s, 'test')]))
            print('\n'.join(['Epoch: {:04d}, Image: '.format(epoch + 1), model.format_metrics(val_metrics_i, 'test')]))
            print('\n'.join(['Epoch: {:04d}, Text: '.format(epoch + 1), model.format_metrics(val_metrics_t, 'test')]))
            print('\n'.join(['Epoch: {:04d}, Multi-modal: '.format(epoch + 1), model.format_metrics(val_metrics_mm, 'test')]))
            print("\n\n")
            print("\n\n")

            
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        with torch.no_grad():
            best_test_metrics, _ = corpus.get_validation_pred(model, 'test')
    print('\n'.join(['Test set results:', model.format_metrics(best_test_metrics, 'test')]))
    print("\n\n\n\n\n\n")

    if args.save:
        save_dir = f'./checkpoint/{args.dataset}/{current_time}'
        os.makedirs(save_dir, exist_ok=True) 
        torch.save(model.state_dict(), os.path.join(save_dir, f'{args.model}.pth'))
        print('Saved model!')

if __name__ == '__main__':
    train_decoder(args)
