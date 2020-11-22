import warnings
warnings.filterwarnings("ignore")
import argparse
import json
# import matplotlib
# import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cuda
import sys, os
import random
import numpy as np
from sklearn import metrics
import models as Model
# from SiameseLoss import ContrastiveLoss
import evaluate
import data
import gc
import csv
import pandas as pd
from pdb import set_trace as stop

from tqdm import tqdm_notebook
import kipoi
from copy import deepcopy
from scipy.stats import pearsonr

# python train.py --cell_type=Cell1 --model_name=attchrome --epochs=120 --lr=0.0001 --data_root=data/ --save_root=Results/

parser = argparse.ArgumentParser(description='DeepDiff')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--model_type', type=str, default='attchrome', help='DeepDiff variation')
parser.add_argument('--clip', type=float, default=1,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout) if n_layers LSTM > 1')
parser.add_argument('--cell_type', type=str, default='E003', help='cell type 1')
parser.add_argument('--save_root', type=str, default='./Results/', help='where to save')
parser.add_argument('--model_root', type=str, default=None, help='where to save')
parser.add_argument('--data_root', type=str, default='./data/', help='data location')
parser.add_argument('--gpuid', type=int, default=0, help='CUDA gpu')
parser.add_argument('--gpu', type=int, default=0, help='CUDA gpu')
parser.add_argument('--n_hms', type=int, default=5, help='number of histone modifications')
parser.add_argument('--n_bins', type=int, default=100, help='number of bins')
parser.add_argument('--bin_rnn_size', type=int, default=32, help='bin rnn size')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--unidirectional', action='store_true', help='bidirectional/undirectional LSTM')
parser.add_argument('--save_attention_maps',action='store_true', help='set to save validation beta attention maps')
parser.add_argument('--attentionfilename', type=str, default='beta_attention.txt', help='where to save attnetion maps')
parser.add_argument('--test_on_saved_model',action='store_true', help='only test on saved model')
parser.add_argument('--kipoi_model',action='store_true', help='only test on saved model')
parser.add_argument('--pgd',action='store_true', help='pgd')
parser.add_argument('--pgd_steps',type=int, default=2, help='pgd')
parser.add_argument('--pgd_mask',type=str, default=None, help='pgd mask, fg or bg')
parser.add_argument('--pgd_mask_threshold',type=int, default=1, help='pgd mask threshold')
parser.add_argument('--save_adv_inputs', action='store_true', help='save generated adversarial inputs')
parser.add_argument('--only_run_model', type=str, default=None, help='Run only a specific model instead of all 55')
args = parser.parse_args()



def main(args):
    torch.manual_seed(1)


    model_name = ''
    model_name += (args.cell_type)+('_')

    model_name+=args.model_type



    args.bidirectional=not args.unidirectional

    print('the model name: ',model_name)
    args.data_root+=''
    args.save_root+=''
    args.dataset=args.cell_type
    args.data_root = os.path.join(args.data_root)
    print('loading data from:  ',args.data_root)
    args.save_root = os.path.join(args.save_root,args.dataset)
    print('saving results in from: ',args.save_root)
    if args.model_root is None:
      model_dir = os.path.join(args.save_root,model_name)
    else:
      args.model_root = os.path.join(args.model_root,args.dataset)
      model_dir = os.path.join(args.model_root,model_name)
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)



    attentionmapfile=model_dir+'/'+args.attentionfilename
    orig_attentionmapfile=model_dir+'/'+'orig_'+args.attentionfilename
    print('==>processing data')
    Train,Valid,Test = data.load_data(args)






    print('==>building model')
    model = Model.att_chrome(args)



    if torch.cuda.device_count() > 0:
      torch.cuda.manual_seed_all(1)
      dtype = torch.cuda.FloatTensor
      # cuda.set_device(args.gpuid)
      model.type(dtype)
      print('Using GPU '+str(args.gpuid))
    else:
      dtype = torch.FloatTensor

    #print(model)
    if(args.test_on_saved_model==False):
      print("==>initializing a new model")
      for p in model.parameters():
        p.data.uniform_(-0.1,0.1)

    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    #optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)


    def pgd_attack(model, inputs_1, batch_diff_targets, eps=1.0, alpha=1.0, iters=2):

        ori_inputs_1 = inputs_1.data

        if args.pgd_mask:
            if args.pgd_mask == 'fg':
                pgd_mask = (ori_inputs_1 > args.pgd_mask_threshold).to(torch.float32)
            elif args.pgd_mask == 'bg':
                pgd_mask = (ori_inputs_1 <= args.pgd_mask_threshold).to(torch.float32)

        #print(ori_images.min(), ori_images.max())
        for i in range(iters) :
            inputs_1.requires_grad = True

            if i == 0:
                batch_predictions_orig, batch_beta_orig, batch_alpha_orig = model(inputs_1.type(dtype))

            batch_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype))

            if i < iters - 1 :

              model.zero_grad()

              loss = F.binary_cross_entropy_with_logits(batch_predictions, batch_diff_targets.cuda(),reduction='mean')

              loss.backward()
              grad_values = inputs_1.grad
              adv_inputs_1 = inputs_1 + alpha*grad_values.sign()
              eta = torch.clamp(adv_inputs_1 - ori_inputs_1, min=-eps, max=eps)
              if args.pgd_mask:
                  eta = eta * pgd_mask

              inputs_1 = torch.clamp(ori_inputs_1 + eta, min=ori_inputs_1.min(), max=ori_inputs_1.max()).detach_()

        ori_inputs_1_flat = ori_inputs_1.detach().cpu().numpy().flatten()
        inputs_1_flat = inputs_1.detach().cpu().numpy().flatten()
        grad_values_flat = grad_values.detach().cpu().numpy().flatten()

        corr = pearsonr(inputs_1.detach().cpu().numpy().flatten(), ori_inputs_1.cpu().numpy().flatten())[0]

        return batch_predictions, corr, inputs_1_flat, grad_values_flat, ori_inputs_1_flat, batch_beta, batch_alpha, batch_predictions_orig, batch_beta_orig, batch_alpha_orig


    def train(TrainData):
      model.train()
      # initialize attention
      diff_targets = torch.zeros(TrainData.dataset.__len__(),1)
      predictions = torch.zeros(diff_targets.size(0),1)

      all_attention_bin=torch.zeros(TrainData.dataset.__len__(),(args.n_hms*args.n_bins))
      all_attention_hm=torch.zeros(TrainData.dataset.__len__(),args.n_hms)

      num_batches = int(math.ceil(TrainData.dataset.__len__()/float(args.batch_size)))
      all_gene_ids=[None]*TrainData.dataset.__len__()
      per_epoch_loss = 0
      print('Training')
      for idx, Sample in enumerate(TrainData):

        start,end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, TrainData.dataset.__len__())


        inputs_1 = Sample['input']
        batch_diff_targets = Sample['label'].unsqueeze(1).float()


        optimizer.zero_grad()

        if args.pgd:
          batch_predictions, corr, adv_inputs, grad_values, ori_inputs, batch_beta, batch_alpha, batch_predictions_orig, batch_beta_orig, batch_alpha_orig = pgd_attack(model, inputs_1.type(dtype), batch_diff_targets, eps=1.0, alpha=1.0, iters=args.pgd_steps)

        else:
          batch_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype))

        loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets,reduction='mean')

        per_epoch_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        diff_targets[start:end,0] = batch_diff_targets[:,0]
        all_gene_ids[start:end]=Sample['geneID']
        batch_predictions = torch.sigmoid(batch_predictions)
        predictions[start:end] = batch_predictions.data.cpu()

        all_attention_bin[start:end]=batch_alpha.data
        all_attention_hm[start:end]=batch_beta.data

      per_epoch_loss=per_epoch_loss/num_batches
      return predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss



    def test(ValidData):
      if args.pgd:
        model.train()
      else:
        model.eval()

      diff_targets = torch.zeros(ValidData.dataset.__len__(),1)
      predictions = torch.zeros(diff_targets.size(0),1)
      predictions_orig = torch.zeros(diff_targets.size(0),1)

      all_attention_bin=torch.zeros(ValidData.dataset.__len__(),(args.n_hms*args.n_bins))
      all_attention_hm=torch.zeros(ValidData.dataset.__len__(),args.n_hms)

      all_attention_bin_orig=torch.zeros(ValidData.dataset.__len__(),(args.n_hms*args.n_bins))
      all_attention_hm_orig=torch.zeros(ValidData.dataset.__len__(),args.n_hms)

      num_batches = int(math.ceil(ValidData.dataset.__len__()/float(args.batch_size)))
      all_gene_ids=[None]*ValidData.dataset.__len__()
      per_epoch_loss = 0
      per_epoch_corr = 0

      all_adv_inputs = []
      all_grad_values = []
      all_ori_inputs = []

      for idx, Sample in enumerate(ValidData):

        start,end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, ValidData.dataset.__len__())
        optimizer.zero_grad()

        inputs_1 = Sample['input']
        batch_diff_targets= Sample['label'].unsqueeze(1).float()

        if args.pgd:
          batch_predictions, corr, adv_inputs, grad_values, ori_inputs, batch_beta, batch_alpha, batch_predictions_orig, batch_beta_orig, batch_alpha_orig = pgd_attack(model, inputs_1.type(dtype), batch_diff_targets, eps=1.0, alpha=1.0, iters=args.pgd_steps)
          all_adv_inputs = all_adv_inputs + list(adv_inputs)
          all_grad_values = all_grad_values + list(grad_values)
          all_ori_inputs = all_ori_inputs + list(ori_inputs)

          per_epoch_corr += corr

        else:
          batch_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype))
          batch_predictions_orig = batch_predictions

          #batch_predictions = model(inputs_1.type(dtype))

        loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets,reduction='mean')

        all_attention_bin[start:end]=batch_alpha.data
        all_attention_hm[start:end]=batch_beta.data
        all_attention_bin_orig[start:end]=batch_alpha_orig.data
        all_attention_hm_orig[start:end]=batch_beta_orig.data

        diff_targets[start:end,0] = batch_diff_targets[:,0]
        all_gene_ids[start:end]=Sample['geneID']
        batch_predictions = torch.sigmoid(batch_predictions)
        batch_predictions_orig = torch.sigmoid(batch_predictions_orig)
        predictions[start:end] = batch_predictions.data.cpu()
        predictions_orig[start:end] = batch_predictions_orig.data.cpu()
        per_epoch_loss += loss.item()

      per_epoch_loss=per_epoch_loss/num_batches
      per_epoch_corr=per_epoch_corr/num_batches

      return predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss,all_gene_ids, per_epoch_corr, np.array(all_adv_inputs), np.array(all_grad_values), np.array(all_ori_inputs), all_attention_bin_orig, all_attention_hm_orig, predictions_orig




    best_valid_loss = 10000000000
    best_valid_avgAUPR=-1
    best_valid_avgAUC=-1
    best_test_avgAUC=-1
    if(args.test_on_saved_model==False):
      for epoch in range(0, args.epochs):
        print('---------------------------------------- Training '+str(epoch+1)+' -----------------------------------')
        predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss = train(Train)
        train_avgAUPR, train_avgAUC = evaluate.compute_metrics(predictions,diff_targets)
        predictions,diff_targets,alpha_valid,beta_valid,valid_loss,gene_ids_valid,test_corr,adv_inputs_valid, grad_values_valid, ori_inputs_valid, _, _, _ = test(Valid)
        valid_avgAUPR, valid_avgAUC = evaluate.compute_metrics(predictions,diff_targets)

        predictions,diff_targets,alpha_test,beta_test,test_loss,gene_ids_test,test_corr, adv_inputs_test, grad_values_test, ori_inputs_test, _, _, _  = test(Test)
        test_avgAUPR, test_avgAUC = evaluate.compute_metrics(predictions,diff_targets)

        if(valid_avgAUC >= best_valid_avgAUC):
            # save best epoch -- models converge early
          best_valid_avgAUC = valid_avgAUC
          best_test_avgAUC = test_avgAUC
          torch.save(model.cpu().state_dict(),model_dir+"/"+model_name+'_avgAUC_model.pt')
          model.type(dtype)

        print("Epoch:",epoch)
        print("train avgAUC:",train_avgAUC)
        print("valid avgAUC:",valid_avgAUC)
        print("test avgAUC:",test_avgAUC)
        print("best valid avgAUC:", best_valid_avgAUC)
        print("best test avgAUC:", best_test_avgAUC)


      print("\nFinished training")
      print("Best validation avgAUC:",best_valid_avgAUC)
      print("Best test avgAUC:",best_test_avgAUC)



      if(args.save_attention_maps):
        attentionfile=open(attentionmapfile,'w')
        attentionfilewriter=csv.writer(attentionfile)
        beta_test=beta_test.numpy()
        for i in range(len(gene_ids_test)):
          gene_attention=[]
          gene_attention.append(gene_ids_test[i])
          for e in beta_test[i,:]:
            gene_attention.append(str(e))
          attentionfilewriter.writerow(gene_attention)
        attentionfile.close()

      return best_test_avgAUC, test_corr


    else:
      if args.kipoi_model:
          model.load_state_dict(kipoi.get_model("AttentiveChrome/{}".format(args.cell_type)).model.state_dict())
      else:
          model.load_state_dict(torch.load(model_dir+"/"+model_name+'_avgAUC_model.pt'))

      predictions,diff_targets,alpha_test,beta_test,test_loss,gene_ids_test,test_corr, adv_inputs_test, grad_values_test, ori_inputs_test, alpha_test_orig, beta_test_orig, predictions_orig = test(Test)
      test_avgAUPR, test_avgAUC = evaluate.compute_metrics(predictions,diff_targets)
      print("test avgAUC:",test_avgAUC)
      print("test corr:",test_corr)

      if(args.save_attention_maps):
        attentionfile=open(attentionmapfile,'w')
        attentionfilewriter=csv.writer(attentionfile)
        beta_test=beta_test.numpy()
        for i in range(len(gene_ids_test)):
          gene_attention=[]
          gene_attention.append(gene_ids_test[i])
          for e in beta_test[i,:]:
            gene_attention.append(str(e))
          attentionfilewriter.writerow(gene_attention)
        attentionfile.close()
        if (args.save_adv_inputs):
            attentionfile=open(orig_attentionmapfile,'w')
            attentionfilewriter=csv.writer(attentionfile)
            beta_test_orig=beta_test_orig.numpy()
            for i in range(len(gene_ids_test)):
              gene_attention=[]
              gene_attention.append(gene_ids_test[i])
              for e in beta_test_orig[i,:]:
                gene_attention.append(str(e))
              attentionfilewriter.writerow(gene_attention)
            attentionfile.close()


      if (args.save_adv_inputs):
          return test_avgAUC, test_corr, gene_ids_test, adv_inputs_test, grad_values_test, ori_inputs_test, alpha_test, alpha_test_orig, predictions, predictions_orig, diff_targets, beta_test, beta_test_orig


      return test_avgAUC, test_corr




####################
###### SCRIPT ######
####################

auc_list = []
corr_list = []
og_args = deepcopy(args)

if args.only_run_model != None:
    cell_type_name = args.only_run_model
    args.cell_type = cell_type_name

    if (args.save_adv_inputs):
        auc, corr, gene_ids, adv_inputs, grad_values, ori_inputs, alphas, alphas_orig, yhats, yhats_orig, ys, betas, betas_orig = main(args)
    else:
        auc, corr = main(args)

    if args.pgd:
      corr_list.append(corr)
      np.savetxt(args.save_root+'total_corr.txt', corr_list)
      np.save(args.save_root+'total_corr.npy', corr_list)
      if (args.save_adv_inputs):
          all_gene_ids = np.reshape([int(gene_id) for gene_id in gene_ids for i in range(args.n_bins)],(-1,1))
          ori_inputs = np.reshape(ori_inputs, (-1, args.n_hms))
          adv_inputs = np.reshape(adv_inputs, (-1, args.n_hms))
          etas = adv_inputs - ori_inputs
          grad_values = np.reshape(grad_values, (-1, args.n_hms))
          alphas_orig = np.reshape(alphas_orig.numpy(), (-1, args.n_hms))
          alphas = np.reshape(alphas.numpy(), (-1, args.n_hms))
          #betas_orig
          #betas
          ys = np.reshape([y for y in ys for i in range(args.n_bins)],(-1,1))
          yhats_orig = np.reshape([yhat for yhat in yhats_orig for i in range(args.n_bins)],(-1,1))
          yhats = np.reshape([yhat for yhat in yhats for i in range(args.n_bins)],(-1,1))
          vis_output = np.concatenate((all_gene_ids, ori_inputs, adv_inputs, etas, grad_values, alphas_orig, alphas, ys, yhats_orig, yhats), axis=1)
          np.save(args.save_root+'vis_output.npy', vis_output)
          print("Saved output for visualization [geneID, original inputs, adversarial inputs, etas, grads, original attention, attacked attention, label, prediction] as a numpy file!")

    auc_list.append(auc)
    np.savetxt(args.save_root+'total_auc.txt', auc_list)
    np.save(args.save_root+'total_auc.npy', auc_list)

    args = deepcopy(og_args)

else:

    # UNCOMMENT STUFF TO SELECT ONLY A SUBSET OF MODELS TO RUN (everything from EXXX onwards)
    #skip = True
    for cell_type_name in np.sort(list(os.listdir(args.data_root))):
        # there is no E059...
        if cell_type_name == 'E059':
            continue
        #if cell_type_name == 'E109':
        #  skip = False
        #if skip:
        #  continue
        args.cell_type = cell_type_name

        if (args.save_adv_inputs):
            auc, corr, gene_ids, adv_inputs, grad_values, ori_inputs, alphas, alphas_orig, yhats, yhats_orig, ys, betas, betas_orig = main(args)
        else:
            auc, corr = main(args)

        if args.pgd:
          corr_list.append(corr)
          np.savetxt(args.save_root+'total_corr.txt', corr_list)
          np.save(args.save_root+'total_corr.npy', corr_list)
          if (args.save_adv_inputs):
              all_gene_ids = np.reshape([int(gene_id) for gene_id in gene_ids for i in range(args.n_bins)],(-1,1))
              ori_inputs = np.reshape(ori_inputs, (-1, args.n_hms))
              adv_inputs = np.reshape(adv_inputs, (-1, args.n_hms))
              etas = adv_inputs - ori_inputs
              grad_values = np.reshape(grad_values, (-1, args.n_hms))
              alphas_orig = np.reshape(alphas_orig.numpy(), (-1, args.n_hms))
              alphas = np.reshape(alphas.numpy(), (-1, args.n_hms))
              #betas_orig
              #betas
              ys = np.reshape([y for y in ys for i in range(args.n_bins)],(-1,1))
              yhats_orig = np.reshape([yhat for yhat in yhats_orig for i in range(args.n_bins)],(-1,1))
              yhats = np.reshape([yhat for yhat in yhats for i in range(args.n_bins)],(-1,1))
              vis_output = np.concatenate((all_gene_ids, ori_inputs, adv_inputs, etas, grad_values, alphas_orig, alphas, ys, yhats_orig, yhats), axis=1)
              np.save(args.save_root+'vis_output.npy', vis_output)
              print("Saved output for visualization [geneID, original inputs, adversarial inputs, etas, grads, original attention, attacked attention, label, prediction] as a numpy file!")

        auc_list.append(auc)
        np.savetxt(args.save_root+'total_auc.txt', auc_list)
        np.save(args.save_root+'total_auc.npy', auc_list)

        args = deepcopy(og_args)

print(auc_list)
print('total auc: {}'.format(np.mean(auc_list)))
