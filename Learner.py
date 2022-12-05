import os

from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import torch.nn as nn
import bcolz


class face_learner(object):
    def __init__(self, conf, load=None,inference=False):
        device_id=[0,1]
        print(conf)
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))

        self.model=nn.DataParallel(self.model,device_ids=device_id).to(conf.device)

        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)        

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = nn.DataParallel(Arcface(embedding_size=conf.embedding_size, classnum=self.class_num),device_ids=device_id).to(conf.device)

            print('two model heads generated',conf.lr)

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model.module)

            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.module.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.module.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)

            # self.schedule_lr(set_to=1)
            #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5,factor=0.5,mode='min')

            print('optimizers generated')
            self.define_interval()
            # self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(self.loader.dataset.root.parent)
            self.lfw, self.lfw_issame = get_val_data(self.loader.dataset.root.parent)
        else:
            self.threshold = conf.threshold

        self.init_head=False

    def define_interval(self):

        self.board_loss_every = 100
        self.evaluate_every = len(self.loader) // 5
        self.save_every = len(self.loader) // 5
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        torch.save(
            self.model.state_dict(), os.path.join(save_path,
            ('model_{}_accuracy-{}_step-{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
        if not model_only:
            torch.save(
                self.head.state_dict(), os.path.join(save_path,
                ('head_{}_accuracy-{}_step-{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
            torch.save(
                self.optimizer.state_dict(), os.path.join(save_path ,
                ('optimizer_{}_accuracy-{}_step-{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))

        torch.save(
            self.model.state_dict(), os.path.join(conf.save_path,
                                                  ('model_ir_se50.pth'.format(get_time(), accuracy,
                                                                                                self.step, extra))))
        if not model_only:
            torch.save(
                self.head.state_dict(), os.path.join(conf.save_path,
                                                     ('head_ir_se50.pth'.format(get_time(), accuracy,
                                                                                                  self.step, extra))))
            torch.save(
                self.optimizer.state_dict(), os.path.join(conf.save_path,
                                                          ('optimizer_ir_se50.pth'.format(get_time(),
                                                                                                            accuracy,
                                                                                                            self.step,
                                                                                                            extra))))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False, from_multiple_GPU=False):
        self.init_head=model_only
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        state_dict=torch.load(os.path.join(save_path,'model_{}'.format(fixed_str)),map_location='cuda')
        print("Load Model: ",os.path.join(save_path,'model_{}'.format(fixed_str)))
        # if from_multiple_GPU:
        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k, v in state_dict.items():
        #         name = k[7:]
        #         new_state_dict[name] = v
        #     state_dict=new_state_dict
        if from_multiple_GPU:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.module.load_state_dict(state_dict)
        if not model_only:
            self.head.load_state_dict(torch.load(os.path.join(save_path,'head_{}'.format(fixed_str)),map_location='cuda'))
            self.optimizer.load_state_dict(torch.load(os.path.join(save_path,'optimizer_{}'.format(fixed_str)),map_location='cuda'))
        else:
            self.schedule_lr(set_to=1)
            for param in self.model.module.input_layer.parameters():
                param.requires_grad = False

            for param in self.model.module.body.parameters():
                param.requires_grad = False

            for param in self.model.module.output_layer.parameters():
                param.requires_grad = False
        
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
        
    def evaluate(self, conf, carray, issame, nrof_folds = 5, tta = False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    # def find_lr(self,
    #             conf,
    #             init_value=1e-8,
    #             final_value=10.,
    #             beta=0.98,
    #             bloding_scale=3.,
    #             num=None):
    #     if not num:
    #         num = len(self.loader)
    #     mult = (final_value / init_value)**(1 / num)
    #     lr = init_value
    #     for params in self.optimizer.param_groups:
    #         params['lr'] = lr
    #     self.model.train()
    #     avg_loss = 0.
    #     best_loss = 0.
    #     batch_num = 0
    #     losses = []
    #     log_lrs = []
    #     for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):
    #
    #         imgs = imgs.to(conf.device)
    #         labels = labels.to(conf.device)
    #         batch_num += 1
    #
    #         self.optimizer.zero_grad()
    #
    #         embeddings = self.model(imgs)
    #         thetas = self.head(embeddings, labels)
    #         loss = conf.ce_loss(thetas, labels)
    #
    #         #Compute the smoothed loss
    #         avg_loss = beta * avg_loss + (1 - beta) * loss.item()
    #         self.writer.add_scalar('avg_loss', avg_loss, batch_num)
    #         smoothed_loss = avg_loss / (1 - beta**batch_num)
    #         self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
    #         #Stop if the loss is exploding
    #         if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
    #             print('exited with best_loss at {}'.format(best_loss))
    #             plt.plot(log_lrs[10:-5], losses[10:-5])
    #             return log_lrs, losses
    #         #Record the best loss
    #         if smoothed_loss < best_loss or batch_num == 1:
    #             best_loss = smoothed_loss
    #         #Store the values
    #         losses.append(smoothed_loss)
    #         log_lrs.append(math.log10(lr))
    #         self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
    #         #Do the SGD step
    #         #Update the lr for the next step
    #
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         lr *= mult
    #         for params in self.optimizer.param_groups:
    #             params['lr'] = lr
    #         if batch_num > num:
    #             plt.plot(log_lrs[10:-5], losses[10:-5])
    #             return log_lrs, losses

    def train(self, conf, epochs):
        running_loss = 0.
        for e in range(epochs):

            self.model.train()
            print("Set Learning Rate:",self.optimizer.param_groups[0]['lr'])
            print("Set Learning Rate: {}".format(self.optimizer.param_groups[1]['lr']))
            print('epoch {} started'.format(e))
            # if e == self.milestones[0]:
            #
            #
            #
            #
            # self.schedule_lr()
            # if e == self.milestones[1]:
            #     self.schedule_lr()
            # if e == self.milestones[2]:
            #     self.schedule_lr()
            for imgs, labels in (pbar := tqdm(iter(self.loader))):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    pbar.set_description(f"Loss {loss_board}")
                    self.scheduler.step(loss_board)
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    # accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30, self.agedb_30_issame)
                    # self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    # accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    # self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)

                if self.init_head:
                    # if self.step%10==0:
                    #     self.schedule_lr()
                    if self.step==700:
                        for param in self.model.module.output_layer.parameters():
                            param.requires_grad = True
                    if self.step>1000:
                        print("Init Head Finished")
                        self.schedule_lr(set_to=conf.lr)
                        self.init_head=False
                        for param in self.model.module.input_layer.parameters():
                            param.requires_grad = True

                        for param in self.model.module.body.parameters():
                            param.requires_grad = True

                        for param in self.model.module.output_layer.parameters():
                            param.requires_grad = True
                # elif self.step%100==0:
                #     self.schedule_lr(gamma=0.8)
                self.step += 1


            conf.dataset_id+=1
            conf.dataset_id%=10
            print("Set Dataset_id: {}".format(conf.dataset_id))
            print("Accuracy: {}".format(accuracy))
            self.loader, self.class_num = get_train_loader(conf)

        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self,set_to=None,gamma=0.5):
        print(self.optimizer)
        for params in self.optimizer.param_groups:
            print(params['lr'])
            if set_to:
                lr=params['lr']=set_to
            else:
                lr=params['lr']*gamma
            print(lr)
            params['lr'] = lr
        print(self.optimizer)
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               