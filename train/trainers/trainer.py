import os
import time
import torch
from train.dataset import AvgMeter
from train.d_model.discriminator import Discriminator


class LTRTrainer:
    
    def __init__(self, net, objective, optimizer, train_loader, settings, t=0):
       
        self.net = net
        self.objective = objective
        self.optimizer = optimizer
        self.train_loader = train_loader 
        self.settings = settings
        self.t = t
        
    def train(self, diter_num):
        discriminator = Discriminator()
        if torch.cuda.is_available():
            discriminator.cuda()
    
        d_optim = torch.optim.Adagrad(discriminator.parameters(), 0.0003)
        curr_iter = self.settings.last_iter
        start_time = time.time()
        while True:
            total_loss_record, loss0_record = AvgMeter(), AvgMeter()

            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                batch_size = inputs.size(0)
                inputs = inputs.cuda()
                labels = labels.cuda()
           
                if self.t < diter_num:
                    d_optim.zero_grad()
                    inp_d = torch.cat((inputs, labels), 1)
                    outputs = discriminator(inp_d).squeeze()
                    d_loss = torch.sum(torch.log(outputs))
                    d_loss.backward()
                    d_optim.step()
                    print('[t %d], [d_loss %.5f]' %(self.t, d_loss))
                    self.t += 1
            
                else:
                
                    self.optimizer.param_groups[0]['lr'] = 2 * self.settings.lr * (1 - float(curr_iter) / self.settings.iter_num
                                                                ) ** self.settings.lr_decay
                    self.optimizer.param_groups[1]['lr'] = self.settings.lr * (1 - float(curr_iter) / self.settings.iter_num
                                                            ) ** self.settings.lr_decay

                    self.optimizer.zero_grad()
                    outputs0 = self.net(inputs)
                    inp_d = torch.cat((inputs,outputs0), 1)
                    outputs = discriminator(inp_d)
                    g_loss = self.objective(outputs0,labels)
                    g_dis_loss = -torch.log(outputs)
                    total_loss = torch.sum(0.2*g_dis_loss + g_loss) if diter_num > 0 else g_loss
                    #total_loss = g_loss # for first training 
            
                    total_loss.backward()
                    self.optimizer.step()

                    total_loss_record.update(total_loss.item(), batch_size)
                    loss0_record.update(g_loss.item(), batch_size)
           
                    curr_iter += 1

                    log = '[iter %d], [total loss %.5f], [lr %.10f]' % (curr_iter, total_loss_record.avg, self.optimizer.param_groups[1]['lr'])
                    print (log)
                    open(self.settings.log_path, 'a').write(log + '\n')

                    if curr_iter == self.settings.iter_num:
                        end_time = time.time()
                        T = '%d' % (end_time-start_time)
                        print(T)
                        self.net = self.net.module if torch.cuda.device_count() > 1 else self.net
                        torch.save(self.net.state_dict(), os.path.join(self.settings.ckpt, '%d.pth' % curr_iter))
                        torch.save(self.optimizer.state_dict(),
                           os.path.join(self.settings.ckpt, '%d_optim.pth' % curr_iter))
                        return
