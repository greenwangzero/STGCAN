# from load2 import *
import joblib
import numpy as np
import time
import random
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
import logging
from model import *
import argparse
import torch.distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='nyc', type=str)
parser.add_argument("--device", default='cuda:7', type=str)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument("--batch_size", default=15, type=int)
parser.add_argument("--dim", default=40, type=int)
parser.add_argument("--neibour_size", default=4, type=int)
parser.add_argument("--max_len", default=100, type=int)

args = parser.parse_args()
max_len = args.max_len
neibour_size = args.neibour_size
device = args.device
dname = args.data

def calculate_acc(prob, label):
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        _, topk_predict_batch = torch.topk(prob, k=k)
        topk_predict_batch = topk_predict_batch + 1
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1

    return np.array(acc_train)


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]-1 
    label = label.view(-1)  
    init_label = np.linspace(0, num_label-1, num_label) 
    init_prob = torch.zeros(size=(num_label, num_neg+len(label))) 

    random_ig = random.sample(range(1, l_m+1), num_neg)  
    while len([lab for lab in label if lab in random_ig]) != 0: 
        random_ig = random.sample(range(1, l_m+1), num_neg)

    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i-len(label)]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label) 


class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length, neibour, neibour_relation, neibour_location, location_data, ex, location_candidate):
        self.traj, self.mat1, self.vec, self.label, self.length , self.neibour, self.neibour_relation = \
            traj, m1, v, label, length, neibour, neibour_relation
        self.neibour_location, self.location_data = neibour_location, location_data
        self.location_candidate = location_candidate
        self.ex = ex

    def __getitem__(self, index):
        try:
            traj = self.traj[index].to(device)
            mats1 = self.mat1[index].to(device)
            vector = self.vec[index].to(device) 
            label = self.label[index].to(device) 
            length = self.length[index].to(device) 
            neibour = self.neibour[index].to(device)
            neibour_relation = self.neibour_relation[index].to(device)
            neibour_location = self.neibour_location[index].to(device)
            location_data = self.location_data[index].to(device)
            ex = self.ex[0][index].to(device),self.ex[1][index].to(device)
            location_candidate = self.location_candidate[index]
            return traj, mats1, vector, label, length, neibour, neibour_relation, neibour_location, location_data, ex, location_candidate
        except Exception as e:
            print(index," error")
    
    def __len__(self):
        return len(self.traj)


class Trainer:
    def __init__(self, model, record):
        # load other parameters
        self.model = model.to(device)
        self.records = record
        self.start_epoch = 1
        self.num_neg = 10
        self.batch_size = 1
        self.learning_rate = 0.003
        self.num_epoch = 100
        self.threshold = 0 

        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len = \
            trajs, mat1, mat2s, mat2t, labels, lens

        self.neibour = neibour
        self.neibour_relation = neibour_relation

        self.neibour_location = neibour_location
        self.location_data = location_data
        self.ex = ex
        self.location_candidate = location_candidate
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label-1, self.len, self.neibour, \
            self.neibour_relation,self.neibour_location,self.location_data,self.ex,self.location_candidate)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        
    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        for t in range(self.num_epoch):
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]

            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
                person_input, person_m1, person_m2t, person_label, \
                person_traj_len, person_neibour, person_neibour_relation, \
                neibour_location,location_data,ex,location_candidate = item

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len,2), dtype=torch.float32).to(device)
                neibour_mask = torch.zeros((self.batch_size, max_len, neibour_size), dtype=torch.long).to(device)

                for mask_len in range(1, person_traj_len[0]):

                    input_mask[:, :mask_len] = 1
                    neibour_mask[:,:mask_len] = 1   
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1] 
                    neibour_mask = person_neibour * neibour_mask
                    neibour_relation_mask = person_neibour_relation* neibour_mask

                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len, \
                    neibour_mask, neibour_relation_mask, neibour_location, location_data, ex, location_candidate) 
                    
                    prob = prob[:, mask_len-1, :].squeeze(1)
           
                    if mask_len <= person_traj_len[0] * 0.8: 
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    elif mask_len > person_traj_len[0] * 0.8 and mask_len <= person_traj_len[0] * 0.9:
                        valid_size += person_input.shape[0]
                        acc_valid += calculate_acc(prob, train_label)

                    elif mask_len > person_traj_len[0] * 0.9:
                        test_size += person_input.shape[0]
                        acc_test += calculate_acc(prob, train_label)
                bar.update(self.batch_size)
            bar.close()

            acc_valid = np.array(acc_valid) / valid_size
            logging.warning('epoch:{}, time:{}, valid_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_valid))

            acc_test = np.array(acc_test) / test_size
            logging.warning('epoch:{}, time:{}, test_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_test))  


if __name__ == '__main__':
    # load data
    if dname =='NYC':
        file = open('./data/nyc/' + dname + '_data.pkl', 'rb')
        file_data = joblib.load(file)
        [trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = file_data

        ### neibour_4
        neibour = open("./data/nyc/neibour_size4.pkl",'rb')
        neibour_data = joblib.load(neibour)
        neibour, neibour_relation = neibour_data

        d = open("./data/nyc/neibour_location_4.pkl",'rb')
        neibour_location = joblib.load(d)
        d = open("./data/nyc/neibour_location_4_r.pkl",'rb')
        location_data = joblib.load(d)
        
    mat1, mat2s, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2s), \
                               torch.FloatTensor(mat2t).to(device), torch.LongTensor(lens).to(device)
    location_candidate = location_candidate.to(torch.int64)
    neibour_location = (neibour_location).to(torch.int64)


    neibour = torch.LongTensor(neibour)
    ex = mat1[:, :, :, 0].max(-1)[0].max(-1)[0].to(device), mat1[:, :, :, 1].max(-1)[0].max(-1)[0].to(device)
 
    u_max = neibour.max()
    l_max = mat2s.shape[0]
    model = Model(t_dim=hours+1, l_dim=l_max+1, u_dim=u_max+1, embed_dim=32, dropout=0.2)

    records = {'epoch': [], 'acc_valid': [], 'acc_test': []}
    start = time.time()

    trainer = Trainer(model, records)
    trainer.train()

