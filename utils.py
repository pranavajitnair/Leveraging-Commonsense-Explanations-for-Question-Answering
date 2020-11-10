import json
import random

def prepare_input():
        dev_com=list(open('dev_rand_split.jsonl'))
        result_dev=[json.loads(jline) for jline in dev_com]
        train_com=list(open('train_rand_split.jsonl'))
        result_train=[json.loads(jline) for jline in train_com]
        
        dev_cose=list(open('cose_dev.jsonl'))
        exp_dev=[json.loads(jline) for jline in dev_cose]
        train_cose=list(open('cose_train.jsonl'))
        exp_train=[json.loads(jline) for jline in train_cose]
        
        dict_train,dict_dev,dict_train_cose,dict_dev_cose={},{},{},{}
        for line in result_dev:
                dict_dev[line['id']]=line['question']['stem']+' </s> '+ \
                line['question']['choices'][ord(line['answerKey'])-ord('A')]['text']
        for line in result_train:
                dict_train[line['id']]=line['question']['stem']+' </s> '+ \
                line['question']['choices'][ord(line['answerKey'])-ord('A')]['text']
                
        for line in exp_dev:
                dict_dev_cose[line['id']]=line['explanation']['open-ended']
        for line in exp_train:
                dict_train_cose[line['id']]=line['explanation']['open-ended']
                
        train_data,dev_data=[],[]
        
        for k in dict_train:
                train_data.append([dict_train[k],dict_train_cose[k]])
        for k in dict_dev:
                dev_data.append([dict_dev[k],dict_dev_cose[k]])
                
        return train_data,dev_data
    
def prepare_anlg():
        dev=open('dev-w-comet-preds.jsonl')
        result_dev=[json.loads(jline) for jline in dev]
        train=open('train-w-comet-preds.jsonl')
        result_train=[json.loads(jline) for jline in train]
        
        out_dev,out_train=[],[]
        for lines in result_dev:
                store='hyp'+lines['label']
                out_dev.append([lines['obs1']+' </s> '+lines['obs2'],lines[store]])
        for lines in result_train:
                store='hyp'+lines['label']
                out_train.append([lines['obs1']+' </s> '+lines['obs2'],lines[store]])
                
        return out_train,out_dev
    
def prepare_recolr():
        dev=open('val.json')
        result_dev=[json.loads(jline) for jline in dev]
        train=open('train.json')
        result_train=[json.loads(jline) for jline in train]
        
        dev_data,train_data=[],[]
        for lines in result_dev[0]:
                dev_data.append([lines['context'],lines['question'],lines['answers'][0],
                            lines['answers'][1],lines['answers'][2],
                            lines['answers'][3],lines['label']])
        for lines in result_train[0]:
                train_data.append([lines['context'],lines['question'],lines['answers'][0],
                            lines['answers'][1],lines['answers'][2],
                            lines['answers'][3],lines['label']])
        
        return train_data,dev_data
            
class Dataloader(object):
        def __init__(self,data,bs):
                self.data=data
                self.bs=bs
                
        def get_data(self):
                data=random.sample(self.data,self.bs)
                inp,label=[],[]
                for dat in data:
                        inp.append(dat[0])
                        label.append(dat[1])
                        
                return inp,label
            

class DataLoader(object):
        def __init__(self,data,bs):
                self.data=data
                self.bs=bs
                
        def get_data(self):
                data=random.sample(self.data,self.bs)
                inp,label=[],[]
                for line in data:
                        inp.append(line[0]+' '+line[1]+' [SEP] '+line[2])
                        inp.append(line[0]+' '+line[1]+' [SEP] '+line[3])
                        inp.append(line[0]+' '+line[1]+' [SEP] '+line[4])
                        inp.append(line[0]+' '+line[1]+' [SEP] '+line[5])
                        
                        label.append(1*(line[6]==0))
                        label.append(1*(line[6]==1))
                        label.append(1*(line[6]==2))
                        label.append(1*(line[6]==3))
                        
                return inp,label