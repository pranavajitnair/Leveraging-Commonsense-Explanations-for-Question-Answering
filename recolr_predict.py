from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch
import torch.optim as optim

import random
import json

lr=0.000003         #max 52.0% without anlg explanations 49% 51.8% only with cose(no answer while training cose) with answer 51.8%
bs=8                # cose with answers 51.6% but first question then passage while generating explanations 49.8% cose without answer question first setting 44.4% alni question first
iterations=4000

tokenizer=AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
model=AlbertForSequenceClassification.from_pretrained('albert-xxlarge-v2', return_dict=True) #.cuda()
optimizer=optim.Adam(model.parameters(),lr=lr)

class DataLoader(object):
        def __init__(self,data,bs,exp_train):  
                self.data=data
                self.bs=bs
                for i,line in enumerate(self.data):
                        line.append(exp_train[i][0])

        def get_data(self):
                data=random.sample(self.data,self.bs)
                inp,label=[],[]
                for line in data:
                        inp.append(line[0]+' '+line[7]+' '+line[1]+' [SEP] '+line[2])
                        inp.append(line[0]+' '+line[7]+' '+line[1]+' [SEP] '+line[3])
                        inp.append(line[0]+' '+line[7]+' '+line[1]+' [SEP] '+line[4])
                        inp.append(line[0]+' '+line[7]+' '+line[1]+' [SEP] '+line[5])
                        
                        label.append(1*(line[6]==0))
                        label.append(1*(line[6]==1))
                        label.append(1*(line[6]==2))
                        label.append(1*(line[6]==3))
                        
                return inp,label


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

train_data,dev_data=prepare_recolr()

dict=torch.load('exp.pth')
exp_train=dict['train']
exp_dev=dict['dev']

data_loader=DataLoader(train_data,bs,exp_train)

def val_recolr(dev_data,exp_dev):
        count=0

        for i,line in enumerate(dev_data):
                inp=[]
                inp.append(line[0]+' '+exp_dev[i][0]+' '+line[1]+' [SEP] '+line[2])  
                inp.append(line[0]+' '+exp_dev[i][0]+' '+line[1]+' [SEP] '+line[3])
                inp.append(line[0]+' '+exp_dev[i][0]+' '+line[1]+' [SEP] '+line[4])
                inp.append(line[0]+' '+exp_dev[i][0]+' '+line[1]+' [SEP] '+line[5])
        
                inputs=tokenizer(inp,padding=True,return_tensors="pt")

                for k in inputs:
                        inputs[k]=inputs[k] #.cuda()
                outputs=model(**inputs)
                logits=outputs.logits
                ans=torch.argmax(logits[:,-1].squeeze()).item()
                if ans==line[6]:
                        count+=1

        return 100*count/len(dev_data)

scalar=0
for i in range(iterations):
        inp,label=data_loader.get_data()
        inputs=tokenizer(inp,padding=True,return_tensors='pt')
        for k in inputs:
                inputs[k]=inputs[k] #.cuda()
        label=torch.tensor(label) #.cuda()

        optimizer.zero_grad()
        output=model(**inputs,labels=label)
        loss=output.loss
        scalar+=loss.item()
        if (i+1)%20==0:
                print('iterations={}, loss={}'.format(i+1,scalar/20))
                scalar=0
        if(i+1)%50==0:
                 acc=val_recolr(dev_data,exp_dev)
                 print('validation accuracy={}'.format(acc))
                
        loss.backward()
        optimizer.step()