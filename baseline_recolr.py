from transformers import AlbertTokenizer, AlbertForSequenceClassification

import torch
import torch.optim as optim

import random
import json


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


lr=0.00003
bs=32
iterations=2000

tokenizer=AlbertTokenizer.from_pretrained('albert-xlarge-v2')
model=AlbertForSequenceClassification.from_pretrained('albert-xlarge-v2', return_dict=True) #.cuda()
optimizer=optim.Adam(model.parameters(),lr=lr)

train_data,dev_data=prepare_recolr()
data_loader=DataLoader(train_data,bs)

def val_recolr(dev_data):
        count=0

        for line in dev_data:
                inp=[]
                inp.append(line[0]+' '+line[1]+' [SEP] '+line[2])
                inp.append(line[0]+' '+line[1]+' [SEP] '+line[3])
                inp.append(line[0]+' '+line[1]+' [SEP] '+line[4])
                inp.append(line[0]+' '+line[1]+' [SEP] '+line[5])
        
                inputs=tokenizer(inp,return_tensors="pt")
                # for k in inputs:
                #         inputs[k]=inputs[k].cuda()
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
        # for k in inputs:
        #         inputs[k]=inputs[k].cuda()
        label=torch.tensor(label) #.cuda()

        optimizer.zero_grad()
        output=model(**inputs,labels=label)
        loss=output.loss
        scalar+=loss.item()
        if (i+1)%10==0:
                print('iterations={}, loss={}'.format(i+1,scalar/100))
                scalar=0
        if(i+1)%10==0:
                 acc=val_recolr(dev_data)
                 print('validation accuracy={}'.format(acc))
                
        loss.backward()
        optimizer.step()