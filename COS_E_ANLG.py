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
                dict_dev[line['id']]=line['question']['stem'] # +' </s> '+ \
               # line['question']['choices'][ord(line['answerKey'])-ord('A')]['text']
        for line in result_train:
                dict_train[line['id']]=line['question']['stem'] # +' </s> '+ \
               # line['question']['choices'][ord(line['answerKey'])-ord('A')]['text']
                
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
            
import torch.optim as optim
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torchtext.data.metrics import bleu_score
import torch

lr=0.00003
#bs=16 for cose base max 7.974 new one 8.445          7.27 mx without answer only cose
#bs=32 for anlg                                       5.94 only
#iters=3000 for cose
#iters=8000 for anlg
iters=4000
bs=16

train_data,dev_data=prepare_input()
data_loader=Dataloader(train_data,bs)

tokenizer=T5Tokenizer.from_pretrained('t5-large')
model=T5ForConditionalGeneration.from_pretrained('t5-large', return_dict=True).cuda()
# model.load_state_dict(torch.load('model_t5_anlg_first.pth'))
optimizer=optim.Adam(model.parameters(),lr=lr)

def val(dev_data):
        candidate_corpus,references_corpus=[],[]
        for line in dev_data:
                inp,label=[line[0]],[line[1]]
                input=tokenizer.prepare_seq2seq_batch(src_texts=inp,
                                                      tgt_texts=label, padding=True, return_tensors='pt')
                
                output=model.generate(input_ids=input['input_ids'].cuda(),
                                      num_beams=5, early_stopping=True, max_length=20)
                out=tokenizer.batch_decode(output)
                candidate_corpus.append(tokenizer.tokenize(out[0]))
                references_corpus.append([tokenizer.tokenize(label[0])])

        return 100*bleu_score(candidate_corpus, references_corpus)


scalar=0
val_score=0
for i in range(iters):
        inp,label=data_loader.get_data()
        input=tokenizer.prepare_seq2seq_batch(src_texts=inp,
                                              tgt_texts=label, padding=True, return_tensors='pt')
        outputs=model(input_ids=input['input_ids'].cuda(),labels=input['labels'].cuda())
        loss=outputs[0]

        scalar+=loss.item()
        if(i+1)%100==0:
                print('iteration={}, training loss={}'.format(i+1,scalar/100))
                scalar=0
        if(i+1)%1000==0:
                 bleu=val(dev_data)
                 print('validation bleu={}'.format(bleu))
                 if bleu>=val_score:
                        torch.save(model.state_dict(),'model_t5_cose_with_answer.pth')
                        val_score=bleu

        loss.backward()
        optimizer.step()