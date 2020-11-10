import json

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

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

def data_for_exp(data):
        inp=[]
        for line in data:
                inp.append(line[1]+' [SEP] '+line[0])

        return inp

train_data,dev_data=prepare_recolr()
td=data_for_exp(train_data)
ed=data_for_exp(dev_data)

tokenizer=T5Tokenizer.from_pretrained('t5-base')
model=T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True) #.cuda()
model.load_state_dict(torch.load('model_t5_cose_with_answer.pth'))

def val1(dev_data):
        print('started')
        candidate_corpus=[]
        for line in dev_data:
                inp,label=[line],['0']
                input=tokenizer.prepare_seq2seq_batch(src_texts=inp,
                                                      tgt_texts=label, padding=True, return_tensors='pt')
                
                output=model.generate(input_ids=input['input_ids'],
                                      num_beams=5, early_stopping=True, max_length=20)  #.cuda()
                out=tokenizer.batch_decode(output)
                candidate_corpus.append(out)

        return candidate_corpus

exp_train=val1(td)
exp_dev=val1(ed)

dict={}
dict['train']=exp_train
dict['dev']=exp_dev
torch.save(dict,'exp.pth')