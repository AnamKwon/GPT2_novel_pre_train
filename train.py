# -*- coding: utf-8 -*-
from transformers import GPT2Config, GPT2LMHeadModel
from vocab import vocab_model, tokenizer, sentence_split
from sampling import sample_sequence
import torch
import sentencepiece

DEVICE = torch.device('cuda') #GPU 디바이스 사용
config = GPT2Config(vocab_size=32000,eos_token_id=2, bos_token_id=1,pad_token_id=0) # vocab size와 태그(시작,끝,패딩)를 설정
model = GPT2LMHeadModel(config) #GPT2 모델 생성
model.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/checkpoint.tar',map_location=torch.device('cpu'))['model_state_dict']) #저장된 모델의 가중치를 가져오기
# 여기선 cpu를 사용한 이유는 GPU사용을 최대한 덜 하기위해
model.to(DEVICE) # 모델이 GPU로 동작할 수 있게 설정

with open('/content/drive/My Drive/Colab Notebooks/0001_jojung.txt',encoding='utf-8') as f :
  total_ids = []
  for i in f.readlines():
    to_ids = sp.EncodeAsIds(i.strip())
    total_ids.append(to_ids)
ctx_ids = []
sum_ids = []
for i in total_ids :
  lens = len(i)
  count = 0
  if lens > 1024 :
    while lens > 1024 :
      ctx_ids.append(i[count*1024:(count+1)*1024])
      lens -= 1024
      count += 1
  elif len(sum_ids + i) <= 1024 :
    sum_ids = sum_ids + i
  else :
    sum_ids = ([0]*(1024-len(sum_ids)))+sum_ids
    ctx_ids.append(sum_ids)
    sum_ids = i
'''토큰화 과정'''
train = torch.tensor(ctx_ids) # 학습을 위해 텐서로 변환
model.train() # transformers의 모델들은 학습할때나 예측을 할때 지정을 해줘야한다. train() or eval()
model.to(DEVICE)
learning_rate = 1e-5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # 경사하강법 적용 Adam
count = 0
epoch = 1
for epoch in range(500) :
  count = 0
  for i in range(0,len(train)-3,3) :
    optimizer.zero_grad()
    data = train[i:i+3] # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
    data = data.to(DEVICE)
    outputs = model(data, labels=data) # 학습 이전에 train모드로 하였기에 학습진행
    loss, logits = outputs[:2]
    loss.backward()
    optimizer.step() #GPU
    #xm.optimizer_step(optimizer, barrier=True) #TPU
    if count % 200 == 0 :
      print('epoch no.{} train no.{}  loss = {}' . format(epoch, count+1, loss))
    count += 1
  torch.save({
        'epoch': epoch,
        'train_no': count,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':loss
      }, '/content/drive/My Drive/Colab Notebooks/checkpoint.tar')
