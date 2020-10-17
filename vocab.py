# -*- coding: utf-8 -*-
import sentencepiece as spm
import os

def vocab_model(input_path,vocab_size,model_type='bpe'):
    templates= '--input={} \
    --pad_id={} \
    --bos_id={} \
    --eos_id={} \
    --unk_id={} \
    --model_prefix={} \
    --vocab_size={} \
    --character_coverage={} \
    --model_type={}'
    if not os.isdir('vocab'):
        os.mkdir('vocab')
    output_path = f"./vocab/{input_path.split('/')[-1].split('.')[0]}"
    pad_id=0  #<pad> token을 0으로 설정
    bos_id=1 #<start> token을 1으로 설정
    eos_id=2 #<end> token을 2으로 설정
    unk_id=3 #<unknown> token을 3으로 설정
    character_coverage = 1.0 # to reduce character set 
    
    cmd = templates.format(input_path,
                    pad_id,
                    bos_id,
                    eos_id,
                    unk_id,
                    output_path,
                    vocab_size,
                    character_coverage,
                    model_type)
    spm.SentencePieceTrainer.Train(cmd)
    return output_path

def tokenizer(output_path,mod = 'train') :
    sp = spm.SentencePieceProcessor()
    sp.load(f'{ouput_path}.model')
    if mod == 'train :
        sp.SetEncodeExtraOptions('bos:eos')
    elif mod in ['eval','predict'] :
        sp.SetEncodeExtraOptions('bos')
    else :
        print(f'The "{mod}" could not be found. Progress in prediction mode')
        sp.SetEncodeExtraOptions('bos')
    return sp


def sentence_split(pred, vocab, eos_id=2) :
    start = 0
    end = 0
    for i in pred :
        if i == eos_id :
            print(vocab(pred[start:end]))
            start = end
        end += 1
    print(vocab(pred[start:end]))
