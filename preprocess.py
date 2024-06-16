import re
import jieba
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)  # 移除非中文和非英文字符
    text = re.sub(r'[，。！？：；“”‘’（）]', '', text)  # 移除中文标点符号
    special_chars = r"[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~★…《》【】“”‘’]"
    text = re.sub(special_chars, '', text)  # 移除特殊字符
    text = re.sub(r'\s+', ' ', text).strip()  # 移除多余的空白字符
    return text

def segment_text(text):
    sentences = text.split('\n')
    segmented_text = []
    for sentence in sentences:
        if sentence.strip():
            segmented_sentence = " ".join(jieba.cut(sentence.strip()))
            segmented_text.append(f"<START> {segmented_sentence} <END>")
    return "\n".join(segmented_text)

def process_file(input_file):
    with open(input_file, 'r', encoding='gb18030') as infile:
        text = infile.read()
    cleaned_text = clean_text(text)
    segmented_text = segment_text(cleaned_text)
    return segmented_text

def process_corpus(corpus_files, output_file, max_len):
    combined_text = ""
    for file in corpus_files:
        combined_text += process_file(file) + "\n"

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(combined_text)

    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts([combined_text])
    sequences = tokenizer.texts_to_sequences([combined_text])
    
    tokenizer.word_index['<START>'] = len(tokenizer.word_index) + 1
    tokenizer.word_index['<END>'] = len(tokenizer.word_index) + 1
    
    sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    return tokenizer, sequences

if __name__ == "__main__":
    corpus_files = [
        "jyxstxtqj_downcc.com/inf.txt",
        "jyxstxtqj_downcc.com/三十三剑客图.txt",
        "jyxstxtqj_downcc.com/书剑恩仇录.txt",
        "jyxstxtqj_downcc.com/侠客行.txt",
        "jyxstxtqj_downcc.com/倚天屠龙记.txt",
        "jyxstxtqj_downcc.com/天龙八部.txt",
        "jyxstxtqj_downcc.com/射雕英雄传.txt",
        "jyxstxtqj_downcc.com/白马啸西风.txt",
        "jyxstxtqj_downcc.com/碧血剑.txt",
        "jyxstxtqj_downcc.com/神雕侠侣.txt",
        "jyxstxtqj_downcc.com/笑傲江湖.txt",
        "jyxstxtqj_downcc.com/越女剑.txt",
        "jyxstxtqj_downcc.com/连城诀.txt",
        "jyxstxtqj_downcc.com/雪山飞狐.txt",
        "jyxstxtqj_downcc.com/飞狐外传.txt",
        "jyxstxtqj_downcc.com/鸳鸯刀.txt",
        "jyxstxtqj_downcc.com/鹿鼎记.txt"
    ]
    
    output_file = "cleaned_corpus.txt"
    max_len = 100
    tokenizer, sequences = process_corpus(corpus_files, output_file, max_len)
    
    with open("tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())
    np.save("sequences.npy", sequences)
