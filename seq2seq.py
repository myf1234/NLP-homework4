import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping

# 启用急切执行模式以便调试
tf.config.run_functions_eagerly(True)

# 加载 tokenizer
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))

# 读取并加载文本数据
with open('cleaned_corpus.txt', 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

# 设置最大序列长度
max_sequence_length = 100

# 分割数据集
train_size = int(len(lines) * 0.8)
train_lines = lines[:train_size]
val_lines = lines[train_size:]

# 确保训练集和验证集不为空
if len(train_lines) == 0 or len(val_lines) == 0:
    raise ValueError("Training or validation lines are empty. Please check your data.")

# 转换为序列
train_sequences = tokenizer.texts_to_sequences(train_lines)
val_sequences = tokenizer.texts_to_sequences(val_lines)

# Padding
train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post')
val_sequences = pad_sequences(val_sequences, maxlen=max_sequence_length, padding='post')

# 准备模型输入输出数据
encoder_input_data = train_sequences
decoder_input_data = np.zeros_like(encoder_input_data)
decoder_input_data[:, 1:] = encoder_input_data[:, :-1]

# 确保 <START> token 存在于词汇表中
if '<START>' in tokenizer.word_index:
    start_token_index = tokenizer.word_index['<START>']
else:
    raise KeyError("<START> token not found in word index")

decoder_input_data[:, 0] = start_token_index
train_target_data = np.expand_dims(encoder_input_data, -1)

val_encoder_input_data = val_sequences
val_decoder_input_data = np.zeros_like(val_encoder_input_data)
val_decoder_input_data[:, 1:] = val_encoder_input_data[:, :-1]
val_decoder_input_data[:, 0] = start_token_index
val_target_data = np.expand_dims(val_encoder_input_data, -1)

# 模型定义
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
lstm_units = 256

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm1 = LSTM(lstm_units, return_sequences=True, return_state=True)
encoder_outputs1, state_h1, state_c1 = encoder_lstm1(encoder_embedding)
encoder_lstm2 = LSTM(lstm_units, return_state=True)
encoder_outputs2, state_h2, state_c2 = encoder_lstm2(encoder_outputs1)
encoder_states = [state_h2, state_c2]

decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm1 = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs1, _, _ = decoder_lstm1(decoder_embedding, initial_state=encoder_states)
decoder_lstm2 = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs2, _, _ = decoder_lstm2(decoder_outputs1)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs2)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 使用Adam优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

# 使用提前停止
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
history = model.fit(
    [encoder_input_data, decoder_input_data],
    train_target_data,
    batch_size=128,
    epochs=100,
    validation_data=([val_encoder_input_data, val_decoder_input_data], val_target_data),
    callbacks=[early_stopping]
)

# 保存模型
model.save('seq2seq_model.h5')

# 编码器模型
encoder_model = Model(encoder_inputs, encoder_states)

# 解码器模型
decoder_state_input_h = Input(shape=(lstm_units,))
decoder_state_input_c = Input(shape=(lstm_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_embedding2 = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
decoder_inputs2 = Input(shape=(None,))
decoder_embedding = decoder_embedding2(decoder_inputs2)
decoder_lstm1 = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs1, state_h, state_c = decoder_lstm1(decoder_embedding, initial_state=decoder_states_inputs)
decoder_lstm2 = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs2, _, _ = decoder_lstm2(decoder_outputs1)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs2] + decoder_states_inputs, [decoder_outputs] + [state_h, state_c])

# 定义句子生成函数
def decode_sequence(input_seq, max_length):
    # 将输入序列编码到状态向量
    states_value = encoder_model.predict(input_seq)

    # 生成一个目标序列的起始字符
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<START>']

    # 存储解码后的句子
    stop_condition = False
    decoded_sentence = ''
    generated_words = set()
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 获取预测的token的索引
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        # 防止生成重复词
        while sampled_token_index in generated_words:
            output_tokens[0, -1, sampled_token_index] = 0
            sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_char = tokenizer.index_word.get(sampled_token_index, '')

        # 如果预测的词不存在于词汇表中，退出循环
        if not sampled_char:
            stop_condition = True
            continue

        decoded_sentence += ' ' + sampled_char
        generated_words.add(sampled_token_index)

        # 退出条件：达到最大长度或找到结束字符
        if (sampled_char == '<END>' or len(decoded_sentence.split()) > max_length):
            stop_condition = True

        # 更新目标序列
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # 更新状态
        states_value = [h, c]

    return decoded_sentence.strip()

# 测试生成函数
def generate_text(input_text, max_length):
    # 对输入文本进行分词
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')
    decoded_sentence = decode_sequence(input_seq, max_length)
    return decoded_sentence

# 示例输入
input_text = "张三丰"
output_length = 10  # 可以自由调整生成段落的长短
output_text = generate_text(input_text, output_length)
print("输入:", input_text)
print("输出:", output_text)
