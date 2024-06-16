from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 使用GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 创建数据集
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="cleaned_corpus.txt",
    block_size=32,  # 进一步减少block_size
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 训练模型
training_args = TrainingArguments(
    output_dir="./gpt2_jin_yong",
    overwrite_output_dir=True,
    num_train_epochs=1,  # 减少训练周期数
    per_device_train_batch_size=16,  # 增加批量大小
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

def generate_text_transformer(seed_text, next_words, model, tokenizer):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt')
    output = model.generate(
        input_ids, 
        max_length=next_words + len(input_ids[0]), 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # 调整温度参数
        top_k=50,  # 调整top_k参数
        top_p=0.9,  # 调整top_p参数
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 示例输入和输出
seed_text = "张三丰"
next_words = 100  # 生成文本的长度
output_text = generate_text_transformer(seed_text, next_words, model, tokenizer)
print("输入:", seed_text)
print("输出:", output_text)
