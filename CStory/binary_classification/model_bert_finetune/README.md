

#新闻脉络关系分类任务测试

#训练不包含关键句的分类模型
#BERT

cd CStory #进入根目录
python binary_classification/model_bert_finetune/train.py  --template_type base --train_data_path dataset/A_train.tsv  --dev_data_path dataset/A_dev.tsv --test_data_path dataset/A_test.tsv --out_dir binary_classification/output/no_key_sentence_bert   --model_path bert-base-chinese

#RoBERTa
python model_bert_finetune/train.py  --template_type base --train_data_path dataset/A_train.tsv  --dev_data_path dataset/A_dev.tsv --test_data_path dataset/A_test.tsv --out_dir output/no_key_sentence_roberta  --model_path hfl/chinese-roberta-wwm-ext



#训练包含关键句子的分类模型
#BERT
cd CStory #进入根目录
python model_bert_finetune/train.py  --template_type base --train_data_path /data/skj/data/key_sentence_train.tsv  --dev_data_path dataset/key_sentence_dev.tsv --test_data_path dataset/key_sentence_test.tsv --out_dir output/key_sentence_bert   --model_path bert-base-chinese

#RoBERTa
cd CStory #进入根目录
python model_bert_finetune/train.py  --template_type base --train_data_path /data/skj/data/key_sentence_train.tsv  --dev_data_path /data/skj/data/key_sentence_dev.tsv --test_data_path /data/skj/data/key_sentence_test.tsv --out_dir output/key_sentence_roberta   --model_path hfl/chinese-roberta-wwm-ext

#获取模型在测试集上的效果
cd CStory #进入根目录
#load_path是你自己生成模型的路径
python model_bert_finetune/test.py --test_data_path dataset/A_test_keysentence.tsv   --load_path  binary_classification/output/key_sentence_roberta/base/epoch_2_dev_97.2846_test_97.3439_train_93.24.ckpt



