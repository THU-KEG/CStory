
#新闻脉络关系分类任务训练与测试
#为了示范，我将所有的训练数据和测试数据都设置为1000条，实际训练或者测试的时候，开发者需要将数据文件替换成云盘里实际的数据文件。


#训练不包含关键句的分类模型
#BERT
cd CStory #进入根目录
python binary_classification/binary_classification/model_bert_finetune/train.py  --template_type base --train_data_path dataset/A_train.tsv  --dev_data_path dataset/A_dev.tsv --test_data_path dataset/A_test.tsv --out_dir binary_classification/output/no_key_sentence_bert   --model_path bert-base-chinese

#RoBERTa
python binary_classification/model_bert_finetune/train.py  --template_type base --train_data_path dataset/A_train.tsv  --dev_data_path dataset/A_dev.tsv --test_data_path dataset/A_test.tsv --out_dir output/no_key_sentence_roberta  --model_path hfl/chinese-roberta-wwm-ext
#获取模型在测试集上的效果
cd CStory #进入根目录
#load_path是你自己生成模型的路径
python binary_classification/model_bert_finetune/test.py --test_data_path dataset/A_test.tsv   --load_path  不包含关键句的测试文件的路径


#训练包含关键句子的分类模型
#BERT
cd CStory #进入根目录
python binary_classification/model_bert_finetune/train.py  --template_type base --train_data_path dataset/key_sentence_train.tsv  --dev_data_path dataset/key_sentence_dev.tsv --test_data_path dataset/key_sentence_test.tsv --out_dir output/key_sentence_bert   --model_path bert-base-chinese

#RoBERTa
cd CStory #进入根目录
python binary_classification/model_bert_finetune/train.py  --template_type base --train_data_path dataset/key_sentence_train.tsv  --dev_data_path dataset/key_sentence_dev.tsv  --test_data_path dataset/key_sentence_test.tsv --out_dir output/key_sentence_roberta   --model_path hfl/chinese-roberta-wwm-ext

#获取模型在测试集上的效果
cd CStory #进入根目录
#load_path是你自己生成模型的路径
python binary_classification/model_bert_finetune/test.py --test_data_path dataset/A_test.tsv   --load_path  包含关键句的测试文件的路径


#测试无监督模型结果
多种特征综合： CStory/binary_classification/mixed_feature/run_mixed_feature.py
PMI特征：CStory/binary_classification/PMI/textPMI.py
TFIDF:CStory/binary_classification/TFIDF/run_tfief.py
这三种方法直接启动对应文件就可以测试