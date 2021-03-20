#Data Augmentation
python data_augmentation.py 
	--pretrained_bert_model ${BERT_BASE_DIR}$ \   #download link: https://huggingface.co/bert-base-uncased
	--glove_embs ${GLOVE_EMB}$ \                         # you may use stanford glove 
                --glue_dir ${GLUE_DIR}$ \  		      
                --task_name ${TASK_NAME}$

#intermediate layer distillation
python task_distill.py 
	--teacher_model ${FT_BERT_BASE_DIR}$ \                      # you can find all 8 models here https://huggingface.co/textattack
	--student_model ${GENERAL_TINYBERT_DIR}$ \             #provided by: https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT
	--data_dir ${TASK_DIR}$ \
	--task_name ${TASK_NAME}$ \ 
	--output_dir ${TMP_TINYBERT_DIR}$ \
	--max_seq_length 128 \
	--train_batch_size 32 \
	--num_train_epochs 10 \
	--aug_train \
	--do_lower_case


#prediction layer distillation
python task_distill.py 
	--pred_distill  \
 	--teacher_model ${FT_BERT_BASE_DIR}$ \
	--student_model ${TMP_TINYBERT_DIR}$ \
	--data_dir ${TASK_DIR}$ \
	--task_name ${TASK_NAME}$ \
	--output_dir ${TINYBERT_DIR}$ \
	--aug_train  \  
	--do_lower_case \
	--learning_rate 3e-5  \
	--num_train_epochs  3  \
	--eval_step 100 \
	--max_seq_length 128 \
	--train_batch_size 32

#Evaluation
python task_distill.py --do_eval \
	--student_model ${TINYBERT_DIR}$ \ # model to be tested
	--data_dir ${TASK_DIR}$ \
	--task_name ${TASK_NAME}$ \
	--output_dir ${OUTPUT_DIR}$ \
	--do_lower_case \
	--eval_batch_size 32 \
	--max_seq_length 128 