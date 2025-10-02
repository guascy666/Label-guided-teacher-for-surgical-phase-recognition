mpath='/data2/guanjl/Swin_32_8_40/features_before_normal/40-40/cholec80swinv2b_epoch70_train_9973_val_8621_test_8212'


python Teacher_trainer.py -pd $mpath -bw 500 -dm 1024 -dr 4 -ld 0.1 -act 0 
python Student_trainer.py -pd $mpath -bw 500 -dm 1024 -nl 2 -dr 4 -dr_S 4 -ld 0.1 -act 0 -act_S 1  

