

##swin
# joint train deeplab+swin
python train_cata_swin.py --dataset cata --arch swinPlus --log_name cata_swin \
 --t 4 --batch_size 24 --lr 3e-4 --gpu 0,1,2,3 --tag 1 --ver 0
# deeplab init
python train_cata_swin.py --dataset cata --arch swinPlus --log_name cata_swin --pre_log_name yourpreviouslogname \
--pre_checkpoint xxx --t 4 --batch_size 24 --lr 3e-5 --gpu 0,1,2,3 --tag 1 --ver 0

## after CL
python train_CL_ft_swin_sgd.py --dataset cata --arch swinPlus --log_name swinPlus_neg_CL_sgd --pre_log_name yourpreviouslogname --pre_checkpoint 140 \
--t 4 --batch_size 8 --lr 1e-3 --num_epochs 200 --gpu 0,1,2,3 --tag 1 --ver 0


##test
#base (deeplabv3+)
python cata_test.py --arch puredeeplab18 --log_name yourlogname --checkpoint xxx --tag 1 --gpu 0