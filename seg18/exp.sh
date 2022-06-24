

#mswin
# joint train deeplab+swin
python train_swin.py --dataset cata --arch swinPlus --log_name mswin \
 --t 4 --batch_size 8 --lr 3e-4 --gpu 0,1 --ver 0
# deeplab init
python train_swin.py --dataset endovis2018 --arch swinPlus --log_name mswin --pre_log_name xxx \
--t 4 --batch_size 8 --lr 3e-5 --gpu 0,1 --ver 0
#note: The whole framework (Resnet18+Swin) can be trained jointly; 
#or Resnet18 can be firstly trained separately to provide a good parameter init (using --arch puredeeplab), and then jointly trained with Swin part


#after CL
python train_CL_ft_mswin_sgd_minput.py --dataset endovis2018 --arch swinPlus --log_name mswin_CL --pre_log_name mswinv1_minput_ver_2 --pre_checkpoint 140 \
--t 4 --batch_size 8 --lr 1e-3 --num_epochs 200 --gpu 0,1 --ver 1


#test
python test.py --arch swinPlus --log_name mswin_CL_ver_3 --t 4 --gpu 0 --checkpoint xx

