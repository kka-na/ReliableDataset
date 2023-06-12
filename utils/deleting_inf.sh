
CUDA_VISIBLE_DEVICES=0 python deleting_inf.py  $1 $2 a a &
CUDA_VISIBLE_DEVICES=1 python deleting_inf.py $1 $2 b b &
CUDA_VISIBLE_DEVICES=2 python deleting_inf.py $1 $2 c c
wait
CUDA_VISIBLE_DEVICES=0 python deleting_inf.py $1 $2 a b &
CUDA_VISIBLE_DEVICES=1 python deleting_inf.py $1 $2 a c &
CUDA_VISIBLE_DEVICES=2 python deleting_inf.py $1 $2 b a
wait 
CUDA_VISIBLE_DEVICES=0 python deleting_inf.py $1 $2 b c &
CUDA_VISIBLE_DEVICES=1 python deleting_inf.py $1 $2 c a & 
CUDA_VISIBLE_DEVICES=2 python deleting_inf.py $1 $2 c b 
wait
exit 0
