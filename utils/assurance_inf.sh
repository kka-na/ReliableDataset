
# CUDA_VISIBLE_DEVICES=0 python assurance_inf.py KITTI before a &
# CUDA_VISIBLE_DEVICES=1 python assurance_inf.py KITTI before b &
# CUDA_VISIBLE_DEVICES=2 python assurance_inf.py KITTI after a 
# wait
# CUDA_VISIBLE_DEVICES=0 python assurance_inf.py KITTI after b &
# CUDA_VISIBLE_DEVICES=1 python assurance_inf.py WAYMO before a &
# CUDA_VISIBLE_DEVICES=2 python assurance_inf.py WAYMO before b
# wait 
# CUDA_VISIBLE_DEVICES=0 python assurance_inf.py WAYMO after a &
# CUDA_VISIBLE_DEVICES=1 python assurance_inf.py WAYMO after b 
CUDA_VISIBLE_DEVICES=0 python assurance_inf.py AIHub1 before a &
CUDA_VISIBLE_DEVICES=1 python assurance_inf.py AIHub1 before b &
CUDA_VISIBLE_DEVICES=2 python assurance_inf.py AIHub1 after a 
wait
CUDA_VISIBLE_DEVICES=0 python assurance_inf.py AIHub1 after b &
CUDA_VISIBLE_DEVICES=0 python assurance_inf.py AIHub2 before a &
CUDA_VISIBLE_DEVICES=1 python assurance_inf.py AIHub2 before b 
wait
CUDA_VISIBLE_DEVICES=0 python assurance_inf.py AIHub2 after a &
CUDA_VISIBLE_DEVICES=1 python assurance_inf.py AIHub2 after b 
wait
exit 0
