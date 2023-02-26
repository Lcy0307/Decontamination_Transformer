# MPN
#python train_MPN.py --dataset imagenet --output_dir mpn_imagenet_strokemask --iter 600001 --batch 16
python train_MPN.py --dataset places2 --output_dir mpn_places2_strokemask --iter 600001 --batch 16
#python train_MPN.py --dataset ffhq --output_dir mpn_ffhq_strokemask --iter 550001 --batch 16


# CPN & IIN
#python train_CPN_IIN.py --dataset imagenet --output_dir RRtrans_gtmask_8_imagenet_stroke --use_gtmask --batch 8 --iter 600001
python train_CPN_IIN.py --dataset places2 --output_dir RRtrans_gtmask_8_places2_stroke --use_gtmask --batch 8 --iter 600001 
#python train_CPN_IIN.py --dataset ffhq --output_dir RRtrans_gtmask_8_ffhq_stroke --use_gtmask --batch 8 --iter 550001

# cotrain
#python train_cotrain.py --dataset imagenet --inpaint_ckpt RRtrans_gtmask_8_imagenet_stroke/checkpoint.pt --mask_ckpt mpn_imagenet_strokemask/checkpoint.pt --output_dir RRtrans_cotrain_imagenet --batch 8 --iter 200001
python train_cotrain.py --dataset places2 --inpaint_ckpt RRtrans_gtmask_8_places2_stroke/checkpoint.pt --mask_ckpt mpn_places2_strokemask/checkpoint.pt --output_dir RRtrans_cotrain_places2 --batch 8 --iter 200001
#python train_cotrain.py --dataset ffhq --inpaint_ckpt RRtrans_gtmask_8_ffhq_stroke/checkpoint.pt --mask_ckpt mpn_ffhq_strokemask/checkpoint.pt --output_dir RRtrans_cotrain_ffhq --batch 8 --iter 150001