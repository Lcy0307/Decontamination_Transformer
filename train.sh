# MPN
python train_MPN.py --output_dir mpn_imagenet_strokemask --dataset ffhq --iter 600001

# CPN & IIN
python train_CPN_IIN.py --dataset ffhq --output_dir RRtrans_gtmask_8_imagenet_stroke --use_gtmask --batch 8 --iter 600001

# cotrain
python train_cotrain.py --inpaint_ckpt RRtrans_gtmask_8_imagenet_stroke/checkpoint.pt --mask_ckpt mpn_imagenet_strokemask/checkpoint.pt --output_dir RRtrans_cotrain/checkpoint.pt --dataset imagenet --batch 8 --iter 200001