cd ../../../
device=$1
ablation=$2
patch_size=$3
CUDA_VISIBLE_DEVICES=${device}  python                  main.py                     \
                                --dataset               mnist                       \
                                --data                  ~/datasets                  \
                                --arch                  lenet                       \
                                --epochs                20                          \
                                --lr                    0.1                         \
                                --step-lr               8                           \
                                --drop-rate             0                           \
                                --batch-size            128                         \
                                --weight-decay          5e-4                        \
                                --adv-train             0                           \
                                --freeze-level          -1                          \
                                --ablation-type         col                         \
                                --ablation-size         ${ablation}                 \
                                --patch-size            ${patch_size}               \
                                --ablate-input                                      \
                                --random-patch
                                