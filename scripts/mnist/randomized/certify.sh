cd ../../../
device=$1
ablation=$2
patch_size=$3
CUDA_VISIBLE_DEVICES=${device}  python                  main.py                     \
                                --dataset               mnist                       \
                                --data                  ~/datasets                  \
                                --arch                  lenet                       \
                                --adv-train             0                           \
                                --freeze-level          -1                          \
                                --ablation-type         col                         \
                                --ablation-size         ${ablation}                 \
                                --patch-size            ${patch_size}               \
                                --batch-size            10000                       \
                                --eval-only             1                           \
                                --certify-mode          col                         \
                                --certify-ablation-size ${ablation}                 \
                                --certify-patch-size    ${patch_size}               \
                                --resume                                            \
                                --certify                                           \
                                --random-patch