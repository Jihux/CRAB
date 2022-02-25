cd ../../../
device=$1
arch=$2
ablation=$3
patch_size=$4
batch=$5
CUDA_VISIBLE_DEVICES=${device}  python                  main.py                     \
                                --dataset               cifar10                     \
                                --cifar-preprocess-type simple224                   \
                                --data                  ~/datasets                  \
                                --arch                  deit_${arch}_patch16_224    \
                                --adv-train             0                           \
                                --freeze-level          -1                          \
                                --ablation-type         col                         \
                                --ablation-size         ${ablation}                 \
                                --patch-size            ${patch_size}               \
                                --batch-size            ${batch}                    \
                                --eval-only             1                           \
                                --certify-mode          col                         \
                                --certify-ablation-size ${ablation}                 \
                                --certify-patch-size    ${patch_size}               \
                                --drop-tokens                                       \
                                --resume                                            \
                                --certify                                           \
                                --random-patch