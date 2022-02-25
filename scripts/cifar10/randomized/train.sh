cd ../../../
device=$1
arch=$2
ablation=$3
patch_size=$4
CUDA_VISIBLE_DEVICES=${device}  python                  main.py                     \
                                --dataset               cifar10                     \
                                --cifar-preprocess-type simple224                   \
                                --data                  ~/datasets                  \
                                --arch                  deit_${arch}_patch16_224    \
                                --epochs                30                          \
                                --lr                    0.01                        \
                                --step-lr               10                          \
                                --batch-size            128                         \
                                --weight-decay          5e-4                        \
                                --drop-rate             0.3                         \
                                --adv-train             0                           \
                                --freeze-level          -1                          \
                                --ablation-type         col                         \
                                --ablation-size         ${ablation}                 \
                                --patch-size            ${patch_size}               \
                                --pytorch-pretrained                                \
                                --drop-tokens                                       \
                                --ablate-input                                      \
                                --random-patch
