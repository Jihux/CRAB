cd ../../../
device=$1
ablation=$2
CUDA_VISIBLE_DEVICES=${device}  python                  main.py                     \
                                --dataset               mnist                       \
                                --data                  ~/datasets                  \
                                --arch                  lenet                       \
                                --epochs                15                          \
                                --lr                    0.1                         \
                                --step-lr               5                           \
                                --drop-rate             0.4                         \
                                --batch-size            128                         \
                                --weight-decay          5e-4                        \
                                --adv-train             0                           \
                                --freeze-level          -1                          \
                                --ablation-type         col                         \
                                --ablation-size         ${ablation}                 \
                                --patch-size            0                           \
                                --ablate-input                                      \
                                --model-smoothing
                                