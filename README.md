# SceneUnderstandingViT

To download Dreyeve images:
```
aws s3 sync s3://dreyeve-source ./
```

To prepare the dataset:
```

```

To see data distribution:
```
for i in ./*/*; do echo $i; ls "$i" | wc -l; done
```

To train MPL teacher-student:
```
python semsup_learning/main.py \
--data-path=./data/dreyeve1k_mpl \
--name=SimpleViT-MPL \
--batch-size=6 \
--train-steps=500000 \
--eval-steps=1000 \
--device=cuda:0
```

To train SL model:
```
python sup_learning/main.py \
--name=SimpleViT-SL \
--data-path=./data/dreyeve1k_sup \
--batch-size=20 \
--train-steps=10000 \
--eval-steps=200 \
--device=cuda:1
```

To validate the model:
```
python sup_learning/validator.py \
--data-path=./data/dreyeve1k_sup \
--model-path=./checkpoints/tmp.pth \
--data-count=10 \
--batch-size=16
```
