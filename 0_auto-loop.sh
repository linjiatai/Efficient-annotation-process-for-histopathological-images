VERSION=v1

python 1_annotation_RGB2P.py --version $VERSION
python 2_sampling.py --version $VERSION
python 3_train_DeeplabV3.py --version $VERSION
python 4_region_inference_with_DeeplabV3.py --version $VERSION
