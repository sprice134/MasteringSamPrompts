# MasteringSamPrompts

conda create -n PromptEnv python=3.10
conda activate PromptEnv
pip install matplotlib transformers scikit-image opencv-python timm pandas ultralytics
pip install torch torchvision
pip install pycocotools

cd segment-anything/
pip install -e .

cd ..
mkdir checkpoints
wget -O checkpoints/sam_vit_l.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget -O checkpoints/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth