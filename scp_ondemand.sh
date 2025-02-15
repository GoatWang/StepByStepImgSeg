cd /home/wanghsuanchung/Projects/StepByStepImgSeg
scp coco_dataset/*zip wanghsuanchung@51214.od.fbinfra.net:/data/sandcastle/boxes/fbsource/fbcode/gsst_pcd/FloorPlanRecognition/LLaVA/CocoDataset/data/
scp -r tasks_data_preparation wanghsuanchung@51214.od.fbinfra.net:/data/sandcastle/boxes/fbsource/fbcode/gsst_pcd/FloorPlanRecognition/LLaVA/CocoDataset/

cd /home/wanghsuanchung/Projects/StepByStepImgSeg/temp/LLaVA
scp -r ./checkpoints/vicuna-7b-v1.5-pretrain wanghsuanchung@51214.od.fbinfra.net:/data/sandcastle/boxes/fbsource/fbcode/gsst_pcd/FloorPlanRecognition/LLaVA/checkpoints/vicuna-7b-v1.5-pretrain


scp -r /home/wanghsuanchung/.cache/huggingface wanghsuanchung@76164.od.fbinfra.net:/home/wanghsuanchung/.cache/huggingface

