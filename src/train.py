from ultralytics import YOLO

model = YOLO('yolov8n.yaml').load('yolov8n.pt')

dataset = 'postprocessed2' # baseline. Generated with notebook/prepare_data/pytorch_datasets_dev.ipynb

data = f'/home/azureuser/Alpine-Aster/data/{dataset}/yolo_data.yaml'

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data=data, imgsz=1280, epochs=100)

# Evaluate the model's performance on the validation set
results = model.val()