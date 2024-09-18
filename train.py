from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO("ultralytics/cfg/models/v8/yolov8-carafe.yaml")
    model.load("yolov8n.pt")

    # Train the model
    train_results = model.train(
        data="ultralytics/cfg/datasets/coco8.yaml",  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        batch=8,

    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model

if __name__ == "__main__":
    train()