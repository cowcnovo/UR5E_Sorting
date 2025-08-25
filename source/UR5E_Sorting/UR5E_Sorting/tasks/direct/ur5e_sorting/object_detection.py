import torch
from ultralytics import YOLO
from isaaclab.sensors import TiledCamera

_model = None

def load_model():
    global _model
    if _model is None:
        _model = YOLO("/home/ubuntu/Desktop/yolo/runs/detect/train/weights/best.pt")
    return _model

def inference(
    camera: TiledCamera
) -> torch.Tensor:

    # Set the device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtaining the bounding boxes
    images = camera.data.output["rgb"]
    images = images.permute(0, 3, 1, 2).to(torch.float32).to(device) / 255.0

    model = load_model()
    model.to(device)
    results = model.predict(source=images, verbose=False)

    bounding_boxes = torch.zeros(len(images), 4, device=device)
    for i in range(len(results)):
        if len(results[i].boxes.xywhn) > 0:
            boxesLocations = torch.tensor(results[i].boxes.xywhn[0], device=device)  # Take the first bounding box only
            bounding_boxes[i] = boxesLocations

    # Depth estimation
    depth_data = camera.data.output["depth"]
    depth_data = depth_data.permute(0, 3, 1, 2).to(torch.float32).to(device)

    object_distances = []
    image_width = depth_data.shape[3]
    image_height = depth_data.shape[2]
    for i in range(len(bounding_boxes)):
        x_center, y_center, _, _ = bounding_boxes[i]
        x_center_pixels = int(x_center * image_width)
        y_center_pixels = int(y_center * image_height)
        depth = depth_data[i, 0, y_center_pixels, x_center_pixels]
        if depth > 0:  # Check if depth is valid
            object_distances.append(depth.item())
        else:
            object_distances.append(0.8)

    object_distances = torch.tensor(object_distances, device=device)

    # Use object image bounding box and depth to calculate object world coordinates
    K = camera.data.intrinsic_matrices
    object_coords = torch.zeros((len(bounding_boxes), 3), device=device)
    for i in range(len(bounding_boxes)):
        x_center, y_center, _, _ = bounding_boxes[i]
        u = int(x_center * image_width)
        v = int(y_center * image_height)
        z = object_distances[i]

        p = torch.tensor([u, v, 1.0], device=device)
        inv_K = torch.linalg.inv(K[0]) # Taking the first camera's intrinsic matrix
        p_c = (inv_K @ p) * z

        object_coords[i] = p_c 

    # Printing
    # print("\n\nResults for Env 1:")
    # print(f"Number of detected objects: {len(results[0].boxes.xywhn)}")
    # if len(results[0].boxes.xywhn) > 0:
    #     print(f"Class of Best: {results[0].boxes.cls[0]}")
    #     print(f"Confidence of Best: {results[0].boxes.conf[0]}")
    #     print(f"Bounding Box of Best: {bounding_boxes[0]}")
    # else:
    #     print("No objects detected.")
       
    # print(f"Shape of Depth Image: {depth_data.shape}")

    # if len(object_distances) > 0:
    #     print(f"Object Distance: {object_distances[0]}")

    return object_coords