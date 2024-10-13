# https://docs.ultralytics.com/tasks/detect/#can-i-validate-my-yolo11-model-using-a-custom-dataset
# https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation
# https://docs.ultralytics.com/modes/predict/

from ultralytics import YOLO
import cv2
import os, glob


class Model:
    def __init__(self, model_path):
        pass

    def update(self):
        """Fine-tuning pre-trained model and save the weights"""
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class YoloModel(Model):
    def __init__(self, model_path=None):
        # Load the pre-trained YOLOv11 model if model_path None
        self.model = YOLO("yolo11n.pt") if model_path is None else YOLO(model_path)

    def update(
        self,
        data="/dataset/augmented_data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        save_dir="/dataset/",
    ):

        self.resize = imgsz

        # Train the pre-trained model and save the best weights
        if save_dir:
            self.model.train(
                data=data,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                save_dir=save_dir,
            )
            self.model_path = os.path.join(save_dir, "weights/best.pt")
        else:
            self.model.train(
                data=data, epochs=epochs, imgsz=imgsz, batch=batch, workers=workers
            )
            self.model_path = "/runs/train/exp/weights/best.pt"

        # Validate the model on the test set (if specified in data.yaml)
        metrics = self.model.val()
        print(f"Model mAP: {metrics.box.map}")

    def resize_image(self, image):
        """Resize image to the target size, return the resized image and its original dimensions."""
        h, w = image.shape[:2]
        resized_image = cv2.resize(image, (self.resize, self.resize))
        return resized_image, (h, w)

    def rescale_boxes(self, boxes, original_shape):
        """Rescale bounding boxes to the original image size."""
        orig_h, orig_w = original_shape
        scale_x = orig_w / self.resize
        scale_y = orig_h / self.resize

        # Rescale bounding boxes
        boxes[:, [0, 2]] *= scale_x  # Rescale x-coordinates
        boxes[:, [1, 3]] *= scale_y  # Rescale y-coordinates

        return boxes

    def predict_one(self, image_path, output_dir):
        """Predict bounding boxes on a single image and save results."""
        # Load and resize the image
        img = cv2.imread(image_path)
        resized_img, original_shape = self.resize_image(img)

        # Perform inference on the resized image
        results = self.model(resized_img)

        # Rescale bounding boxes to the original image size
        boxes = results[0].boxes.xyxy.cpu().numpy()
        boxes = self.rescale_boxes(boxes, original_shape)

        # Draw bounding boxes on the original image
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Save the image with bounding boxes
        fname = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_dir, fname), img)

    def predict(self, image_dir="/dataset/images/test", output_dir="/image/output/"):
        """Batch predict bounding boxes on multiple images."""

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.join(output_dir, "yolo"), exist_ok=True)

        # Get all images in the directory
        image_paths = glob.glob(os.path.join(image_dir, "*.png"))

        # Use YOLO's batch processing for inference
        results = self.model(image_paths, imgsz=self.resize)

        # Loop over results and save images with bounding boxes
        for result in results:
            img_with_boxes = result.plot()
            fname = os.path.basename(result.path)
            cv2.imwrite(os.path.join(output_dir, fname), img_with_boxes)


"""
todo: build digit and letter classification model
"""
