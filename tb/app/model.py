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

    def load_model(self, model_path="/runs/train/exp/weights/best.pt"):
        self.model = YOLO(model_path)

    def update(
        self,
        data="/dataset/augmented_data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        save_dir=None,
        verbose=True,
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
                verbose=verbose,
            )
            self.model_path = os.path.join(save_dir, "weights/best.pt")
        else:
            self.model.train(
                data=data,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                verbose=verbose,
            )
            self.model_path = "/runs/train/exp/weights/best.pt"

        # Validate the model on the test set (if specified in data.yaml)
        metrics = self.model.val(save_json=True)

    def predict(
        self,
        image_dir="/dataset/images/test",
        output_dir="/yolo_output",
        font_scale=0.9,
        font_thickness=1,
    ):
        """Batch predict bounding boxes on multiple images."""

        # Create the output directory if it doesn't exist
        os.makedirs(os.path.join(output_dir, "test_result"), exist_ok=True)

        # Get all images in the directory
        image_paths = glob.glob(os.path.join(image_dir, "*.png"))

        # Use YOLO's batch processing for inference
        results = self.model(image_paths)

        # Loop over results and save images with custom bounding boxes and labels
        for result in results:
            img = cv2.imread(result.path)  # Read the image

            # Loop over detected boxes in the result
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Get the class label and confidence score
                label = int(box.cls.cpu().numpy()[0])  # Extract label
                confidence = box.conf.cpu().numpy()[0]  # Extract confidence

                # Customize font, color, and thickness for bounding boxes and text
                box_color = (0, 0, 255)  # Red color for boxes
                text_color = (255, 0, 0)  # blue color for labels
                label_text = f"Class: {label} ({confidence:.2f})"

                # Draw the bounding box
                cv2.rectangle(
                    img, (x1, y1), (x2, y2), box_color, 2
                )  # Box thickness of 2

                # Put the label above the box without the white background
                text_size = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )[0]
                text_x, text_y = x1, (
                    y1 - 10 if y1 - 10 > 10 else y1 + 10
                )  # Prevent text from going out of bounds
                cv2.putText(
                    img,
                    label_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )

            # Save the image with bounding boxes
            fname = os.path.basename(result.path)
            output_path = os.path.join(output_dir, "test_result", fname)
            cv2.imwrite(output_path, img)


"""
todo: build digit and letter classification model
"""
