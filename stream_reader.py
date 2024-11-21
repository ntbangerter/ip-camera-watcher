import cv2
from transformers import ViTImageProcessor, ViTForImageClassification

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224', device_map='cuda')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', device_map='cuda')

stream = cv2.VideoCapture('http://rpi0.local:8000/stream.mjpg')

while ( stream.isOpened() ):
    # Read a frame from the stream
    ret, img = stream.read()
    if ret: # ret == True if stream.read() was successful
        cv2.imshow('preview', img)

        if cv2.waitKey(1) == ord('p'):
            inputs = processor(images=img, return_tensors='pt').to('cuda')
            outputs = model(**inputs)
            logits = outputs.logits
            # model predicts one of the 1000 ImageNet classes
            predicted_class_idx = logits.argmax(-1).item()
            print("Predicted class:", model.config.id2label[predicted_class_idx])

        if cv2.waitKey(1) == ord('q'):
            break

stream.release()
cv2.destroyAllWindows()
