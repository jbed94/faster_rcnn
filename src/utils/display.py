import cv2
import seaborn as sns

bbox_palette = sns.color_palette('bright', 10)


def draw_bounding_boxes(image, bboxes, labels=None, font=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=1):
    h, w, _ = image.shape

    for i in range(len(bboxes)):
        i_bbox = bboxes[i]
        color = [int(c * 255) for c in bbox_palette[i % len(bbox_palette)]]
        start_point = int(i_bbox[1] * w), int(i_bbox[0] * h)
        end_point = int(i_bbox[3] * w), int(i_bbox[2] * h)
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

        if labels is not None:
            text_point = (start_point[0], start_point[1] - thickness * 3)
            image = cv2.putText(image, str(labels[i]), text_point, font, fontScale, color, thickness, cv2.LINE_AA, False)

    return image
