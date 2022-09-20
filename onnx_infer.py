import onnxruntime as rt
import cv2
import numpy as np
from scipy.special import softmax
class P2PNet():
    def __init__(self, modelPath, confThreshold=0.5):
        self.model = rt.InferenceSession(modelPath)
        self.inputNames = 'input'
        self.outputNames = ['pred_logits', 'pred_points']
        self.confThreshold = confThreshold
        self.mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
        self.std_ = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))
    def detect(self, srcimg):
        resizeimg = cv2.resize(srcimg, (640, 640), interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(resizeimg, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_resize = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)
        img = (img_resize.astype(np.float32) / 255.0 - self.mean_) / self.std_
        img=img.transpose((2,0,1))
        img = np.expand_dims(img, axis=0)
        print(img.shape)
        # Forward
        outputBlob = self.model.run(None,{"input":img})
        outputs_scores=softmax(outputBlob[0], axis=-1)[:, :, 1][0]
        outputs_points=outputBlob[1]
        scores = outputs_scores[outputs_scores > self.confThreshold]
        points = outputs_points[0][outputs_scores > self.confThreshold]
        ratioh, ratiow = srcimg.shape[0]/img_resize.shape[0], srcimg.shape[1]/img_resize.shape[1]
        points[:, 0] *= ratiow
        points[:, 1] *= ratioh
        return scores, points
if __name__=='__main__':
    srcimg = cv2.imread("2022061510522205_edt9f7.jpg")
    net = P2PNet("taipanlan_wights_h640_w640.onnx", confThreshold=0.4)
    scores, points = net.detect(srcimg)
    print('have', scores.shape[0], 'cell')
    for p in points:
        img_to_draw = cv2.circle(srcimg, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
    cv2.imwrite("result.jpg", img_to_draw)
    srcimg = cv2.imread("2022061514585568_36hivm.jpg")
    net = P2PNet("Ctypb_huizhong_weights_h640_w640.onnx", confThreshold=0.4)
    scores, points = net.detect(srcimg)
    print('have', scores.shape[0], 'cell')
    for p in points:
        img_to_draw = cv2.circle(srcimg, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
    cv2.imwrite("result2.jpg", img_to_draw)
