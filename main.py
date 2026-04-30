import cv2 as cv

cv.namedWindow('frame', cv.WINDOW_AUTOSIZE)

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        k = cv.waitKey(1)
        if cv.getWindowProperty('frame', cv.WND_PROP_VISIBLE) < 1:
            break
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()
