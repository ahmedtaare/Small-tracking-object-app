import cv2

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open webcam or there are no webcam.")
        return

    # Reading the first frame 
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        cap.release()
        return

    # here the user can select Region of Interest
    bbox = cv2.selectROI("Select Object to Track", frame, False)
    cv2.destroyWindow("Select Object to Track")

    # here we can create open cv tracker and i guess CSRT is more accurate for this task
    tracker = cv2.TrackerCSRT_create()

    # first frame and bounding box
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update tracker
        success, bbox = tracker.update(frame)

        if success:
            # Draw bounding box if tracking is successfull
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Display tracking failure message
            cv2.putText(frame, "Lost Track", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Object Tracker", frame)

        # Exit with e key
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    # Clean
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
