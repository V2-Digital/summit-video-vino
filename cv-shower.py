import cv2
from args import get_args
from object_detection.people_counter import PeopleCounter

people_counter = PeopleCounter(get_args())

def frame_generator():
    while True:
        frame = people_counter.get_latest_frame()
        if frame is not None:
            cv2.imshow("frame", frame)
            # Wait for a key press for 1 millisecond
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit the loop if 'q' is pressed

if __name__ == "__main__":
    frame_generator()
    cv2.destroyAllWindows()  # Close all OpenCV windows