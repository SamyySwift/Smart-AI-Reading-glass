import pytesseract
from PIL import Image
import os
import speech_recognition as sr
import pyttsx3
import cv2 as cv
import imutils
from threading import Thread
from transform import *


# Initialize libraries
r = sr.Recognizer()
engine = pyttsx3.init()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


# Set global variable
capture = False
terminate = False

def speak(text):
    voices = engine.getProperty('voices')
    engine.setProperty('rate', 150)
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()


def speech_recognition():
    global capture

    while True:
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)

        # recognize speech using Google Speech Recognition
        try:
            text_output = r.recognize_google(audio)
            print(f"Google Speech Recognition thinks you said: {text_output}")

             # Check for specific keywords
            if "snap" in text_output:
                capture = True
            elif "stop" in text_output:
                break
            elif "terminate" in text_output:
                terminate = True
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service {e}")
        
       

x = Thread(target=speech_recognition)
x.start()



def recognize_page(image):


    img = cv.imread(image)
    ratio = img.shape[0] / 500.0
    orig_img = img.copy()
    img = imutils.resize(img, height=500)


    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_blurred = cv.GaussianBlur(gray, (5, 5), 0)
    canny_edged = cv.Canny(gray_blurred, 100, 200)


    cnts = cv.findContours(canny_edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]

    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02*peri, True)
        
        if len(approx) == 4:
            screenCnt = approx
            break
        
        
    # apply the four point transform to obtain a top-down view of the
    # original image
    warped = four_point_transform(orig_img, screenCnt.reshape(4, 2) * ratio)
    warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    _, warped_thesh = cv.threshold(warped_gray, 140, 255, 3)
    # adpative_thresh = cv.adaptiveThreshold(warped_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 5)

    
    print("Recoginizing printed Page...")
    result = pytesseract.image_to_string(warped_thesh)
    print(result)
    speak(result)



def recognize_printed_book(image):
    image = cv.imread(image)
    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_resized = imutils.resize(image, height = 500)

    gray = cv.cvtColor(image_resized, cv.COLOR_BGR2GRAY)
    # adpative_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    _, thresh = cv.threshold(gray, 140, 255, 0)


    print("Recoginizing printed Book...")
    print('-------------------------------')
    result = pytesseract.image_to_string(img_rgb)
    print(result)
    speak(result)
    
   

def read_printed_text(image):
    try:
        recognize_page(image)

    except:
        recognize_printed_book(image)



def main():
    global capture, terminate

    cam = cv.VideoCapture(0)

    while True:
        _, frame = cam.read()

        cv.imshow('Image Frame', frame)

        if capture == True:
            cv2.imwrite('picture.jpg', frame)
            
            
        
            read_printed_text('picture.jpg')
            capture = False

        
        if cv.waitKey(1) == ord("q") or terminate == True:
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
    

