# import paket-paket yang diperlukan
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from pygame import mixer
import numpy as np
import playsound
import pygame
import argparse
import imutils
import time
import dlib
import cv2
import os
import winaudio


def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)
    #pygame.mixer.init()
    #pygame.mixer.music.load(path)
    #pygame.mixer.music.play()


def aspek_rasio_mata(mata):
    # Menghitung jarak euclidean antara dua set
    # landmark mata vertikal (x, y) -dikordinasikan
    A = dist.euclidean(mata[1], mata[5])
    B = dist.euclidean(mata[2], mata[4])

    # hitung jarak euclidean antara horizontal
    # mata landmark (x, y) -coordinate
    C = dist.euclidean(mata[0], mata[3])

    # menghitung rasio aspek mata
    arm = (A + B) / (2.0 * C)

    # mengembalikan rasio aspek mata
    return arm


# bangun argumen parse dan parsing argumen
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark prediksi")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="path alarm .WAV file")                
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

# tentukan dua konstanta, satu untuk rasio aspek mata untuk ditunjukkan
# berkedip dan kemudian konstanta kedua untuk jumlah berturut-turut
# membingkai mata harus di bawah ambang batas
thres_kedip = 0.2
batas_ambang = 50

# menginisialisasi penghitung bingkai dan jumlah total kedipan
hitung = 0
total = 0
ALARM_ON = False
# inisialisasi pendeteksi wajah dlib (berbasis HOG) lalu buat
# Prediktor tengara wajah
print("[INFO] loading facial landmark prediksi...")
detektor = dlib.get_frontal_face_detector()
prediksi = dlib.shape_predictor(args["shape_predictor"])

# ambil indeks landmark wajah untuk kiri dan
# mata kanan, masing-masing
(krMulai, krAkhir) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(knMulai, knAkhir) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# mulai utas streaming video
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

# lompati frame dari aliran video
while True:
    # jika ini adalah aliran video file, maka kita perlu memeriksa apakah
    # masih ada frame yang tersisa di buffer untuk diproses
    if fileStream and not vs.more():
        break

    # ambil bingkai dari aliran file video berulir, ubah ukuran
    # itu, dan ubah menjadi grayscale
    # saluran)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # mendeteksi wajah dalam bingkai abu-abu
    rects = detektor(gray, 0)

    # loop over the face detections
    for rect in rects:
        # tentukan landmark wajah untuk wilayah wajah, lalu
        # Konversikan landmark wajah (x, y) -dikordinasikan ke NumPy
        # Himpunan
        bentuk = prediksi(gray, rect)
        bentuk = face_utils.shape_to_np(bentuk)

        # ekstrak koordinat mata kiri dan kanan, lalu gunakan
        # Koordinat untuk menghitung rasio aspek mata untuk kedua mata
        matakanan = bentuk[krMulai:krAkhir]
        matakiri = bentuk[knMulai:knAkhir]
        armkiri = aspek_rasio_mata(matakanan)
        armkanan = aspek_rasio_mata(matakiri)

        # rata-rata rasio aspek mata bersama untuk kedua mata
        arm = (armkiri + armkanan) / 2.0

        # hitung cembung lambung untuk mata kiri dan kanan, lalu
        # Visualisasikan setiap mata
        frame_matakanan = cv2.convexHull(matakanan)
        frame_matakiri = cv2.convexHull(matakiri)
        cv2.drawContours(frame, [frame_matakanan], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [frame_matakiri], -1, (0, 255, 0), 1)

        # periksa untuk melihat apakah rasio aspek mata di bawah blink
        # threshold, dan jika ya, tambahkan penghitung bingkai blink
        if arm < thres_kedip:
            hitung += 1

        # jika tidak, rasio aspek mata tidak di bawah blink
        # threshold
        #else:
            # jika mata ditutup untuk jumlah yang cukup
            # lalu menambah jumlah total kedipan
            if hitung >= batas_ambang:
                if not ALARM_ON:
                    ALARM_ON = True
                
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                           args=(args["alarm"],))
                        t.deamon = True
                        t.start()
                #s = Sound() 
                #s.read('alarm1.wav') 
                #s.play() 
                #pygame.mixer.init()
                #pygame.mixer.music.load("alarm1.wav")
                #pygame.mixer.music.play()
                #winsound.PlaySound(r'04321024C-Rancang_bangun_deteksi_kantuk_berbasis_facial_landmark_menggunakan_Dlib_dan_Opencv\deteksi ngantuk alarm.wav', winsound.SND_ASYNC)
                #open('alarm1.wav')
                #os.startfile('alarm1.wav')
                #os.system(r'D:\hadi\deteksi ngantuk alarm1.wav')
                        #draw an alarm on the frame
                cv2.putText(frame, "NGANTUK NDANG BUBUQ!", (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # while total >= 5
            # print ('supir mengantuk')
            # mengatur ulang penghitung bingkai mata
        else:
            hitung = 0
            #pygame.mixer.music.stop
            #winsound.PlaySound(None, winsound.SND_PURGE)
            #os.stopfile('alarm1.wav')
            ALARM_ON = False
        # gambarkan jumlah total kedipan pada bingkai bersama dengan
        # rasio aspek mata yang dihitung untuk bingkai
        #cv2.putText(frame, "Berkedip: {}".format(total), (10, 30),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, "Rasio Mata: {:.2f}".format(arm), (125, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # cv2.putText(frame, "Rasio Mata: ".format(total2), (125, 300),
    # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # tunjukkan bingkai
    cv2.imshow("Hasil Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # jika tombol `q` ditekan, patahkan dari loop
    if key == ord("q"):
        break

# lakukan sedikit pembersihan
cv2.destroyAllWindows()
vs.stop()
