# import required modules
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify

import cv2
import numpy as np

app = Flask(__name__, template_folder="./")
vc = cv2.VideoCapture(0)

# ddepth = cv2.CV_16S
ddepth = cv2.CV_64F
kernel_size = 5


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')
    if request.method == 'POST':
        if request.form['submit']:
            return redirect(url_for('test'))
        if request.form['submit_b']:
            return redirect(url_for('test1'))
        if request.form['submit_c']:
            return redirect(url_for('test2'))


@app.route('/test', methods=['GET', 'POST'])
def test():
    return render_template('test.html')
    if request.method == 'POST':
        if request.form['submit']:
            return redirect(url_for('index'))
        if request.form['submit_b']:
            return redirect(url_for('test1'))
        if request.form['submit_c']:
            return redirect(url_for('test2'))


@app.route('/test1', methods=['GET', 'POST'])
def test1():
    return render_template('test1.html')
    if request.method == 'POST':
        if request.form['submit']:
            return redirect(url_for('index'))
        if request.form['submit_b']:
            return redirect(url_for('test'))
        if request.form['submit_c']:
            return redirect(url_for('test2'))


@app.route('/test2', methods=['GET', 'POST'])
def test2():
    return render_template('test2.html')
    if request.method == 'POST':
        if request.form['submit']:
            return redirect(url_for('index'))
        if request.form['submit_b']:
            return redirect(url_for('test'))
        if request.form['submit_c']:
            return redirect(url_for('test1'))


def genCanny(lowthresh, highthresh):
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    while True:
        rval, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur = cv2.medianBlur(gray, 5)
        blur = cv2.GaussianBlur(gray, (5, 5), 5)
        # canny
        edges = cv2.Canny(blur, lowthresh, highthresh)
        cv2.imwrite('pic.jpg', edges)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open(
                   'pic.jpg', 'rb').read() + b'\r\n')


def genLapl():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    while True:
        rval, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #laplacian
        dst = cv2.Laplacian(gray, ddepth, kernel_size)
        abs_dst = cv2.convertScaleAbs(dst)

        cv2.imwrite('pic.jpg', abs_dst)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open(
                   'pic.jpg', 'rb').read() + b'\r\n')


def genSobel():
    cap = cv2.VideoCapture(0)
    while True:
        rval, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # sobel operator
        sobelx8u = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        # sobelx64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        abs_sobel8u = np.absolute(sobelx8u)
        sobel_8u = np.uint8(abs_sobel8u)
        cv2.imwrite('pic.jpg', sobel_8u)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open(
                   'pic.jpg', 'rb').read() + b'\r\n')


def genHough():
    cap = cv2.VideoCapture(0)
    while True:
        rval, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is not None:

            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # for x1, y1, x2, y2 in lines[0]:
        #     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite('pic.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open(
                   'pic.jpg', 'rb').read() + b'\r\n')


@app.route('/index/canny_feed')
def canny_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(
        genCanny(50, 50), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/index/lapl_feed')
def lapl_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        genLapl(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/index/sobel_feed')
def sobel_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        genSobel(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/index/hough_feed')
def hough_feed():
    return Response(
        genHough(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
