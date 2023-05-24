# -*-coding: utf-8 -*-
"""
    @Author : 
    @E-mail : 
    @Date   : 2023-05-16 08:53:11
    @Brief  :
"""
import flask, os, sys, time
from flask import Flask, render_template, request, make_response
import func

interface_path = os.path.dirname(__file__)
sys.path.insert(0, interface_path)

app = flask.Flask(__name__, template_folder=interface_path)


@app.route('/', methods=['get'])
def index():
    return render_template("index.html")


@app.route('/upload', methods=['post'])
def upload():
    fname = request.files['img']
    print("@@@@@@@@@@@@@@@@@@@@@@")
    print(fname.filename)
    newName = r'static/upload/' + fname.filename
    fname.save(newName)
    func.getRet(newName)
    # image_data = open("ldh_output.jpg", "rb").read()
    # response = make_response(image_data)
    # response.headers['Content-Type'] = 'image/jpg'
    # return response

    return render_template('result.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug='True')
