import os
import time
from pyDes import *


def encrypt(key, model_file_path, encrypt_file_path):
    start = time.time()
    print('Read model {}'.format(model_file_path))
    with open(model_file_path, 'rb') as model_file:
        model_bytes = model_file.read()
    print('Read model {} time: {}'.format(model_file_path, time.time()-start))
    start = time.time()
    encrypt_model = key.encrypt(model_bytes)
    print('Encrypt model {} time: {}'.format(model_file_path, time.time()-start))
    start = time.time()
    with open(encrypt_file_path, 'wb') as encrypt_file:
        encrypt_file.write(encrypt_model)
    print('Write model {} time: {}'.format(model_file_path, time.time()-start))
    print('Encrypt model {} finish'.format(model_file_path))


def decrypt(key, encrypt_file_path, decrypt_file_path):
    start = time.time()
    print('Read model {}'.format(encrypt_file_path))
    with open(encrypt_file_path, 'rb') as encrypt_file:
        encrypt_model = encrypt_file.read()
    print('Read model {} time: {}'.format(encrypt_file_path, time.time() - start))
    start = time.time()
    model_bytes = key.decrypt(encrypt_model)
    print('Decrypt model {} time: {}'.format(encrypt_file_path, time.time() - start))
    with open(decrypt_file_path, 'wb') as encrypt_file:
        encrypt_file.write(model_bytes)
    print('Decrypt model {} finish'.format(encrypt_file_path))


if __name__ == '__main__':
    model_dir = './'
    des_key = 'abcdefgh'
    iv = '01010101'
    key = des(des_key, CBC, iv, pad=None, padmode=PAD_PKCS5)
    model_names = [
        'XMC2-Det_teacher_detector',
    ]
    model_types = ['.onnx']
    for model_name in model_names:
        for model_type in model_types:
            model_file_path = os.path.join(model_dir, model_name + model_type)
            print(model_file_path)
            encrypt_model_file_path = os.path.join(model_dir, model_name + model_type + '.encrypt')
            decrypt_model_file_path = os.path.join(model_dir, model_name + model_type + '.decrypt')
            encrypt(key, model_file_path, encrypt_model_file_path)
            decrypt(key, encrypt_model_file_path, decrypt_model_file_path)
