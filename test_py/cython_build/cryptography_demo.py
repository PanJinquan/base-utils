# -*-coding: utf-8 -*-
"""
    @Author : PKing
    @E-mail : 
    @Date   : 2023-09-22 10:32:02
    @Brief  : https://www.iotword.com/5307.html
"""
import os
from cryptography.fernet import Fernet
from pybaseutils import file_utils


def create_license():
    """
    :return: 生成密钥
    """
    key = Fernet.generate_key()
    return key


def file_encryption(file, encrypt, license):
    """
    文件加密
    :param file: 原始文件
    :param encrypt: 加密文件
    :param license: 密钥或者license文件
    :return: 
    """
    key = file_utils.read_file(license) if os.path.isfile(license) else license
    bytes = file_utils.read_file(file)
    encryption = Fernet(key).encrypt(bytes)
    file_utils.write_file(encrypt, encryption)
    return encrypt


def file_decryption(encrypt, decrypt, license):
    """
    :param encrypt: 加密文件
    :param decrypt: 解密后的文件
    :param license: 密钥或者license文件
    :return:
    """
    key = file_utils.read_file(license) if os.path.isfile(license) else license
    encryption = file_utils.read_file(encrypt)
    try:
        bytes = Fernet(key).decrypt(encryption)
        file_utils.write_file(decrypt, bytes)
    except Exception as e:
        print("Error,Invalid Token, key is {}".format(key.decode('utf-8')))
        decrypt = ""
    return decrypt


if __name__ == '__main__':
    file1 = '/media/PKing/新加卷1/SDK/base-utils/test_py/cython_build/test1.png'
    file2 = '/media/PKing/新加卷1/SDK/base-utils/test_py/cython_build/test2.png'
    license = '/media/PKing/新加卷1/SDK/base-utils/test_py/cython_build/license'
    encrypt = '/media/PKing/新加卷1/SDK/base-utils/test_py/cython_build/test.png'
    license = create_license()
    # file_encryption(file1, encrypt, license)
    file_decryption(encrypt, file2, license)
