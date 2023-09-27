import os
import rsa
import time


def create_keys(pubkey_path, privkey_path, width):
    (pubkey, privkey) = rsa.newkeys(width)
    pub = pubkey.save_pkcs1()
    with open(pubkey_path, 'wb')as f:
        f.write(pub)

    pri = privkey.save_pkcs1()
    with open(privkey_path, 'wb')as f:
        f.write(pri)


def encrypt(pubkey_path, model_file_path, encrypt_file_path):  # 用公钥加密
    with open(pubkey_path, 'rb') as pubkey_file:
        p = pubkey_file.read()
    pubkey = rsa.PublicKey.load_pkcs1(p)
    with open(model_file_path, 'rb') as model_file:
        model_bytes = model_file.read()
    res = []
    for i in range(0, len(model_bytes), 200):
        res.append(rsa.encrypt(model_bytes[i:i+200], pubkey))
    # encrypt_model = rsa.encrypt(model_bytes, pubkey)
    print(res)
    encrypt_model = b''.join(res)
    with open(encrypt_file_path, 'wb') as encrypt_file:
        encrypt_file.write(encrypt_model)
    print('Encrypt model {} finish'.format(model_file_path))


def decrypt(privkey_path, encrypt_file_path, decrypt_file_path):
    with open(privkey_path, 'rb') as private_file:
        p = private_file.read()
    privkey = rsa.PrivateKey.load_pkcs1(p)
    with open(encrypt_file_path, 'rb') as encrypt_file:
        encrypt_model = encrypt_file.read()
    res = []
    for i in range(0, len(encrypt_model), 256):
        res.append(rsa.decrypt(encrypt_model[i:i+256], privkey))
    # model_bytes = rsa.decrypt(encrypt_model, privkey)
    model_bytes = b''.join(res)
    with open(decrypt_file_path, 'wb') as model_file:
        model_file.write(model_bytes)
    print('Decrypt model {} finish'.format(encrypt_file_path))


if __name__ == '__main__':
    model_dir = '/home/PKing/nasdata/release/edu-engineering/yolov5/runs/equipment23-v2/yolov5m/weights'
    pubkey_path = os.path.join(model_dir, 'rsa.public')
    privkey_path = os.path.join(model_dir, 'rsa.private')
    create_keys(pubkey_path, privkey_path, 2048)
    model_names = [
        'best',
    ]
    model_types = ['.onnx']
    for model_name in model_names:
        for model_type in model_types:
            model_file_path = os.path.join(model_dir, model_name + model_type)
            print(model_file_path)
            start = time.time()
            encrypt_model_file_path = os.path.join(model_dir, model_name + model_type + '.encrypt')
            encrypt(pubkey_path, model_file_path, encrypt_model_file_path)
            print('enctypt time: {}'.format(time.time() - start))
            start = time.time()
            decrypt_model_file_path = os.path.join(model_dir, model_name + model_type + '.decrypt')
            decrypt(privkey_path, encrypt_model_file_path, decrypt_model_file_path)
            print('decrypt time: {}'.format(time.time() - start))
