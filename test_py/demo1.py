from pybaseutils import file_utils,base64_utils,image_utils



if __name__ == "__main__":
    file ="/home/PKing/Downloads/TextCaps-rerank.json"
    data = file_utils.load_json(file)
    data = base64_utils.deserialization(data)
    image_utils.show_image("image",data["image"])
    print(data)
