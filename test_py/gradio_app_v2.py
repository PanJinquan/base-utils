import gradio as gr
import os

root = "/home/PKing/Downloads/images"
def load_page_images(img_dir, page=1, page_size=8):
    all_images = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.endswith(('.png', '.jpg'))]
    total_pages = (len(all_images) + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    return all_images[start_idx: start_idx + page_size], total_pages


with gr.Blocks() as demo:
    img_dir = gr.Textbox(value="path", label="图片目录")
    page = gr.Number(value=1, label="当前页码", precision=0)
    gallery = gr.Gallery(label="图片展示", columns=4, height="auto")
    total_pages = gr.Textbox(label="总页数")

    img_dir.change(
        fn=lambda d: load_page_images(d, page=1),
        inputs=img_dir,
        outputs=[gallery, total_pages]
    )
    page.change(
        fn=lambda d, p: load_page_images(d, page=p)[0],
        inputs=[img_dir, page],
        outputs=gallery
    )
demo.launch(allowed_paths=[root])
