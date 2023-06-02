import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"
try:
    os.system("pip install --upgrade  torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html")
except Exception as e:
    print(e)

from pydoc import describe
from huggingface_hub import hf_hub_download
import gradio as gr
import os
from datetime import datetime
from PIL import Image
import torch
import torchvision
from diffusers import StableDiffusionImg2ImgPipeline
import skimage
import paddlehub
import numpy as np
from lib.options import BaseOptions
from apps.crop_img import process_img
from apps.eval import Evaluator
from types import SimpleNamespace
import trimesh
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, revision="fp16", safety_checker=None) if torch.cuda.is_available() else StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker=None)
pipe = pipe.to(device)

print(
    "torch: ", torch.__version__,
    "\ntorchvision: ", torchvision.__version__,
    "\nskimage:", skimage.__version__
)

print("EnV", os.environ)

net_C = hf_hub_download("radames/PIFu-upright-standing", filename="net_C")
net_G = hf_hub_download("radames/PIFu-upright-standing", filename="net_G")


opt = BaseOptions()
opts = opt.parse_to_dict()
opts['batch_size'] = 1
opts['mlp_dim'] = [257, 1024, 512, 256, 128, 1]
opts['mlp_dim_color'] = [513, 1024, 512, 256, 128, 3]
opts['num_stack'] = 4
opts['num_hourglass'] = 2
opts['resolution'] = 128
opts['hg_down'] = 'ave_pool'
opts['norm'] = 'group'
opts['norm_color'] = 'group'
opts['load_netG_checkpoint_path'] = net_G
opts['load_netC_checkpoint_path'] = net_C
opts['results_path'] = "./results"
opts['name'] = "spaces_demo"
opts = SimpleNamespace(**opts)
print("Params", opts)
evaluator = Evaluator(opts)
bg_remover_model = paddlehub.Module(name="U2Net")

def resize(value,img):
    img = Image.open(img)
    img = img.resize((value,value))
    return img

def infer(source_img, prompt, negative_prompt, guide, steps, seed, Strength):
    generator = torch.Generator(device).manual_seed(seed)     
    source_image = resize(768, source_img)
    source_image.save('source.png')
    image = pipe(prompt, negative_prompt=negative_prompt, image=source_image, strength=Strength, guidance_scale=guide, num_inference_steps=steps).images[0]
    return image

def process(img_path):
    base = os.path.basename(img_path)
    img_name = os.path.splitext(base)[0]
    print("\n\n\nStarting Process", datetime.now())
    print("image name", img_name)
    img_raw = Image.open(img_path).convert('RGB')

    img = img_raw.resize(
        (512, int(512 * img_raw.size[1] / img_raw.size[0])),
        Image.Resampling.LANCZOS)

    try:
        # remove background
        print("Removing Background")
        masks = bg_remover_model.Segmentation(
            images=[np.array(img)],
            paths=None,
            batch_size=1,
            input_size=320,
            output_dir='./PIFu/inputs',
            visualization=False)
        mask = masks[0]["mask"]
        front = masks[0]["front"]
    except Exception as e:
        print(e)

    print("Aliging mask with input training image")
    print("Not aligned", front.shape, mask.shape)
    img_new, msk_new = process_img(front, mask)
    print("Aligned", img_new.shape, msk_new.shape)

    try:
        time = datetime.now()
        data = evaluator.load_image_from_memory(img_new, msk_new, img_name)
        print("Evaluating via PIFu", time)
        evaluator.eval(data, True)
        print("Success Evaluating via PIFu", datetime.now() - time)
        result_path = f'./{opts.results_path}/{opts.name}/result_{img_name}'
    except Exception as e:
        print("Error evaluating via PIFu", e)

    try:
        mesh = trimesh.load(result_path + '.obj')
        # flip mesh
        mesh.apply_transform([[-1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]])
        mesh.export(file_obj=result_path + '.glb')
        result_gltf = result_path + '.glb'
        return [result_gltf, result_gltf]

    except Exception as e:
        print("error generating MESH", e)


examples = sorted(glob.glob('examples/*.png'))

iface1 = gr.Interface(fn=infer, inputs=[gr.Image(source="upload", type="filepath", label="Raw Image. Must Be .png"), gr.Textbox(label = 'Prompt Input Text. 77 Token (Keyword or Symbol) Maximum'), gr.Textbox(label='What you Do Not want the AI to generate.'),
    gr.Slider(2, 15, value = 7, label = 'Guidance Scale'),
    gr.Slider(1, 25, value = 10, step = 1, label = 'Number of Iterations'),
    gr.Slider(label = "Seed", minimum = 0, maximum = 987654321987654321, step = 1, randomize = True), 
    gr.Slider(label='Strength', minimum = 0, maximum = 1, step = .05, value = .5)], 
    outputs='image')

iface2 = gr.Interface(
    fn=process,
    inputs=gr.Image(type="filepath", label="Input Image"),
    outputs=[
        gr.Model3D(
            clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model"),
        gr.File(label="Download 3D Model")
    ],
    examples=examples,
    allow_flagging="never",
    cache_examples=True
)

demo = gr.TabbedInterface([iface1, iface2], ["Image-Edit-with-Text", "Image-to-3D-Model"])

if __name__ == "__main__":
    demo.launch()

# if __name__ == "__main__":
#     iface1.launch(debug=True, enable_queue=False)
#     iface2.launch(debug=True, enable_queue=False)