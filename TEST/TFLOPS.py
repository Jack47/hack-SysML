import torch
#from eva_clip import create_model_and_transforms, get_tokenizer
import open_clip
from PIL import Image

# model_name = "EVA01-CLIP-g-14" 
model_name = "OpenaiCLIP-L-14-336"#"EVA02-CLIP-bigE-14" 
pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

image_path = "/mnt/workspace/EVA/EVA-CLIP/assets/CLIP.png"
caption = ["a diagram", "a dog", "a cat"]

device =  "cpu"
#model, preprocess = clip.load("ViT-L/14@336px", device=device)
model, _, preprocess = open_clip.create_model_and_transforms('EVA02-E-14') 
#model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k') 
#torch.save(model.state_dict(), '/home/ldh/my/EVA/EVA-CLIP/rei/zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzmodel_weights.pth')
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model) 
print(f'The model has {num_params} parameters.')
tokenizer = open_clip.get_tokenizer('EVA02-E-14')
model = model.to(device)

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
#text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)
text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)
from thop import profile
#flops, params = profile(model, inputs=(input, ))
with torch.no_grad(), torch.cuda.amp.autocast():
    input = torch.randn(1, 3, 224, 224)
   # flops = stat(model.visual,(image,))#thop.profilehooks.profile(model.visual, inputs=(image,))[0]
    flops, params = profile(model.visual, inputs=(input, ))
    image_features = model.encode_image(image)
    print(f'FLOPs: {flops/1e12}')
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[0.8275, 0.1372, 0.0352]]
print("tflops:" ,flops / 1e12)