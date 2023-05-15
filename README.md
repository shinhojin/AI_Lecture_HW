# AI_Lecture_HW

- 이름: 신호진
- 학번: 72210295
- 학과: 컴퓨터학과

## Model Explanation


## How to Execute the code

₩₩₩python

    !pip install timm==0.6.11
    !git clone https://github.com/sail-sg/metaformer.git
    !wget https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/cat.jpg
    
    cd /content/metaformer
    
    import metaformer_baselines # MetaFormer 모델 가져오기
    from PIL import Image
    from timm.data import create_transform
    model = metaformer_baselines.caformer_s18(pretrained=True) # 다른 모델을 바꿔서 실험 가능
    model.eval()
    transform = create_transform(input_size=224, crop_pct=model.default_cfg['crop_pct'])
    image = Image.open('../cat.jpg')
    input_image = transform(image).unsqueeze(0)

    pred = model(input_image) // 모델 학습 함수
    print(f'Prediction: {imagenet_classes[int(pred.argmax())]}.') # 추론 결과 출력
    image # 이미지 출력
₩₩₩

## Model Pesudo Code


## Code details

- Line: 521 - 637 at metaformer_baselines.py

## Comments

- 해당 Repository는 2022년에 발표된 Weihao Yu et al., "MetaFormer Baselines for Vision" 논문에서 가져온 자료입니다.
- Git url: https://github.com/sail-sg/metaformer.git



| Model | Resolution | Params | MACs | Top1 Acc | Download |

