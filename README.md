# AI_Lecture_HW

- 이름: 신호진
- 학번: 72210295
- 학과: 컴퓨터학과

## Model Explanation


## How to Execute the code

해당 코드는 **Google Colab** 환경에서 진행하였습니다.

**dependency**는 Google Colab 환경에서 아래 코드를 진행하면 문제없이 작동합니다.

```python

    !pip install timm==0.6.11
    !git clone https://github.com/sail-sg/metaformer.git
    !wget https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/cat.jpg
    
    cd /content/metaformer
    
    import metaformer_baselines # MetaFormer 모델 가져오기
    from PIL import Image
    from timm.data import create_transform
    model = metaformer_baselines.caformer_s18(pretrained=True) # 다른 모델을 바꿔서 실험 가능
    model.eval()
    transform = create_transform(input_size=224, crop_pct=model.default_cfg['crop_pct']) # transformer 생성
    image = Image.open('../cat.jpg')
    input_image = transform(image).unsqueeze(0)

    pred = model(input_image) # 예측 함수
    print(f'Prediction: {imagenet_classes[int(pred.argmax())]}.') # 추론 결과 출력
    image # 이미지 출력
```

## Model Pesudo Code

- 모델 수도 코드 (MetaFormer)

***
**MetaFormer**: Model Pesudo Code
***
___Require: Arguments___

* in_chans (int): 입력 이미지 채널 수

* num_classes(int): 분류 헤드의 클래스 수

* depths(list or tuple): 각 단계의 블록 수

* dims(int): 각 단계의 기능 차원

* downsample_layers: (list or tuple): 각 stage 전에 downsampling layers

* token_mixers (list, tuple or token_fcn): 각 단계에 대한 토큰 믹서

* mlps(list, tuple or mlp_fcn): 각 단계에 대한 Mlp

* norm_layers (list, tuple or norm_fcn): 각 단계에 대한 norm layers

* drop_path_rate (float): 확률적 깊이 비율

* head_dropout(float): MLP 분류자에 대한 드롭아웃

* layer_scale_init_values (list, tuple, float or None): 레이어 스케일의 초기 값

* res_scale_init_values(list, tuple, float or None): 레이어 스케일의 초기 값

* output_norm: 분류기 헤드 이전의 norm

* head_fn: 분류 헤드
  
**Procedure** _init_(Args) # Args는 위에 있는 파라미터입니다.
    
    # classes와 stages 값을 초기화
    
    self.num_classes '&larr;' num_classes
    
    self.num_stages '&larr;' len(depths)
    
    # downsample layers를 초기화
    self.downsample_layers '&larr;' nn.ModuleList([downsample_layers(in_chans, dims[i]) for i in range(self.num_stages)])
    
    # token mixers를 초기화
    self.token_mixers '&larr;' nn.ModuleList([token_mixers] * self.num_stages)
    
    # MLPs를 초기화
    self.mlps '&larr;' nn.ModuleList([mlps] * self.num_stages)
    
    # norm layers를 초기화
    self.norm_layers '&larr;' nn.ModuleList([norm_layers] * self.num_stages)
    
    # drop path rates를 초기화
    self.drop_path_rates '&larr;' [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
    
    # layer scale 초기 값을 설정
    self.layer_scale_init_values '&larr;' [layer_scale_init_values] * self.num_stages
    
    # res scale 초기 값을 설정
    self.res_scale_init_values '&larr;' [res_scale_init_values] * self.num_stages
    
    cur '&larr;' 0
    
    # stages를 초기화
    self.stages '&larr;' nn.ModuleList()
    for i in range(self.num_stages):
        stage '&larr;' nn.Sequential(
            *[MetaFormerBlock(dim '&larr;' dims[i],
            token_mixer '&larr;' token_mixers[i],
            mlp '&larr;' mlps[i],
            norm_layer '&larr;' norm_layers[i],
            drop_path '&larr;' dp_rates[i + j],
            layer_scale_init_value '&larr;' layer_scale_init_values[i],
            res_scale_init_value '&larr;' res_scale_init_values[i],
            ) for j in range(depths[i])]
        )
        self.stages.append(stage)
    
    # norm layer를 초기화
    self.norm '&larr;' output_norm(dims[-1])
    
    # head를 초기화
    if head_dropout > 0.0:
        self.head '&larr;' head_fn(dims[-1], num_classes, head_dropout '&larr' head_dropout)
    else:
        self.head '&larr;' head_fn(dims[-1], num_classes)
      
    # weights를 초기화
    init_weights()
    
  **procedure** _init_weights_
    for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std '&larr;' .02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
  **procedure** _forward_features_
    for i in range(self.num_stages):
        x '&larr;' self.downsample_layers[i](x)
        x '&larr;' self.stages[i](x)
    return self.norm(x.mean[1, 2]))
    
  **procedure** _forward_
    x '&larr;' self.forward_features(x)
    x '&larr;' self.head(x)
    return x
***

## Code details

- You can find model code details at the below location
- Line: **521 - 637** at **metaformer_baselines.py**

## Comments

- 해당 Repository는 2022년에 발표된 Weihao Yu et al., "MetaFormer Baselines for Vision" 논문에서 가져온 자료입니다.
- Git url: https://github.com/sail-sg/metaformer.git
