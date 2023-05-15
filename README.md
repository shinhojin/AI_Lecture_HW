# AI_Lecture_HW

- 이름: 신호진
- 학번: 72210295
- 학과: 컴퓨터학과

## Model Explanation

 **MetaFormer**는 기존 Transformer를 확장하고 성능 향상시키기 위해 메타학습에 기반한 딥러닝 모델입니다.
 
 MetaFormer는 Transformer의 추상화된 구조입니다.
 
 $$ X' = X + TokenMixer(Norm_1(X)) $$
 
 $$ X'' = X' + \sigma(Norm_2(X')W_1)W_2 $$
 
 $$ X = token\,sequence N = length, C = channel\,dimension, Norm() = normalization, \sigma = activation\,function, W = learnable\,parameter\,in\,channel\,MLP $$
 
 여기서 말하는 **메타 학습 (MetaLearning)** 이란 적은 양의 데이터와 주어진 환경만으로 스스로 학습하고, 학습한 정보와 알고리즘을 새로운 문제에 적용하여 해결하는 학습 방식을 의미합니다.
 
 MetaFormer를 적용한 4가지 **(IdentityFormer, RandFormer, ConvFormer, CAFormer)** 를 보여줍니다.
 
 IdentityFormer를 통해 MetaFormer 성능의 하한선을 보여주고, RandFormer를 통해 보편적으로 활용할 수 있음을 보여줍니다.
 
 본 MetaFormer에서 활성화 함수로 **StarReLU**를 제안합니다.
 
 기존에 있는 활성화 함수인 ReLU는 1 FLOPs를 보이며, 이를 개선한 GELU는 성능이 좋아졌지만 14 FLOPs로 비용이 높아졌습니다.
 
 이를 다시 개선한 Squared ReLU는 2 FLOPs를 보여줍니다.
 
 MetaFormer에서 사용한 StarReLU는 4 FLOPs를 가지면서 기존 활성화 함수들보다 높은 성능을 보여줍니다.
 
 StarReLU는 입력에 대한 분포 편향을 제거하는 구조를 가지며, 입력의 제곱을 취한 다음 입력을 제곱의 제곱근으로 나누어 수행됩니다.
 
 이를 통해 기존 ReLU보다 분포 편향에 덜 민감한 활성화 함수가 생성됩니다.
 
 $$ StarReLU(x) = \frac{(ReLU(x))^2 - E((ReLU(x))^2)}{\sqrt{Var((ReLU(x))^2}} = \frac{(ReLU(x))^2 - 0.5}{\sqrt{1.25}} $$

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
    # 다른 모델 설정 예시
    # model = metaformer_baselines.indetityformer_s24(pretrained=True)
    # model = metaformer_baselines.randformer_s36(pretrained=True)
    # model = metaformer_baselines.poolformerv2_m48(pretrained=True)
    # model = metaformer_baselines.convformer_s18_384_in21ft1k(pretrained=True)
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
    self.num_classes <- num_classes
    self.num_stages <- len(depths)
    
    # downsample layers를 초기화
    self.downsample_layers <- nn.ModuleList([downsample_layers(in_chans, dims[i]) for i in range(self.num_stages)])
    
    # token mixers를 초기화
    self.token_mixers <- nn.ModuleList([token_mixers] * self.num_stages)
    
    # MLPs를 초기화
    self.mlps <- nn.ModuleList([mlps] * self.num_stages)
    
    # norm layers를 초기화
    self.norm_layers <- nn.ModuleList([norm_layers] * self.num_stages)
    
    # drop path rates를 초기화
    self.drop_path_rates <- [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
    
    # layer scale 초기 값을 설정
    self.layer_scale_init_values <- [layer_scale_init_values] * self.num_stages
    
    # res scale 초기 값을 설정
    self.res_scale_init_values <- [res_scale_init_values] * self.num_stages
    
    cur <- 0
    
    # stages를 초기화
    self.stages <- nn.ModuleList()
    for i in range(self.num_stages):
        stage <- nn.Sequential(
            *[MetaFormerBlock(dim <- dims[i],
            token_mixer <- token_mixers[i],
            mlp <- mlps[i],
            norm_layer <- norm_layers[i],
            drop_path <- dp_rates[i + j],
            layer_scale_init_value <- layer_scale_init_values[i],
            res_scale_init_value <- res_scale_init_values[i],
            ) for j in range(depths[i])]
        )
        self.stages.append(stage)
    end for
    
    # norm layer를 초기화
    self.norm <- output_norm(dims[-1])
    
    # head를 초기화
    if head_dropout > 0.0:
        self.head <- head_fn(dims[-1], num_classes, head_dropout '&larr' head_dropout)
    else:
        self.head <- head_fn(dims[-1], num_classes)
      
    # weights를 초기화
    init_weights()
    
  **Procedure** _init_weights_
  
    for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std '&larr;' .02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
  **Procedure** _forward_features_
  
    for i in range(self.num_stages):
        x <- self.downsample_layers[i](x)
        x <- self.stages[i](x)
    return self.norm(x.mean[1, 2]))
    
  **Procedure** _forward_
  
    x <- self.forward_features(x)
    x <- self.head(x)
    return x
***

## Code details

- You can find model code details at the below location
- Line: **521 - 637** at **metaformer_baselines.py**

## Comments

- 해당 Repository는 2022년에 발표된 Weihao Yu et al., "MetaFormer Baselines for Vision" 논문에서 가져온 자료입니다.
- Git url: https://github.com/sail-sg/metaformer.git
