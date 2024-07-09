---
layout: post
title: Mobile-Diffusion 
date: 2024-02-16 19:10:00 +0900 
category : Diffusion-Model
---

MobileDiffusion: Subsecond Text-to-Image Generation on Mobile Devices [Zhao, Yang, et al., arXiv 2023]    

요약 : MobileDiffusion은 아키텍쳐와 샘플링 기술 최적화를 통해 모바일 기기에서 512 x 512 이미지를 1초 미만의 시간으로 생성할 수 있게 함.    

# 1. 트랜스포머 최적화
&nbsp;
## 1.1. More transformers in the middle of Unet.  
   * 고해상도 트랜스포머를 저해상도(_fewer layers_) 트랜스포머로 대치.<br>
   * 저해상도 트랜스포머의 채널 크기(_Using Separable Conv_)를 줄임.<br>
     * 이렇게 해도 정량적 측정 기준이나 시각적 품질에 영향 없었음.<br>
       * 하지만, 트랜스포머의 폭(_input dim_)을 줄이는 것은 시각적 품질에 악영향을 끼쳤음.<br>
   * 따라서, 더 낮은 해상도의 더 많은 트랜스포머 블럭들을 채널 수를 줄여 통합하였음. <br> - (품질 저하 없이 런타임 효율 26% 향상)<br>    
   &nbsp;
## 1.2. Decouple SA from CA.
   * 셀프 어텐션은 장거리 종속성(*long-range dependencies*)을 캡쳐하는데 중요한 역할을 하지만, 특히 고해상도에서 많은 계산량을 요구함.<br>    
   * 기존 연구에서는 전체 트랜스포머 블럭들을 낮은 해상도로 재배치 함으로써 고해상도에서의 self-attention 과 cross-attention 을 제거함.<br>    
     * 위와 같이하면 성능이 눈에 띄게 저하됨. 
       * (FID`Frechet Inception Distance` : 12.50 -> 17.87, CLIP score : 0.325 -> 0.302) <br>
   * 고 해상도에서 셀프 어텐션 레이어들만 제거하면 성능 저하 없었음.<br>
   * text guidance 가 이미지의 글로벌 레이아웃과 로컬 텍스쳐에 중요하기 때문에 크로스 어텐션(이하 CA)이 다양한 해상도에서 중요하다고 추측함.<br>
   * 특히, 텍스트 임베딩의 시퀀스가 더 짧기 때문에, 고해상도에서 CA의 계산 비용이 SA 보다 훨씬 낮음.<br>
     * 따라서 SA 만 제거하면 효율성이 크게 향상됨.<br>
       * 최고 해상도(64x64)에서 트랜스포머 블럭(SA,CA)을 완전히 제거.<br>
       * 32x32와 외부 16x16에서 SA 제거.<br>
       * 내부 16x16과 가장 안쪽 병목 스택은 완전환 트랜스포머 블럭 유지. <br> - (품질 저하 없이 런타임 효율 15% 향상)<br>
       &nbsp;

## 1.3.  Share key-value projections.
* K = x · Wk , V = x · Wv<br>
  *  Wk 와 Wv 를 공유해도 모델 성능에는 부정적인 영향을 미치지 않았음. <br> - (parameter count 5% 감소.)<br>

## 1.4. Replace gelu with swish.
* GLU (Gated Linear Unit)은 구현할 때, gelu 활성화 함수를 사용하는데, 이는 3차 연산이므로 모바일 기기에서 float16 이나 int8 추론 시 수치적 불안정성을 야기함. <br>
* 따라서 유사한 모양이지만 계산 효율적인 swish 함수를 사용하였음.<br>
  * 측정 항목이나 지각 품질에 저하 없었음.<br>
  - 이 최적화는 훈련 후, 모바일 기기 배포를 위함임.<br> 

## 1.5. Finetune softmax into relu.
* softmax 함수는 attention 계산에 이용 되는데, 비효율적임. <br>
  * 대신에 relu 를 사용하면, 더 빠르고 실현 가능함.<br>
    * 흥미롭게도, pre-trained softmax-attention 을 relu-attention 으로 파인 튜닝이 가능했음.<br>
      * 이 방식이 relu-attention 보다 FID 가 개선됨. (12.50 > 12.31)<br>
* 이 최적화 역시 다양한 모바일 플랫폼 배포를 위함임.
      
## 1.6. Trim feed-forward layers.
* 초기 UNet transformer block 은 feed-forward 레이어에서 확장 비율이 8로, 1280 에서 10240 이라는 엄청난 차원을 가지고 있음. <br>
* 실험 결과, 6 정도로 줄이면 FID 점수는 12.31 에서 12.58 로 상승하지만 파라미터 수는 10% 감소 하였음. <br>

# 2. Convolution Blocks 최적화

## 2.1. Separable convolution. 
* Unet 의 가장자리 convolutional blok 을 제외한 모든 블럭들을 seprable Conv 로 대체함.<br>
* 실험을 통해, 7x7 이나 9x9 처럼 큰 커널 사이즈는 성능 향상에 영향을 끼치지 못했으므로 3x3 크기 커널을 사용하였음.<br>
    * 모델 파라미터 갯수 대략 10% 줄임.<br>

## 2.2. Prune redundant residual blocks.
* 잔여 블럭 갯수를 22개에서 12개로 줄임.<br>
  * SD에서 가장 안쪽을 제외하고 2개가 아닌 1개로 줄임.<br>
- 계산 효율성 19% 향상, 파라미터 15% 감소. <br>

# 3. Sample Efficiency

* off-the-self progressive distillation techniques 을 사용함.<br>
  * distillation 은 최소 **8 단계**는 해야 이미지 품질 저하가 없었음.
* _Ufogen: You forward once large scale text-to-image gener- ation via diffusion gans. Yanwu Xu et al.,_ 에서는 <br> reconstruction term 과 adversarial term 의 하이브리드 목적 함수를 사용하여 pre-trained 디퓨전 모델을 파인튜닝할 것을 제안하였음. <br>
  * 이를 구현하였더니, pre-trained 디퓨전 모델이 엄청난 샘플링 효율성을 갖춘 **single-step** 생성 모델로 탈바꿈 됨. <br>

# 4. Experiments.
![image](https://github.com/apple/ml-stable-diffusion/assets/69250097/56ef39c5-a538-48ef-a3d3-9cd78794b667)
표 1. 다른 레이턴트 디퓨전 모델과의 비교.
![image](https://github.com/apple/ml-stable-diffusion/assets/69250097/4ab61277-c391-4ca6-b125-5a903c9b4e72)
표 2. 파인튜닝 전 후 디코더 변형.

## 4.1. Training

* SD-1.5 와 비교하였을때, MobileDiffusion 은 46.4 의 압축률, MobileDiffusion-lite 는 29.9% 의 압축률을 보여줌.
