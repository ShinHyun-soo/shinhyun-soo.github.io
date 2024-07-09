---   
layout: post   
title:  Stable-Diffusion-3   
date:   2024-07-09 19:20:23 +0900   
category: Diffusion-model  
---   



# DDPM    
> $x_t = (1-t)x_0 + tϵ$    
> $t = [0, 1]$    
> $t$ 가 0 이면 ($x_0$), $x_0$ (원본 이미지)     
> $t$ 가 1 이면 ($x_1$),  $ϵ_t$  (노이즈)       
> 여기서 $ϵ ∼ N(0,1)$ 가우시안 분포임. (노이즈)     

## Backward Process
>  &x_(0.7)& 에서 $M_(theta)$ 를 더하면 $ϵ_t$(순수 노이즈)가 나오게 $M_(theta)$ 학습.     
> 여기서 $x_t$ - $M_(theta)$ = $x_0$ 원본 이미지가 나옴.      
> Chain Process 인 이유는, 예측이 어렵기 때문.    
> $x_t - alpha * ϵ_t$ , $alpha [0, 1]$    






