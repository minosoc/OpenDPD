# Transformer 모델 훈련 개선사항

## 문제점 분석

Transformer 모델 훈련 시 NMSE가 2~3으로 높게 나오고 더 이상 감소하지 않는 문제가 발생했습니다.
GRU 모델은 -30 정도로 잘 훈련되는데 비해 Transformer는 성능이 매우 낮았습니다.

## 주요 원인

1. **Learning Rate Warmup 부재**: Transformer는 초기 학습률을 점진적으로 증가시켜야 합니다
2. **Gradient Clipping 값이 너무 큼**: 기본값 200은 Transformer에 비해 매우 큽니다 (일반적으로 1.0)
3. **Weight Decay 미적용**: 정규화를 위한 weight decay가 명시되지 않았습니다
4. **Learning Rate가 높을 수 있음**: Transformer는 보통 더 낮은 learning rate가 필요합니다

## 적용된 개선사항

### 1. Learning Rate Warmup 추가 ✅
- `--warmup_steps` 파라미터 추가
- Transformer 모델에 대해 초기 학습률을 점진적으로 증가시킵니다
- 예: `--warmup_steps 1000` (첫 1000 스텝 동안 warmup)

### 2. Weight Decay 추가 ✅
- `--weight_decay` 파라미터 추가 (기본값: 0.01)
- 모든 optimizer에 weight_decay 적용
- Transformer 모델의 정규화를 개선합니다

### 3. Gradient Clipping 자동 조정 ✅
- Transformer 모델 감지 시 자동으로 gradient clipping을 1.0으로 조정
- 기본값 200은 RNN 모델용이며, Transformer에는 너무 큽니다
- Gradient norm 모니터링 추가 (디버깅용)

### 4. Gradient Monitoring 추가 ✅
- 훈련 중 gradient norm을 로깅
- `grad_norm`, `grad_norm_max` 메트릭 추가

## 사용 방법

### 기본 사용 (권장 설정)
```bash
python main.py \
    --step train_pa \
    --PA_backbone transformer_encoder \
    --lr 1e-4 \
    --grad_clip_val 1.0 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --lr_schedule 1
```

### Transformer에 적합한 하이퍼파라미터
- **Learning Rate**: `1e-4` ~ `3e-4` (기본값 5e-4보다 낮게)
- **Gradient Clipping**: `1.0` (기본값 200 대신)
- **Warmup Steps**: `500` ~ `2000` (데이터셋 크기에 따라)
- **Weight Decay**: `0.01` (기본값)
- **Batch Size**: 가능하면 크게 (256 이상)

## 예상 효과

1. **안정적인 학습**: Warmup으로 초기 학습 안정화
2. **더 나은 수렴**: 적절한 gradient clipping으로 gradient explosion 방지
3. **정규화 개선**: Weight decay로 overfitting 감소
4. **디버깅 용이**: Gradient monitoring으로 문제 진단 가능

## 추가 권장사항

1. **Learning Rate 조정**: 
   - 작은 모델 (d_model < 512): `3e-4`
   - 중간 모델 (512 <= d_model < 1024): `1e-4`
   - 큰 모델 (d_model >= 1024): `5e-5` ~ `1e-4`

2. **Warmup Steps 계산**:
   - 일반적으로 전체 훈련 스텝의 5-10%
   - 예: 50 epochs, batch_size=256, 10000 samples → 약 2000 steps → warmup_steps=100-200

3. **모니터링**:
   - `grad_norm`이 1.0 근처에 유지되는지 확인
   - `grad_norm`이 계속 증가하면 learning rate를 낮추거나 gradient clipping을 더 작게

## 참고

- 원본 Transformer 논문: "Attention is All You Need" (Vaswani et al., 2017)
- Pre-norm Transformer: 더 안정적인 학습을 위해 이미 적용됨
- AdamW optimizer: Weight decay가 포함된 Adam (기본 사용)
