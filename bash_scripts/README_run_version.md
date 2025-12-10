# run_version.sh - Version-based Workflow Script

이 스크립트는 version 이름과 특징을 입력받아 PA 학습, DPD 학습, 검증을 자동으로 실행하고 로그를 저장합니다.

## 사용법

```bash
./bash_scripts/run_version.sh <version_name> <version_description>
```

## 예시

```bash
# 기본 사용
./bash_scripts/run_version.sh rev6 "target_gain_all_data"

# 환경 변수로 설정 변경
DATASET_NAME=DPA_200MHz \
PA_BACKBONE=gru \
DPD_BACKBONE=gru \
./bash_scripts/run_version.sh rev7 "linear_region_gain"
```

## 실행 단계

1. **PA Modeling**: PA 모델 학습
2. **DPD Learning**: DPD 모델 학습 (PA 모델 자동 로드)
3. **Validation Experiment**: DPD 모델 검증 (DPD 모델 자동 로드)

## 로그 저장 위치

모든 로그는 `terminal_log/{version_name}_{version_description}/` 디렉토리에 저장됩니다:
- `pa.log`: PA 학습 로그
- `dpd.log`: DPD 학습 로그
- `run.log`: DPD 검증 로그

## 환경 변수

다음 환경 변수로 설정을 변경할 수 있습니다:

- `DATASET_NAME`: 데이터셋 이름 (기본값: DPA_200MHz)
- `ACCELERATOR`: 가속기 타입 (기본값: cuda)
- `DEVICES`: 디바이스 번호 (기본값: 0)
- `SEED`: 랜덤 시드 (기본값: 0)
- `PA_BACKBONE`: PA 백본 타입 (기본값: gru)
- `PA_HIDDEN_SIZE`: PA hidden size (기본값: 23)
- `PA_NUM_LAYERS`: PA 레이어 수 (기본값: 1)
- `DPD_BACKBONE`: DPD 백본 타입 (기본값: gru)
- `DPD_HIDDEN_SIZE`: DPD hidden size (기본값: 15)
- `DPD_NUM_LAYERS`: DPD 레이어 수 (기본값: 1)
- `FRAME_LENGTH`: 프레임 길이 (기본값: 200)

## 모델 저장 위치

모든 모델은 `--version` 인자에 따라 다음 경로에 저장됩니다:

- PA 모델: `./save/{dataset_name}/train_pa/{version}/PA_*.pt`
- DPD 모델: `./save/{dataset_name}/train_dpd/{PA_model_id}/{version}/DPD_*.pt`
- DPD 출력: `./dpd_out/{version}/DPD_*.csv`

## 주의사항

- 각 단계가 성공적으로 완료되어야 다음 단계로 진행됩니다
- 모델 파일을 자동으로 찾아서 다음 단계에 사용합니다
- 실패 시 에러 메시지와 함께 종료됩니다

