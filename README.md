# 가위바위보 대결 게임 (LiteRT + 손 검출)

웹캠 앞에서 두 사람이 손을 내밀면 자동으로 카운트다운을 하고, 각자의 손 모양(가위/바위/보)을 인식해 승패와 점수를 화면에 보여주는 2인용 실시간 가위바위보 게임입니다.

- **손 위치 검출**: cvzone `HandDetector` (MediaPipe 기반)
- **손 모양 분류**: LiteRT(TFLite)로 실행하는 YOLO 계열 객체 탐지 모델 (`best.tflite`)
- **화면 출력**: OpenCV

## 주요 기능

- 손 2개가 감지되면 자동으로 게임 시작 (별도 키 입력 없음)
- 3초 카운트다운 후 0.5초 동안 여러 프레임의 인식 결과를 모아 **다수결**로 손 모양 확정
- 화면 왼쪽 손은 P1, 오른쪽 손은 P2로 자동 구분 (화면은 거울처럼 좌우 반전)
- 승자 색상으로 화면 전체를 덮는 결과 연출, 누적 점수 표시
- 하단에 감지된 손 개수와 FPS 표시

## 요구 사항

- Python 3.9 이상 권장
- 웹캠 (장치 번호 0)
- 학습된 모델 파일 `best.tflite`

### 패키지 설치

```bash
pip install ai-edge-litert numpy opencv-python cvzone mediapipe
```

## 모델 조건

코드는 아래 형식의 모델을 기준으로 작성되어 있으며, 조건이 맞지 않으면 시작 시 오류를 냅니다.

| 항목 | 값 |
| --- | --- |
| 입력 자료형 | `float32` (0~1 정규화) |
| 입력 형태 | NCHW `(1, 3, H, W)`, 예: `(1, 3, 320, 320)` |
| 출력 형태 | `(1, 7, N)`, 예: `(1, 7, 2100)` → `x, y, w, h, class0, class1, class2` |
| 박스 좌표 | 0~1로 정규화된 중심 좌표(cx, cy, w, h)라고 가정 |

클래스 번호는 다음과 같습니다.

| class_id | 손 모양 |
| --- | --- |
| 0 | scissors (가위) |
| 1 | rock (바위) |
| 2 | paper (보) |

학습한 모델의 클래스 순서가 다르면 `ansToText`를 수정하세요. 박스 좌표가 픽셀 단위로 나오는 모델이라면 `processImage`에서 `IMG_W`, `IMG_H`를 곱하는 부분을 제거해야 합니다.

## 실행 방법

1. `best.tflite`를 스크립트와 같은 폴더에 둡니다. (다른 경로라면 `modelPath` 수정)
2. 스크립트를 실행합니다. 파일 이름이 `main.py`라면:

```bash
python main.py
```

3. `q` 키를 누르면 종료됩니다.

## 게임 진행 방식

| 상태 | 화면 표시 | 동작 | 다음 상태 |
| --- | --- | --- | --- |
| WAITING | READY | 손 2개가 보이기를 기다림 | 손 2개 감지 → COUNTDOWN |
| COUNTDOWN | GET READY! 3·2·1 | 3초 카운트다운 | 완료 → COLLECTING / 손이 0.2초 넘게 사라짐 → SHOW_ERROR |
| COLLECTING | SHOW YOUR GESTURE! | 0.5초 동안 프레임마다 손 모양을 인식해 투표 수집 | 다수결 성공 → SHOW_RESULT / 인식 실패 → SHOW_ERROR |
| SHOW_RESULT | RESULT, P1/P2 WINS!, DRAW! | 3초간 결과 표시, 승자 점수 +1 | WAITING |
| SHOW_ERROR | ERROR | 1.5초간 오류 메시지 표시 | WAITING |

참고 사항:

- 모델 추론은 COLLECTING 상태에서만 수행하므로, 그 외 상태에서는 손 검출만 돌아갑니다.
- COLLECTING 중 탐지 결과가 정확히 2개인 프레임만 투표에 반영합니다.
- **점수 초기화**: WAITING 상태에서 손 2개가 0.2초 이상 보이지 않으면 두 플레이어의 점수가 0으로 초기화됩니다. 점수를 이어가려면 결과 화면이 끝난 뒤에도 두 손을 계속 화면에 두어야 합니다.

## 설정값

스크립트 상단의 상수로 동작을 조정할 수 있습니다.

| 상수 | 기본값 | 설명 |
| --- | --- | --- |
| `CONF_TH` | 0.4 | 탐지 신뢰도 임계값 |
| `IOU_TH` | 0.45 | NMS IoU 임계값 |
| `COUNTDOWN_TIME` | 3.0 | 카운트다운 시간(초) |
| `COLLECT_TIME` | 0.5 | 손 모양 수집 시간(초) |
| `RESULT_DISPLAY_TIME` | 3.0 | 결과 표시 시간(초) |
| `ERROR_DISPLAY_TIME` | 1.5 | 오류 표시 시간(초) |
| `HAND_LOST_LIMIT` | 0.2 | 손이 사라졌다고 판단하는 시간(초), 점수 초기화 기준 시간으로도 사용 |

카메라는 320x240(MJPG)으로 캡처하고, 창 크기는 480x360입니다. 다른 카메라를 쓰려면 `cv2.VideoCapture(0)`의 번호를 바꾸세요.

## 코드 구성

| 구분 | 함수 | 역할 |
| --- | --- | --- |
| 전처리 | `letterbox`, `preprocess_image` | 비율 유지 리사이즈 + 패딩, RGB 변환, 정규화, NCHW 변환 |
| 추론 | `processImage`, `apply_nms` | LiteRT 추론, 신뢰도 필터링, NMS, 원본 프레임 좌표로 복원 |
| 판정 | `separatePlayers`, `getMajority`, `checkWinner` | 좌우 위치로 플레이어 구분, 다수결, 승패 판정 |
| 화면 | `applyHandOverlay`, `applyResultOverlay` | 플레이어 영역 색상 오버레이, 손 영역은 원본 화면 유지 |
| UI | `drawPlayerHeader`, `drawGameStatus`, `drawResultUI`, `drawErrorUI`, `drawBottomStatus` | 점수, 상태, 결과, 오류, 손 개수/FPS 표시 |

## 문제 해결

| 증상 | 확인할 점 |
| --- | --- |
| `Cannot open camera` | 카메라 연결 상태, 다른 프로그램의 카메라 점유 여부, 장치 번호 |
| `float32 모델이 아닙니다` | 양자화(int8/uint8) 모델이 아닌 float32 모델로 다시 변환 |
| `현재 코드는 NCHW 모델을 기준으로 합니다` | 모델 입력이 NHWC `(1, H, W, 3)`인 경우. 모델을 다시 변환하거나 `preprocess_image`의 `transpose`와 입력 형태 검사를 수정 |
| `Cannot detect gestures`가 자주 뜸 | 조명과 손 위치 확인, `CONF_TH`를 낮추거나 `COLLECT_TIME`을 늘려 보기 |
| `Cannot start the game`이 자주 뜸 | 카운트다운 동안 두 손이 화면 안에 계속 보이도록 유지 |
