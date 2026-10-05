import ai_edge_litert.interpreter as tflite
import numpy as np
import time
import cv2

from collections import Counter
from cvzone.HandTrackingModule import HandDetector


# 모델 설정 (예제)

modelPath = "best.tflite"

interpreter = tflite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]
input_dtype = input_details[0]["dtype"]
input_shape = input_details[0]["shape"]

if input_dtype != np.float32:
    raise ValueError(f"float32 모델이 아닙니다: {input_dtype}")

if len(input_shape) != 4 or input_shape[1] != 3:
    raise ValueError(f"현재 코드는 NCHW 모델을 기준으로 합니다: {input_shape}")

IMG_H = int(input_shape[2])
IMG_W = int(input_shape[3])

print("Model input shape:", input_shape)
print("Model size:", IMG_W, IMG_H)


# 기본 설정

ansToText = {
    0: "scissors",
    1: "rock",
    2: "paper"
}

colorList = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255)
]

CONF_TH = 0.4
IOU_TH = 0.45

P1_COLOR = (0, 255, 255)
P2_COLOR = (0, 255, 0)
WINNER_P1_COLOR = P1_COLOR
WINNER_P2_COLOR = P2_COLOR
DRAW_COLOR = (255, 0, 0)

UI_BG_COLOR = (35, 35, 35)
UI_WHITE = (255, 255, 255)
UI_GRAY = (180, 180, 180)
UI_YELLOW = (0, 255, 255)
UI_RED = (50, 50, 255)

UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

UI_PANEL_ALPHA = 0.78
PLAYER_OVERLAY_ALPHA = 0.30
RESULT_OVERLAY_ALPHA = 0.55

COUNTDOWN_TIME = 3.0
COLLECT_TIME = 0.5
RESULT_DISPLAY_TIME = 3.0
ERROR_DISPLAY_TIME = 1.5
HAND_LOST_LIMIT = 0.2


# 상태 설정 

WAITING = 0
COUNTDOWN = 1
COLLECTING = 2
SHOW_RESULT = 3
SHOW_ERROR = 4

game_state = WAITING

countdown_start = None
collect_start = None
result_start = None
error_start = None
last_two_hands_time = None

error_text = ""
result_color = DRAW_COLOR

p1_votes = []
p2_votes = []

p1_result = None
p2_result = None
winner = None

p1_score = 0
p2_score = 0

score_reset_start = None
score_was_reset = False

# 손 검출기

detector = HandDetector(
    staticMode=False,
    maxHands=2,
    detectionCon=0.7,
    minTrackCon=0.5
)


# UI 함수

def drawText(frame, text, position, font_scale=0.7,
             color=UI_WHITE, thickness=2,
             font=UI_FONT, background=None):

    x, y = position

    if background is not None:
        (tw, th), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        cv2.rectangle(
            frame,
            (x - 5, y - th - 5),
            (x + tw + 5, y + baseline + 5),
            background,
            -1
        )

    cv2.putText(
        frame, text, (int(x), int(y)),
        font, font_scale, color, thickness,
        cv2.LINE_AA
    )


def drawCenteredText(frame, text, y, font_scale=1.0,
                     color=UI_WHITE, thickness=2,
                     font=UI_FONT_BOLD):

    h, w = frame.shape[:2]

    (tw, th), _ = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x = (w - tw) // 2

    drawText(
        frame, text, (x, y),
        font_scale, color, thickness, font
    )


def drawPanel(frame, x1, y1, x2, y2,
              color=UI_BG_COLOR, alpha=UI_PANEL_ALPHA):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        -1
    )

    return cv2.addWeighted(
        overlay, alpha,
        frame, 1.0 - alpha,
        0
    )


def drawPlayerHeader(frame, p1_gesture=None, p2_gesture=None):

    h, w = frame.shape[:2]

    frame = drawPanel(
        frame, 0, 0, w, 48,
        UI_BG_COLOR, 0.88
    )

    cv2.line(
        frame, (w // 2, 5), (w // 2, 43),
        UI_GRAY, 1
    )

    p1_text = "P1  |  " + (
        p1_gesture.upper() if p1_gesture else str(p1_score)
    )

    p2_text = "P2  |  " + (
        p2_gesture.upper() if p2_gesture else str(p2_score)
    )

    drawText(
        frame, p1_text, (15, 31),
        0.55, P1_COLOR, 2, UI_FONT_BOLD
    )

    (tw, th), _ = cv2.getTextSize(
        p2_text, UI_FONT_BOLD, 0.55, 2
    )

    drawText(
        frame, p2_text, (w - tw - 15, 31),
        0.55, P2_COLOR, 2, UI_FONT_BOLD
    )

    return frame


def drawGameStatus(frame, state, countdown_number=None):

    h, w = frame.shape[:2]

    if state == WAITING:
        frame = drawPanel(
            frame, 35, 60, w - 35, 135,
            UI_BG_COLOR, 0.75
        )

        drawCenteredText(
            frame, "READY", 95,
            0.85, UI_YELLOW, 2
        )

        drawCenteredText(
            frame, "Show 2 hands to start", 120,
            0.40, UI_WHITE, 1, UI_FONT
        )

    elif state == COUNTDOWN:
        frame = drawPanel(
            frame, 25, 55, w - 25, 215,
            UI_BG_COLOR, 0.80
        )

        drawCenteredText(
            frame, "GET READY!", 95,
            0.70, UI_YELLOW, 2
        )

        if countdown_number is not None:
            drawCenteredText(
                frame, str(countdown_number), 180,
                2.8, UI_WHITE, 5
            )

    elif state == COLLECTING:
        frame = drawPanel(
            frame, 20, 55, w - 20, 100,
            UI_BG_COLOR, 0.72
        )

        drawCenteredText(
            frame, "SHOW YOUR GESTURE!", 88,
            0.52, UI_YELLOW, 2
        )

    return frame


def drawResultUI(frame, p1_result, p2_result, winner):

    h, w = frame.shape[:2]

    frame = drawPanel(
        frame, 15, 60, w - 15, 215,
        UI_BG_COLOR, 0.88
    )

    drawCenteredText(
        frame, "RESULT", 88,
        0.58, UI_WHITE, 2
    )

    p1_text = f"P1: {p1_result.upper()}" if p1_result else str(p1_score)
    p2_text = f"P2: {p2_result.upper()}" if p2_result else str(p2_score)

    drawText(
        frame, p1_text, (30, 132),
        0.48, P1_COLOR, 2, UI_FONT_BOLD
    )

    drawCenteredText(
        frame, "VS", 132,
        0.42, UI_GRAY, 1, UI_FONT
    )

    (tw, th), _ = cv2.getTextSize(
        p2_text, UI_FONT_BOLD, 0.48, 2
    )

    drawText(
        frame, p2_text, (w - tw - 30, 132),
        0.48, P2_COLOR, 2, UI_FONT_BOLD
    )

    if winner == "P1 Wins":
        winner_color = WINNER_P1_COLOR
        winner_text = "P1 WINS!"
    elif winner == "P2 Wins":
        winner_color = WINNER_P2_COLOR
        winner_text = "P2 WINS!"
    elif winner == "Draw":
        winner_color = DRAW_COLOR
        winner_text = "DRAW!"
    else:
        winner_color = UI_RED
        winner_text = "INVALID"

    drawCenteredText(
        frame, winner_text, 190,
        0.95, winner_color, 3
    )

    return frame


def drawErrorUI(frame, text):

    h, w = frame.shape[:2]

    frame = drawPanel(
        frame, 20, 65, w - 20, 150,
        UI_BG_COLOR, 0.85
    )

    drawCenteredText(
        frame, "ERROR", 100,
        0.65, UI_RED, 2
    )

    drawCenteredText(
        frame, text, 130,
        0.42, UI_WHITE, 1, UI_FONT
    )

    return frame


def drawBottomStatus(frame, hand_count, fps):

    h, w = frame.shape[:2]

    frame = drawPanel(
        frame, 0, h - 28, w, h,
        UI_BG_COLOR, 0.85
    )

    drawText(
        frame, f"Hands: {hand_count}",
        (10, h - 9), 0.40, UI_WHITE, 1
    )

    fps_text = f"FPS: {fps:.1f}"

    (tw, th), _ = cv2.getTextSize(
        fps_text, UI_FONT, 0.40, 1
    )

    drawText(
        frame, fps_text,
        (w - tw - 10, h - 9),
        0.40, UI_GRAY, 1
    )

    return frame


# 이미지 전처리

def letterbox(img, new_shape=(IMG_H, IMG_W),
              color=(114, 114, 114)):

    h, w = img.shape[:2]
    nh, nw = new_shape

    ratio = min(nw / w, nh / h)

    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    resized = cv2.resize(img, (new_w, new_h))

    pad_w = nw - new_w
    pad_h = nh - new_h

    pad_left = pad_w // 2
    pad_top = pad_h // 2

    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, ratio, pad_left, pad_top


def preprocess_image(frame):

    img_rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    img_lb, ratio, pad_x, pad_y = letterbox(
        img_rgb, (IMG_H, IMG_W)
    )

    # float32 모델이므로 0~1 정규화
    img = img_lb.astype(np.float32) / 255.0

    # NHWC: (H, W, C) -> (1, H, W, C)
    img = np.expand_dims(img, axis=0)

    # NHWC -> NCHW
    # (1, 320, 320, 3) -> (1, 3, 320, 320)
    img = np.transpose(img, (0, 3, 1, 2))

    return img.astype(np.float32), ratio, pad_x, pad_y


# NMS

def apply_nms(boxes, scores):

    if len(boxes) == 0:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores.tolist(),
        CONF_TH,
        IOU_TH
    )

    if len(indices) == 0:
        return []

    return np.array(indices).flatten()


# LiteRT 탐지

def processImage(inference_frame, draw_frame=None):

    frame_height, frame_width = inference_frame.shape[:2]

    img, ratio, pad_x, pad_y = preprocess_image(
        inference_frame
    )

    # 반드시 원본 inference_frame으로만 추론
    interpreter.set_tensor(
        input_index,
        img
    )

    interpreter.invoke()

    raw_output = interpreter.get_tensor(
        output_index
    )

    # (1, 7, 2100) -> (2100, 7)
    raw = raw_output[0].transpose()

    # x, y, w, h, class1, class2, class3
    class_scores = raw[:, 4:]

    confidences = np.max(
        class_scores, axis=1
    )

    class_ids = np.argmax(
        class_scores, axis=1
    )

    keep_mask = confidences > CONF_TH

    filtered_raw = raw[keep_mask]
    scores = confidences[keep_mask]
    classes = class_ids[keep_mask]

    if len(filtered_raw) == 0:
        return []

    cx = filtered_raw[:, 0]
    cy = filtered_raw[:, 1]
    box_w = filtered_raw[:, 2]
    box_h = filtered_raw[:, 3]

    # 모델 좌표가 0~1 normalized라고 가정
    x = (cx - box_w / 2) * IMG_W
    y = (cy - box_h / 2) * IMG_H

    box_w = box_w * IMG_W
    box_h = box_h * IMG_H

    boxes = np.stack(
        [x, y, box_w, box_h],
        axis=-1
    )

    boxes_for_nms = [
        [
            int(bx), int(by),
            int(bw), int(bh)
        ]
        for bx, by, bw, bh in boxes
    ]

    keep_indices = apply_nms(
        boxes_for_nms, scores
    )

    detections = []

    for i in keep_indices:

        bx, by, bw, bh = boxes[i]

        # letterbox 좌표 -> 원본 프레임 좌표
        x1 = (bx - pad_x) / ratio
        y1 = (by - pad_y) / ratio
        x2 = (bx + bw - pad_x) / ratio
        y2 = (by + bh - pad_y) / ratio

        x1 = int(np.clip(x1, 0, frame_width - 1))
        y1 = int(np.clip(y1, 0, frame_height - 1))
        x2 = int(np.clip(x2, 0, frame_width - 1))
        y2 = int(np.clip(y2, 0, frame_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(classes[i])
        score = float(scores[i])

        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": (x1 + x2) // 2,
            "center_y": (y1 + y2) // 2,
            "score": score,
            "class_id": class_id
        })

    detections.sort(
        key=lambda d: d["center_x"]
    )

    # 화면 표시용 박스는 추론 후 draw_frame에 그림
    if draw_frame is not None:

        for det in detections:

            x1 = det["x1"]
            y1 = det["y1"]
            x2 = det["x2"]
            y2 = det["y2"]

            class_id = det["class_id"]
            score = det["score"]

            if class_id not in ansToText:
                continue

            color = colorList[class_id]

            cv2.rectangle(
                draw_frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            drawText(
                draw_frame,
                f"{ansToText[class_id]} {int(score * 100)}%",
                (x1, max(y1 - 7, 15)),
                0.5,
                color,
                2
            )

    return detections


# 플레이어 분리 및 판정

def separatePlayers(detections):

    if len(detections) != 2: # 가위바위보 2개 아니면 해당 프레임정보 버림
        return None, None

    detections = sorted(
        detections,
        key=lambda d: d["center_x"]
    )

    return detections[0], detections[1]

# 가위바위보 로직
def checkWinner(p1_gesture, p2_gesture):

    if p1_gesture == p2_gesture:
        return "Draw"

    if p1_gesture == "scissors":
        return "P1 Wins" if p2_gesture == "paper" else "P2 Wins"

    if p1_gesture == "rock":
        return "P1 Wins" if p2_gesture == "scissors" else "P2 Wins"

    if p1_gesture == "paper":
        return "P1 Wins" if p2_gesture == "rock" else "P2 Wins"

    return "Invalid"


def getMajority(votes):

    if len(votes) == 0:
        return None

    return Counter(votes).most_common(1)[0][0]


def getResultColor(winner):

    if winner == "P1 Wins":
        return WINNER_P1_COLOR

    if winner == "P2 Wins":
        return WINNER_P2_COLOR

    return DRAW_COLOR


# 화면 오버레이

def applyHandOverlay(frame, hands, original_frame):

    if len(hands) != 2:
        return frame

    h, w = frame.shape[:2]

    hands_sorted = sorted(
        hands,
        key=lambda hand: hand["center"][0]
    )

    left_hand = hands_sorted[0]
    right_hand = hands_sorted[1]

    split_x = int(
        (
            left_hand["center"][0]
            + right_hand["center"][0]
        ) / 2
    )

    split_x = int(np.clip(split_x, 0, w))

    overlay = frame.copy()

    cv2.rectangle(
        overlay, (0, 0), (split_x, h),
        P1_COLOR, -1
    )

    cv2.rectangle(
        overlay, (split_x, 0), (w, h),
        P2_COLOR, -1
    )

    frame = cv2.addWeighted(
        overlay,
        PLAYER_OVERLAY_ALPHA,
        frame,
        1.0 - PLAYER_OVERLAY_ALPHA,
        0
    )

    # 손 영역은 원본으로 복원
    for hand in (left_hand, right_hand):

        x, y, bw, bh = hand["bbox"]

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w, int(x + bw))
        y2 = min(h, int(y + bh))

        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = original_frame[y1:y2, x1:x2]

    for hand, color in [
        (left_hand, P1_COLOR),
        (right_hand, P2_COLOR)
    ]:

        x, y, bw, bh = hand["bbox"]

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w - 1, int(x + bw))
        y2 = min(h - 1, int(y + bh))

        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            color, 2
        )

    drawText(
        frame, "P1", (15, 75),
        0.65, P1_COLOR, 2, UI_FONT_BOLD
    )

    drawText(
        frame, "P2", (w - 45, 75),
        0.65, P2_COLOR, 2, UI_FONT_BOLD
    )

    return frame


def applyResultOverlay(frame, result_color, hands, original_frame):

    h, w = frame.shape[:2]

    overlay = np.zeros_like(frame)
    overlay[:] = result_color

    result_frame = cv2.addWeighted(
        overlay,
        RESULT_OVERLAY_ALPHA,
        frame,
        1.0 - RESULT_OVERLAY_ALPHA,
        0
    )

    for hand in hands:

        x, y, bw, bh = hand["bbox"]

        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(w, int(x + bw))
        y2 = min(h, int(y + bh))

        if x2 > x1 and y2 > y1:
            result_frame[y1:y2, x1:x2] = (
                original_frame[y1:y2, x1:x2]
            )

    return result_frame


# 카메라

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc("M", "J", "P", "G")
)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("cam", 480, 360)


# 메인 루프

previous_time = time.time()

try:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        current_time = time.time()
        original_frame = frame.copy()

        # 손 검출은 원본 프레임에서 수행
        hands, _ = detector.findHands(
            original_frame.copy(),
            draw=False,
            flipType=False
        )

        hand_count = len(hands)
        two_hands_detected = hand_count == 2

        # 표시용 프레임
        display_frame = original_frame.copy()

        detections = []
        p1, p2 = None, None
        two_gestures_detected = False

        if game_state == COLLECTING:

            detections = processImage(
                original_frame,
                draw_frame=display_frame
            )

            p1, p2 = separatePlayers(detections)

            two_gestures_detected = (
                p1 is not None and p2 is not None
            )
            
        # 상태 로직

        if game_state == WAITING:
            
            # 2명 미인식 시간 확인
            if not two_hands_detected:
                # 손 2개 0.2초 동안 없으면 점수 리셋
                if score_reset_start is None:
                    score_reset_start = current_time
                elif (
                    current_time - score_reset_start >= HAND_LOST_LIMIT
                    and not score_was_reset
                ):
                    # 스코어 리셋 코드 여기다 작성
                    p1_score = 0
                    p2_score = 0
                    print("score reset")
            
                    score_was_reset = True
            else:

                score_reset_start = None
                score_was_reset = False

            if two_hands_detected:

                game_state = COUNTDOWN
                countdown_start = current_time
                last_two_hands_time = current_time

                p1_votes.clear()
                p2_votes.clear()

                p1_result = None
                p2_result = None
                winner = None

        elif game_state == COUNTDOWN:

            elapsed = current_time - countdown_start

            if two_hands_detected:
                last_two_hands_time = current_time

            if (
                last_two_hands_time is None
                or current_time - last_two_hands_time > HAND_LOST_LIMIT
            ):

                error_text = "Cannot start the game"
                error_start = current_time
                game_state = SHOW_ERROR

                p1_votes.clear()
                p2_votes.clear()

            elif elapsed >= COUNTDOWN_TIME:

                game_state = COLLECTING
                collect_start = current_time

                p1_votes.clear()
                p2_votes.clear()

        elif game_state == COLLECTING:

            elapsed_collect = current_time - collect_start

            if two_hands_detected and two_gestures_detected:

                p1_gesture = ansToText.get(p1["class_id"])
                p2_gesture = ansToText.get(p2["class_id"])

                if p1_gesture is not None and p2_gesture is not None:
                    p1_votes.append(p1_gesture)
                    p2_votes.append(p2_gesture)

            if elapsed_collect >= COLLECT_TIME:

                p1_result = getMajority(p1_votes)
                p2_result = getMajority(p2_votes)

                if p1_result is None or p2_result is None:

                    error_text = "Cannot detect gestures"
                    error_start = current_time
                    game_state = SHOW_ERROR

                else:

                    winner = checkWinner(
                        p1_result,
                        p2_result
                    )

                    result_color = getResultColor(winner)
                    # 승리 플레이어 점수 증가 
                    if winner == "P1 Wins":
                        p1_score +=1
                        print(p1_score)
                    elif winner == "P2 Wins":
                        p2_score += 1
                        print(p2_score)
                    result_start = current_time
                    game_state = SHOW_RESULT

        elif game_state == SHOW_RESULT:

            display_frame = applyResultOverlay(
                display_frame,
                result_color,
                hands,
                original_frame
            )

            elapsed_result = current_time - result_start

            if elapsed_result >= RESULT_DISPLAY_TIME:

                game_state = WAITING
                result_start = None
                p1_result = None
                p2_result = None
                winner = None

                p1_votes.clear()
                p2_votes.clear()

        elif game_state == SHOW_ERROR:

            elapsed_error = current_time - error_start

            if elapsed_error >= ERROR_DISPLAY_TIME:

                game_state = WAITING
                error_start = None
                error_text = ""

                p1_votes.clear()
                p2_votes.clear()

        # 화면 오버레이

        if game_state != SHOW_RESULT:

            if two_hands_detected:
                display_frame = applyHandOverlay(
                    display_frame,
                    hands,
                    original_frame
                )

        if game_state == SHOW_RESULT:

            display_frame = drawResultUI(
                display_frame,
                p1_result,
                p2_result,
                winner
            )

        elif game_state == SHOW_ERROR:

            display_frame = drawErrorUI(
                display_frame,
                error_text
            )

        else:

            countdown_number = None

            if game_state == COUNTDOWN:

                elapsed = current_time - countdown_start
                remaining = COUNTDOWN_TIME - elapsed

                if remaining > 0:
                    countdown_number = int(np.ceil(remaining))

            display_frame = drawGameStatus(
                display_frame,
                game_state,
                countdown_number
            )

        display_frame = drawPlayerHeader(
            display_frame,
            p1_result if game_state == SHOW_RESULT else None,
            p2_result if game_state == SHOW_RESULT else None
        )

        fps = 1.0 / max(
            current_time - previous_time,
            1e-6
        )

        previous_time = current_time

        display_frame = drawBottomStatus(
            display_frame,
            hand_count,
            fps
        )

        cv2.imshow("cam", display_frame)

        key = cv2.waitKey(1) & 0xFF
        

        if key == ord("q"):
            break

finally:

    cap.release()
    cv2.destroyAllWindows()
