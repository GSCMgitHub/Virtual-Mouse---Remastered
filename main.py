import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from pynput.mouse import Button, Controller
from datetime import datetime

# --- Importaciones modernas de la Tasks API ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
mouse = Controller()
screen_width, screen_height = pyautogui.size()

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

action_done = '[ACTION]'

def get_angle(a, b, c):
    '''Calcula el ángulo entre tres puntos (a, b, c) dados como tuplas (x, y).'''
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    return np.abs(np.degrees(radians))

def get_distance(landmark_list):
    '''Calcula la distancia entre dos puntos dados como tuplas (x, y).'''
    if len(landmark_list) < 2:
        return 0
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)
    return np.interp(L, [0, 1], [0, 1000])

def move_mouse(index_finger_tip):
    '''Mueve el mouse a la posición dada por el índice del dedo (index_finger_tip) que es un landmark de MediaPipe.'''
    if index_finger_tip is not None:
        #print(index_finger_tip.x, index_finger_tip.y)
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y)
        global action_done
        action_done = 'Mouse Moved'


def is_left_click(landmark_list, thumb_index_dist):
    '''Detecta si el gesto corresponde a un clic izquierdo basado en los landmarks y la distancia entre el pulgar y el índice.'''
    global action_done
    action_done = 'Left Click'
    return (get_distance([landmark_list[5], landmark_list[8]]) < 40 and
             get_distance([landmark_list[9], landmark_list[12]]) > 40 and
             thumb_index_dist > 40)


def is_right_click(landmark_list, thumb_index_dist):
    '''Detecta si el gesto corresponde a un clic derecho basado en los landmarks y la distancia entre el pulgar y el índice.'''
    global action_done
    action_done = 'Right Click'
    return (get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
            thumb_index_dist > 50)

def is_double_click(landmark_list, thumb_index_dist): 
    '''Detecta si el gesto corresponde a un doble clic basado en los landmarks y la distancia entre el pulgar y el índice.'''
    global action_done
    action_done = 'Double Click'
    return (get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50)

def is_screenshot(landmark_list, thumb_index_dist):
    '''Detecta si el gesto corresponde a tomar una captura de pantalla basado en los landmarks y la distancia entre el pulgar y el índice.'''
    global action_done
    action_done = 'Screenshot'
    return (get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) < 50 and
            get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) < 50 and
            thumb_index_dist < 50)

def detect_gesture(frame, landmark_list, hand_landmarks):
    '''Detecta gestos basados en los landmarks y ejecuta acciones como mover el mouse o hacer clic.'''
      # Contador para el número de capturas de pantalla tomadas
    min_intervals_between_screenshots = 500  # Número mínimo de intervalos entre capturas de pantalla para evitar múltiples capturas en rápida sucesión
    intervals_since_last_screenshot = min_intervals_between_screenshots # Contador para evitar múltiples capturas de pantalla en rápida sucesión

    if len(landmark_list) >= 21:
        # El índice 8 es la punta del dedo índice (INDEX_FINGER_TIP)
        index_finger_tip = hand_landmarks[8]
        thumb_index_dist = get_distance([landmark_list[4], landmark_list[5]])

        angulo = get_angle(landmark_list[5], landmark_list[6], landmark_list[8])
        print(thumb_index_dist, angulo)

        if thumb_index_dist < 90 and angulo > 175: 
            move_mouse(index_finger_tip)
            #print("Mouse moved to:", (index_finger_tip.x, index_finger_tip.y))

        # TODO: Ver si hay que ajustar los demás gestos para que ninguno interfiera con el otro.
        elif is_left_click(landmark_list, thumb_index_dist):
           mouse.press(Button.left)
           mouse.release(Button.left)
           cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           print("Left Click performed")

        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("Right Click performed")

        elif is_screenshot(landmark_list, thumb_index_dist):
            if intervals_since_last_screenshot >= min_intervals_between_screenshots:  # Evita múltiples capturas de pantalla en rápida sucesión
                im1 = pyautogui.screenshot()
                label = datetime.now().strftime("%Y%m%d_%H%M%S")  # Etiqueta con la fecha y hora actual para evitar sobrescribir archivos
                im1.save(f'vm_screenshot_{label}.png')
                cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                pyautogui.sleep(0.4)  # Pequeña pausa para evitar múltiples capturas de pantalla en rápida sucesión
                intervals_since_last_screenshot = 0
                print("Screenshot taken and saved as:", f'vm_screenshot_{label}.png')

    intervals_since_last_screenshot += 1

def draw_landmarks_on_image(rgb_image, detection_result):
  '''Dibuja los landmarks y la información de la mano detectada en la imagen (por ejemplo: qué gesto es).'''

  hand_landmarks_list = detection_result.hand_landmarks
 # handedness_list = detection_result.handedness # No se utiliza
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    #handedness = handedness_list[idx] # Reemplazado por la acción realizada (ej: "Left Click", "Right Click", etc.)

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Escribe el nombre del gesto detectado (o la mano) en la imagen
    cv2.putText(annotated_image, action_done, #f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    #formato: cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)

  return annotated_image


def main():
    '''Función principal que captura video, detecta gestos y mueve el mouse.'''

    # Recuerda descargar 'hand_landmarker.task' y ponerlo en la misma carpeta
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options = base_options,
        running_mode = vision.RunningMode.VIDEO,
        num_hands = 1,
        min_hand_detection_confidence = 0.5,
        min_hand_presence_confidence = 0.4,
        min_tracking_confidence = 0.1
    )

    cap = cv2.VideoCapture(0)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Formato requerido por Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB)
            frame_timestamp_ms = int(time.time() * 1000)
            
            # Inferencia con Tasks API
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if detection_result.hand_landmarks:
                hand_landmarks = detection_result.hand_landmarks[0]
                
                # 1. Extraer lista de tuplas para la lógica matemática de tus gestos
                landmarks_list = [(lm.x, lm.y) for lm in hand_landmarks]
                
                # 3. Detectar gestos y mover el mouse
                detect_gesture(frame, landmarks_list, hand_landmarks)

                # 4. Visualizar resultados (opcional pero útil para debugging)
                annotated_image = draw_landmarks_on_image(frameRGB, detection_result)
            else:
                annotated_image = frameRGB
            
            cv2.imshow('Frame', annotated_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()