"""
Одноразовий скрипт перевірки середовища.

Запустіть його ЗАЗДАЛЕГІДЬ, а не на занятті: окрім перевірки пакетів,
він завантажує ваги моделі yolov8n з мережі, і цей крок може тривати
кілька десятків секунд.

Використання:
    python check_env.py
"""
import sys


def check_imports() -> None:
    required = ["fastapi", "uvicorn", "PIL", "ultralytics"]
    missing = []
    for name in required:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        print(f"Відсутні пакети: {', '.join(missing)}")
        print("Встановіть їх командою: pip install -r requirements.txt")
        sys.exit(1)
    print("Усі необхідні пакети встановлено.")


def check_model_download() -> None:
    from ultralytics import YOLO

    print("Завантаження ваг yolov8n.pt (перший раз може тривати до хвилини)...")
    model = YOLO("yolov8n.pt")
    print("Ваги завантажено, модель ініціалізовано успішно.")
    sample_classes = list(model.names.values())[:5]
    print(f"Приклад класів моделі: {sample_classes} ...")


if __name__ == "__main__":
    check_imports()
    check_model_download()
    print("\nСередовище готове. Запуск застосунку:")
    print("    uvicorn app.main:app --reload")
