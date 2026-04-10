Коллаб: как быстро запустить подготовку датасета и тренировку из этого репозитория

Коротко — шаги:
1. В Google Colab выберите Runtime -> Change runtime type -> GPU (preferably T4/P100/P4/RTX).
2. Смонтируйте Google Drive (если хотите сохранять результаты):
   from google.colab import drive
   drive.mount('/content/drive')
3. Скопируйте репозиторий в сессию (либо git clone, либо загрузите архив):
   git clone <your-repo-url> repo && cd repo
4. Установите зависимости (используйте virt env или системно). Пример быстрый:
   pip install -r notebooks/requirements-colab.txt
   # Установите torch соответствующий CUDA версии Colab (пример для CUDA 11.7/T4):
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

5. Отредактируйте notebooks/colab_config.sh если требуется (пути/режимы). По умолчанию скрипт ожидает структуру:
   ${DATA_DIR}/raw  - сырые файлы (obj / images и т.п.)
   ${DATA_DIR}/processed - куда положит prepare_dataset
   ${OUTPUT_DIR} - куда сохранять результаты обучения

6. Запустите подготовленный скрипт (быстрый smoke-run, безопасный для проверки):
   bash notebooks/run_colab.sh --mode smoke

Описание файлов в папке notebooks/
- colab_config.sh — набор переменных окружения по умолчанию (DATA_DIR, OUTPUT_DIR, INPUT_MODE и т.д.).
- run_colab.sh — основной исполняемый скрипт. Поддерживает режимы: smoke (короткая проверка), full (полный тренинг), prepare-only, train-base, train-refiner, train-vae, evaluate.
- requirements-colab.txt — минимальный набор pip-зависимостей, которые ускорят настройку в Colab.

Советы
- Для длительных тренировок сохраняйте checkpoints в Google Drive.
- Если у вас старые чекпоинты с 3 каналами, используйте --input_mode tri или патчите веса (см. docs).
- Для быстрого отлова ошибок сначала используйте --mode smoke, затем увеличивайте epochs/batch_size.

Если нужно — могу сгенерировать полноценный Colab notebook (.ipynb) с ячейками для каждого шага.
