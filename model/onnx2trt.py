import tensorrt as trt
import onnx
import logging
import argparse
import os

# Настройка логгера
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

class TestOnnxModel:
    @staticmethod
    def checkup_model(model: str):
        logger.info("[INFO] Проверка ONNX модели...")
        try:
            onnx_model = onnx.load(model)
            onnx.checker.check_model(onnx_model)
            logger.info("[INFO] Модель корректна!")
        except Exception as e:
            logger.error(f"[ERROR] Модель повреждена: {e}")
            exit()

class LoggerTrt:
    @staticmethod
    def create_logger(flag_verbose: bool) -> trt.Logger:
        return trt.Logger(trt.Logger.VERBOSE if flag_verbose else trt.Logger.INFO)

class Builder:
    @staticmethod
    def build_engine(onnx_file_path: str, h: int, w: int, core_num: int, flag_verbose: bool, memory_limit_gb: int = 1):
        TestOnnxModel.checkup_model(onnx_file_path)
        loggerTrt = LoggerTrt.create_logger(flag_verbose)
        logger.info("[INFO] Создан TensorRT логгер")

        builder = trt.Builder(loggerTrt)
        logger.info("[INFO] Создан TensorRT Builder")

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, loggerTrt)

        logger.info(f"[INFO] Загружаем ONNX-модель: {onnx_file_path}")
        if not parser.parse_from_file(onnx_file_path):
            for i in range(parser.num_errors):
                logger.error(f"[ERROR] {parser.get_error(i)}")
            raise RuntimeError("Ошибка при парсинге ONNX файла!")

        logger.info("[INFO] ONNX успешно разобран!")

        # Создание TensorRT движка
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.TF32)
        config.set_flag(trt.BuilderFlag.FP16)

        logger.info("[INFO] Добавлены флаги TF32 и FP16")

        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        profile.set_shape(input_tensor.name, (1, 3, h, w), (1, 3, h, w), (1, 3, h, w))
        config.add_optimization_profile(profile)

        logger.info("[INFO] Добавлен оптимизационный профиль")

        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, memory_limit_gb << 30)
        logger.info("[INFO] Установлен лимит памяти: {} GB".format(memory_limit_gb))

        if core_num > 0 and builder.num_DLA_cores > 0:
            config.default_device_type = trt.DeviceType.DLA
            config.set_device_type(trt.DeviceType.DLA)
            config.DLA_core = core_num
            logger.info(f"[INFO] Используется DLA с {core_num} ядрами")

        # Создание движка
        logger.info("[INFO] Создаем движок...")
        engine = builder.build_serialized_network(network, config)

        if engine is None:
            logger.error("[ERROR] Движок не создан! Проверьте совместимость модели и TensorRT.")
            exit()

        # Генерация имени файла для движка
        output_name = os.path.splitext(onnx_file_path)[0] + ".engine"
        with open(output_name, "wb") as f:
            f.write(engine)
            logger.info(f"[INFO] Движок сохранен в файл: {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Включить детализированный вывод (для отладки)')
    parser.add_argument("--onnx-file", default="model.onnx",
                        help="Путь к ONNX-файлу")
    parser.add_argument('--dla-core', type=int, default=0,
                        help='ID DLA ядра для инференса (0 ~ N-1)')
    parser.add_argument('--img', type=int, nargs='+',
                        help='Размер изображения: ширина, высота')

    args = parser.parse_args()

    Builder.build_engine(args.onnx_file, args.img[0], args.img[1], args.dla_core, args.verbose)
