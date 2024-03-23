import environs
import inspect
import pathlib
from . import log_util
from dotenv import load_dotenv, dotenv_values

env = load_dotenv(override=True)
env = dotenv_values()
#env = environs.Env()
#env.read_env(".env", overwrite=True)

class config:
    ALIYUN_OSS_URL = "assets.zilliz.com.cn/benchmark/"
    AWS_S3_URL = "assets.zilliz.com/benchmark/"

    LOG_LEVEL = env.get("LOG_LEVEL", "INFO")

    LOG_PATH = env.get("LOG_PATH", "logs.txt")

    DEFAULT_DATASET_URL = env.get("DEFAULT_DATASET_URL", AWS_S3_URL)
    DATASET_LOCAL_DIR = env.get("DATASET_LOCAL_DIR", "/tmp/vectordb_bench/dataset")
    NUM_PER_BATCH = env.get("NUM_PER_BATCH", 5000)

    DROP_OLD = env.get("DROP_OLD", True)
    USE_SHUFFLED_DATA = env.get("USE_SHUFFLED_DATA", True)
    NUM_CONCURRENCY = [1, 5, 10, 15, 20, 25, 30, 35]

    RESULTS_DIR = pathlib.Path(__file__).parent.joinpath("results").__str__()
    RESULTS_LOCAL_DIR = pathlib.Path(env.get("RESULTS_PATH", RESULTS_DIR))

    CAPACITY_TIMEOUT_IN_SECONDS = 24 * 3600 # 24h
    LOAD_TIMEOUT_DEFAULT        = 2.5 * 3600 # 2.5h
    LOAD_TIMEOUT_768D_1M        = 2.5 * 3600 # 2.5h
    LOAD_TIMEOUT_768D_10M       =  25 * 3600 # 25h
    LOAD_TIMEOUT_768D_100M      = 250 * 3600 # 10.41d

    LOAD_TIMEOUT_1536D_500K     = 2.5 * 3600 # 2.5h
    LOAD_TIMEOUT_1536D_5M       =  25 * 3600 # 25h

    OPTIMIZE_TIMEOUT_DEFAULT    = 15 * 60   # 15min
    OPTIMIZE_TIMEOUT_768D_1M    =  15 * 60   # 15min
    OPTIMIZE_TIMEOUT_768D_10M   = 2.5 * 3600 # 2.5h
    OPTIMIZE_TIMEOUT_768D_100M  =  25 * 3600 # 1.04d


    OPTIMIZE_TIMEOUT_1536D_500K =  15 * 60   # 15min
    OPTIMIZE_TIMEOUT_1536D_5M   =   2.5 * 3600 # 2.5h
    def display(self) -> str:
        tmp = [
            i for i in inspect.getmembers(self)
            if not inspect.ismethod(i[1])
            and not i[0].startswith('_')
            and "TIMEOUT" not in i[0]
        ]
        return tmp

log_util.init(config.LOG_LEVEL, config.LOG_PATH)