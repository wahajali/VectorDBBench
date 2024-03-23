import pathlib
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.pgvector.config import PgVectorConfig, PgVectorIndexConfig
from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, TaskConfig, IndexType
import vectordb_bench
import logging
import subprocess
import time
import psycopg2
import json
import importlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def setup_db(config):

    # Database connection parameters
    hostname = config["host"]
    username = config["user_name"]
    password = config["password"]
    port = config["port"]

    # Connect to the PostgreSQL database
    conn = psycopg2.connect(host=hostname, user=username, password=password, dbname="postgres", port=port)
    conn.autocommit = True

    # Create a cursor object
    cur = conn.cursor()

    log.info("Setup Database...")

    try:
        # Create Database
        log.info("Creating Database...")
        cur.execute("CREATE DATABASE ann")

        # Set Maintenance Work MEM
        log.info("Set Maintenance Work MEM...")
        cur.execute(f'ALTER USER {username} SET maintenance_work_mem = "4GB"')

        # Set Parallel Maintenance Workers
        log.info("Set Parallel Maintenance Workers...")
        cur.execute(f'ALTER USER {username} SET max_parallel_maintenance_workers = 2')

    except psycopg2.Error as e:
        log.error(f'Error: {e}')

    finally:
        # Close cursor and connection
        cur.close()
        conn.close()

    conn = psycopg2.connect(host=hostname, user=username, password=password, dbname="ann", port=port)
    cur = conn.cursor()
        # Enable Extension
    try:
        log.info("Enable Extension...")
        cur.execute("CREATE EXTENSION vector")
        conn.commit()
        log.info("Operations completed successfully!")
    except psycopg2.Error as e:
        log.error(f'Error: {e}')

    finally:
        # Close cursor and connection
        cur.close()
        conn.close()

def update_env(label): 
    path = pathlib.Path(__file__).parent.joinpath("results_final").joinpath(label)
    print(path)
    path.mkdir(parents=True, exist_ok=True)
    config_content = f"""\
LOG_LEVEL=DEBUG
LOG_PATH={path.__str__()+"/log.txt"}
RESULTS_PATH={path}

DATASET_LOCAL_DIR="/tmp/vector_db_bench/dataset"
"""

    file_path = ".env"

    # Write the configuration content to the file
    with open(file_path, "w") as file:
        file.write(config_content)

    print(f"Configuration file '{file_path}' created/overwritten successfully.")

def main():

    with open("config.json") as f:
        configuration=json.load(f)
    
    database_config = configuration["database_config"]
    cases = configuration["cases"]

    setup_db(database_config)

    for case in cases:
        runs = case["runs"]
        for i in range(runs):

            index_params = case["index_params"]
            search_params = case["search_params"]

            for j, search_param in enumerate(search_params): 
                if case["index"] == "IVFFLAT":
                    label = f'{database_config["hyperscalar"]}-{case["index"]}-{str(index_params["lists"])}-{str(search_param["probes"])}-{database_config["db_instance"]}-run_{str(i)}'
                    db_case_config=DB.PgVector.case_config_cls(index_type=IndexType.IVFFlat)(metric_type=None, lists=index_params["lists"], probes=search_param["probes"])
                elif case["index"] == "HNSW":
                    label = f'{database_config["hyperscalar"]}-{case["index"]}-{str(index_params["m"])}-{str(index_params["ef_construction"])}-{str(search_param["ef_search"])}-{database_config["db_instance"]}'
                    db_case_config=DB.PgVector.case_config_cls(index_type=IndexType.HNSW)(metric_type=None, m=index_params["m"], ef_construction=index_params["ef_construction"], ef=search_param["ef_search"])
                else:
                    log.info(case)
                    assert "Invalid index type specified"
                
                db_config=PgVectorConfig(db_label=label, user_name=database_config["user_name"], password=database_config["password"], host=database_config["host"], port=database_config["port"], db_name="ann")

                if case["test_type"] == "Performance768D1M":
                    case_config=CaseConfig(case_id=CaseType.Performance768D1M)

                task_config=TaskConfig(
                    db=DB.PgVector,
                    db_config=db_config,
                    db_case_config=db_case_config,
                    case_config=case_config
                )

                update_env(label)
                importlib.reload(vectordb_bench)
                log.info(vectordb_bench.config.display)

                log.info("task_config")
                log.info(task_config)
                log.info("db_config")
                log.info(db_config)
                log.info("db_case_config")
                log.info(db_case_config)
                log.info("case_config")
                log.info(case_config)

                log.info("******************STARTING EXECUTION******************\n\n")
                log.info("Task="+label)
                runner = BenchMarkRunner()

                if j == 0:
                    # Always drop on first run of a case
                    log.info("set drop_old = true, j=0")
                    runner.set_drop_old(True)
                elif case["drop_table"] == "false": 
                    log.info("set drop_old = false")
                    runner.set_drop_old(False)
                else: 
                    runner.set_drop_old(False)

                runner.run([task_config])
                runner._sync_running_task()
                log.info("Sleeping for 2 mins")
                log.info("******************COMPLETED EXECUTION******************\n\n")
                time.sleep(10)



    return
    runs = 1
    for i in range(runs):
        print("STARTING EXECUTION="+str(i))
        process = subprocess.Popen(["bash", "setup.sh"])
        process.wait()

        runner = BenchMarkRunner()

        label = 'ivvflat-1000-32-prewarm-' + str(i)
        print(label)
        db_config=PgVectorConfig(db_label=label, user_name='ann', password='ann', host='aws-postgres.c8jxjby7azcm.us-west-2.rds.amazonaws.com', port=5432, db_name='ann')

        #db_case_config=DB.PgVector.case_config_cls(index_type=IndexType.IVFFlat)(metric_type=None, lists=1000, probes=2)
        # db_case_config=DB.PgVector.case_config_cls(index_type=IndexType.HNSW)(metric_type=None, m=10, ef_construction=10, ef=5)
        db_case_config=DB.PgVector.case_config_cls(index_type=IndexType.IVFFlat)(metric_type=None, lists= 1000, probes= 32)

        case_config=CaseConfig(case_id=CaseType.Performance768D1M)

        task_config=TaskConfig(
            db=DB.PgVector,
            db_config=db_config,
            db_case_config=db_case_config,
            case_config=case_config
        )

        runner.run([task_config])
        runner._sync_running_task()
        print("COMPLETED EXECUTION"+str(1))
        print("Sleeping for 2 mins")
        time.sleep(120)

        #result = runner.get_results()
        #log.info(f"test result: {result}")
        #print(result);

if __name__ == '__main__':
    main()