from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.pgvector.config import PgVectorConfig, PgVectorIndexConfig
from vectordb_bench.interface import BenchMarkRunner
from vectordb_bench.metric import Metric
from vectordb_bench.models import CaseConfig, TaskConfig, IndexType
import logging

log = logging.getLogger(__name__)

def main():
    runner = BenchMarkRunner()

    db_config=PgVectorConfig(db_label='label', user_name='postgres', password='1234', host='localhost', port=5439, db_name='ann')

    #db_case_config=DB.PgVector.case_config_cls(index_type=IndexType.IVFFlat)(metric_type=None, lists=1000, probes=2)
    db_case_config=DB.PgVector.case_config_cls(index_type=IndexType.HNSW)(metric_type=None, m=10, ef_construction=10, ef=5)

    case_config=CaseConfig(case_id=CaseType.Performance768D1M)

    task_config=TaskConfig(
        db=DB.PgVector,
        db_config=db_config,
        db_case_config=db_case_config,
        case_config=case_config
    )

    runner.run([task_config])
    runner._sync_running_task()

    #result = runner.get_results()
    #log.info(f"test result: {result}")
    #print(result);

if __name__ == '__main__':
    main()
