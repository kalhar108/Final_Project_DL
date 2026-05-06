from claimlens.monitoring import drift_report
from claimlens.retrieval import TfidfRetriever

def test_retriever_returns_result():
    r=TfidfRetriever().fit(['model training pipeline','frontend interface'])
    assert r.search('training', k=1)[0]['passage']=='model training pipeline'

def test_drift_report_has_action():
    out=drift_report(['a b c'], ['a b d'], threshold=1.0)
    assert out['action']=='continue_monitoring'
