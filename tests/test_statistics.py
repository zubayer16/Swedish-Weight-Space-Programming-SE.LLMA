import pytest
from src.evaluation import paired_bootstrap


def test_report_measurements_include_sentence_final_values():
    from src.reporting import reported_values
    assert reported_values('Improvement 0.322. English 3.698, base 2.871.') == {'0.322','3.698','2.871'}
    assert reported_values('Do not round 3.6981 or match inside 13.6981.') == set()


def test_paired_bootstrap_token_weighted_and_sign():
    base = [{'id':'a','tokens':1,'loss_sum':2.},{'id':'b','tokens':9,'loss_sum':18.}]
    other = [{'id':'a','tokens':1,'loss_sum':3.},{'id':'b','tokens':9,'loss_sum':27.}]
    result = paired_bootstrap(base,other)
    assert result == {'delta_nll':1.,'ci_low':1.,'ci_high':1.}
    assert paired_bootstrap(base,base)['ci_high'] == 0
    assert paired_bootstrap(other,base)['delta_nll'] == -1
    with pytest.raises(ValueError,match='paired'):
        paired_bootstrap(base,other[::-1])
    other[0]['tokens'] = 2
    with pytest.raises(ValueError,match='counts'):
        paired_bootstrap(base,other)


def test_bootstrap_preserves_pairing_and_reproducibility():
    base = [{'id':str(i),'tokens':i+1,'loss_sum':10*(i+1)} for i in range(10)]
    other = [dict(d,loss_sum=d['loss_sum']+.5*d['tokens']) for d in base]
    a = paired_bootstrap(base,other)
    assert a == paired_bootstrap(base,other)
    assert a['ci_low'] == pytest.approx(.5)
    assert a['ci_high'] == pytest.approx(.5)
