import numpy as np
import pytest
import torch
from src.patches import make_patch, allocate, largest
from src.models import snapshot,compute_delta,apply_patch
from src.patches import balanced_budgets, make_balanced_patch
from src.patches import scaled_dense_patch
from src.evaluation import select_dense_alpha


def test_dense_scaling_endpoints_and_no_input_mutation():
    model = torch.nn.Linear(3,2,bias=False)
    base = snapshot(model)
    adapted = {name:value + .2 for name,value in base.items()}
    delta = compute_delta(base,adapted)
    for alpha in [0.,.3,1.]:
        apply_patch(model,base,scaled_dense_patch(delta,alpha))
        for name,value in model.named_parameters():
            torch.testing.assert_close(value,base[name]*(1-alpha)+adapted[name]*alpha)
            torch.testing.assert_close(delta[name],adapted[name]-base[name])
    for alpha in [-.1,1.1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):
            scaled_dense_patch(delta,alpha)


def test_dense_selection_equal_language_weights_and_stable_ties():
    def metric(sv,en):
        return {'sv:test':{'nll':sv,'tokens':1000},'en:test':{'nll':en,'tokens':10}}
    # Pooling languages by token count would wrongly choose alpha 1.
    development = {'1.0':metric(1.,5.),'0.5':metric(2.,2.),'0.2':metric(2.,2.)}
    assert select_dense_alpha(development)['alpha'] == .2
    with pytest.raises(ValueError):
        select_dense_alpha({'0.2':metric(float('nan'),2.)})


def test_dare_uses_same_random_mask_and_actual_inverse_retention():
    base = {'model.layers.0.a':torch.zeros(23)}
    delta = {'model.layers.0.a':torch.arange(1,24,dtype=torch.float32)}
    random,info = make_patch(base,delta,.3,'random',29)
    dare,scaled = make_patch(base,delta,.3,'dare',29)
    assert info['selected_parameters'] == 6
    assert scaled['rescale'] == 23/6
    assert torch.equal(dare['model.layers.0.a'],random['model.layers.0.a']*(23/6))


def test_global_magnitude_crosses_tensor_boundaries_and_zero_deltas_count():
    delta = {'model.layers.0.a':torch.tensor([0.,-9.,1.]),
             'model.layers.1.b':torch.tensor([8.,2.,0.,0.])}
    patch,info = make_patch(delta,delta,.3,'magnitude')
    assert info['selected_parameters'] == 2
    assert patch['model.layers.0.a'].tolist() == [0.,-9.,0.]
    assert patch['model.layers.1.b'].tolist() == [8.,0.,0.,0.]
    _,full = make_patch(delta,delta,1.,'magnitude')
    assert full['selected_parameters'] == 7 and full['modified_parameters'] == 4


@pytest.mark.parametrize('method',['random','dare','magnitude','layer','layer_random'])
@pytest.mark.parametrize('fraction',[0,.1,.3,1])
def test_exact_budget_and_purity(method,fraction):
    base = {'model.layers.0.a':torch.ones(10),'model.layers.1.b':torch.ones(10)*2}
    delta = {n:torch.arange(1,11,dtype=torch.float32)/100 for n in base}
    saved = {n:t.clone() for n,t in delta.items()}
    patch,info = make_patch(base,delta,fraction,method,7)
    repeat,_ = make_patch(base,delta,fraction,method,7)
    assert info['selected_parameters'] == int(20*fraction)
    assert sum(p.count_nonzero().item() for p in patch.values()) == int(20*fraction)
    for n in base:
        assert torch.equal(patch[n],repeat[n])
        assert torch.equal(delta[n],saved[n])
        if fraction == 1:
            torch.testing.assert_close(patch[n],delta[n])
        if method == 'dare' and fraction > 0:
            mask = patch[n] != 0
            torch.testing.assert_close(patch[n][mask],delta[n][mask]/fraction)


def test_allocator_and_ties():
    with pytest.raises(ValueError):
        allocate([2,8],[1,1],11)
    with pytest.raises(ValueError):
        allocate([2,8],[float('nan'),1],5)
    assert allocate([2,8],[100,0],7) == [2,5]
    assert sum(allocate([2,8],[0,0],7)) == 7
    assert largest(np.ones(5),2).tolist() == [0,1]
    for k in range(11):
        result = allocate([2,3,5],[0,100,1],k)
        assert sum(result) == k
        assert all(0 <= a <= b for a,b in zip(result,[2,3,5]))


def test_restore_tied_weights_and_no_accumulation():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(3,3,bias=False)
            self.b = torch.nn.Linear(3,3,bias=False)
            self.b.weight = self.a.weight
    model = Tiny()
    base = snapshot(model)
    adapted = {n:p+.25 for n,p in base.items()}
    delta = compute_delta(base,adapted)
    assert len(base) == 1
    for f in [1,1,0]:
        patch,_ = make_patch(base,delta,f,'magnitude')
        apply_patch(model,base,patch)
        for n,p in model.named_parameters():
            torch.testing.assert_close(p,adapted[n] if f else base[n])
        assert model.a.weight.data_ptr() == model.b.weight.data_ptr()
    assert torch.equal(base['a.weight'],snapshot(model)['a.weight'])
    with pytest.raises(ValueError):
        apply_patch(model,base,{'bad':torch.ones(3)})


@pytest.mark.parametrize('fraction',[0,.1,.3,1])
def test_balanced_exact_rounding_magnitudes_and_purity(fraction):
    base = {'model.layers.0.a':torch.ones(7),'model.layers.1.b':torch.ones(13)*1000}
    delta = {n:torch.arange(1,p.numel()+1,dtype=torch.float32) for n,p in base.items()}
    saved = {n:v.clone() for n,v in delta.items()}
    patch,info = make_balanced_patch(base,delta,fraction)
    budgets = balanced_budgets([7,13],fraction)
    assert sum(budgets) == int(20*fraction) == info['selected_parameters']
    for (name,value),budget in zip(delta.items(),budgets):
        assert int(patch[name].count_nonzero()) == budget
        expected = value.clone()
        expected[:len(value)-budget] = 0
        assert torch.equal(patch[name],expected)
        assert torch.equal(delta[name],saved[name])
        assert abs(budget-len(value)*fraction) < 1.000001
    other,_ = make_balanced_patch({n:torch.zeros_like(v) for n,v in base.items()},delta,fraction)
    assert all(torch.equal(patch[n],other[n]) for n in patch)


def test_balanced_ties_zero_deltas_and_stable_rounding():
    assert balanced_budgets([5,5],.3) == [2,1]
    delta = {'model.layers.0.a':torch.tensor([1.,1.,1.,1.,1.]),
             'model.layers.1.b':torch.zeros(5)}
    patch,info = make_balanced_patch(delta,delta,.3)
    assert patch['model.layers.0.a'].tolist() == [1.,1.,0.,0.,0.]
    assert info['selected_parameters'] == 3 and info['modified_parameters'] == 2
    assert info['group_budgets']['layer.01']['selected'] == 1
