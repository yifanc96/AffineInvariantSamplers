import importlib.util
from pathlib import Path
BASE = Path(__file__).resolve().parent
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

modules=[]
for name in ['peaches','peams','pickles']:
    old=load(BASE / 'reference' / (name+'.py'),name+'_old')
    new=load(BASE / 'proposal' / (name+'.py'),name+'_new')
    modules.append(new)
    q,p,v,c=jax.random.normal(jax.random.key(15),(4,6,3))
    alpha=jnp.linspace(-.8,0.,6)
    st=old._chees_init(.1,4)
    def call(mod,q,p,v,c):
        tail=(c,c) if name=='pickles' else (c,c,'affine-invariant')
        return mod._chees_update(st,alpha,q,p,v,*tail,trajectory_time=.4)
    # W>D retains every original covariance operation and adaptation value.
    np.testing.assert_array_equal(np.asarray(call(old,q,p,v,c)),np.asarray(call(new,q,p,v,c)))
    # W=D half is singular; full current points span D and transform congruently.
    q,p,v,c=(x[:3] for x in (q,p,v,c));alpha=alpha[:3]
    A=jnp.array([[2.,.3,0.],[0.,.5,1.1],[.1,0.,3.]])
    shift=jnp.array([-1.,3.,.2])
    one=call(new,q,p,v,c);two=call(new,q@A+shift,p@A+shift,v@A,c@A+shift)
    assert np.linalg.matrix_rank(np.cov(np.asarray(c),rowvar=False))==2
    assert np.linalg.matrix_rank(np.cov(np.asarray(jnp.concatenate([q,c])),rowvar=False))==3
    assert np.isfinite(np.asarray(one)).all()
    np.testing.assert_allclose(np.asarray(one),np.asarray(two),rtol=1e-10,atol=1e-10)
    # Collinear full points remain singular: no identity ridge or fake rank added.
    q=jnp.ones((3,3));p=2*q;v=q;c=q
    zero=call(new,q,p,v,c)
    assert zero.m==0 and zero.v==0

print({'helpers_tested':['peaches','peams','pickles'],'W_gt_D_bitwise_identity':True,
       'W_eq_D_affine_equivariance':True,'collinear_covariance_not_regularized':True})

# Whole public samplers retain their W>D behavior, including frozen parameters.
initial=jax.random.normal(jax.random.key(91),(16,2))
log_prob=lambda x:-.5*jnp.sum(x*x,axis=-1)
for name in ['peaches','peams','pickles']:
    old=load(BASE / 'reference' / (name+'.py'),name+'_whole_old')
    new=load(BASE / 'proposal' / (name+'.py'),name+'_whole_new')
    settings=dict(warmup=4,step_size=.1,L=3,max_L=16,find_init_step_size=False,
                  verbose=False,seed=9)
    a,ai=getattr(old,'sampler_'+name)(log_prob,initial,4,**settings)
    b,bi=getattr(new,'sampler_'+name)(log_prob,initial,4,**settings)
    np.testing.assert_array_equal(a,b);assert ai==bi

# A structurally insufficient global ensemble never claims to supply full rank.
for name in ['peaches','peams','pickles']:
    new=load(BASE / 'proposal' / (name+'.py'),name+'_small_global')
    q,p,v,c=jax.random.normal(jax.random.key(92),(4,3,8))
    cov_sizes=[]
    original_cov=jnp.cov
    def record_cov(points,*args,**kwargs):
        cov_sizes.append(points.shape[0]);return original_cov(points,*args,**kwargs)
    jnp.cov=record_cov
    try:
        tail=(c,c) if name=='pickles' else (c,c,'affine-invariant')
        new._chees_update(new._chees_init(.1,4),jnp.zeros(3),q,p,v,*tail,trajectory_time=.4)
    finally:
        jnp.cov=original_cov
    assert cov_sizes==[3]
print({'whole_public_sampler_identity':['peaches','peams','pickles'],
       'globally_insufficient_ensemble_not_relabelled_fullrank':True})
