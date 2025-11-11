Each of the problems have their own respective python script, which can be called directly. The defaults hyperparameters are set, but can be changed via the argparse syntax:


```
python3 burgers.py --int-kernel=ns_gsm --lift-dim=128
```

Otherwise install python 3.10 and then run

```
pip install -U jax[cuda12]
pip install -U equinox
pip install -U optax
```

(changing cuda12 to cpu if necessary)
