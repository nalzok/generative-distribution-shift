## Results

See [`videos`](https://github.com/nalzok/generative-distribution-shift/tree/main/videos).

## How to set up `torch_xla`

### With Pip

```bash
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch==1.12.0
pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.12-cp38-cp38-linux_x86_64.whl
```

### With Pipenv

First, run `pipenv --python 3.8` to initialize a virtual environment with Python 3.8.
This is because `torch_xla` only builds wheels for Python 3.8.
See [this issue](https://github.com/pytorch/xla/issues/3662) for updates on Python 3.9 and 3.10 support.

Next, put the following content in `Pipfile`.
It installs `torch_xla==1.12.0`, which is the latest version at the time of writing.

```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
torch = "==1.12.0"
torch-xla = {file = "https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.12-cp38-cp38-linux_x86_64.whl", extras = ["tpuvm"]}
jax = {extras = ["tpu"]}
numpy = "<1.23"     # https://github.com/google/jax/issues/11221

[dev-packages]

[requires]
python_version = "3.8"
```

Finally, run the following command.
We need to specify the `PIP_FIND_LINK` environment variable to help pip find `libtpu`.

```bash
PIP_FIND_LINKS=https://storage.googleapis.com/jax-releases/libtpu_releases.html pipenv install
```
