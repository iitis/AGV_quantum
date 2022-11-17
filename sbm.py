import io
import json

import click
import numpy as np
import requests
from scipy.io import mmwrite
from scipy.sparse import load_npz


def load_from_npz_coo(fname: str) -> bytes:
    buffer = io.BytesIO()
    J = load_npz(fname)
    mmwrite(
        buffer, J, field="real", symmetry="symmetric", precision=4
    )  # as per docs, precision can be changed
    return buffer.getvalue()


def send_request(addr: str, port: str | None, payload: bytes, params: dict) -> dict:
    if port:
        addr += f":{port}"
    url = f"http://{addr}/solver/ising"
    headers = {"Content-Type": "application/octet-stream"}
    resp = requests.post(url, params=params, data=payload, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        raise ValueError(f"Returned status code: {resp.status_code}")


def _validate_positive(ctx, param, value):
    if value and value > 0 or value is None:
        return value
    raise click.BadParameter("Must be positive")


@click.command()
@click.option("--fname", help="file to load from (coo saved as npz)", required=True)
@click.option("--port", "-p", help="Port of the AWS VM")
@click.option("--addr", help="IP addr of the AWS VM", required=True)
@click.option("--loops", help="Loops setting for SBM")
@click.option("--steps", help="Steps setting for SBM")
@click.option("--dt", help="dt setting for SBM (must be positive)", callback=_validate_positive)
@click.option("--xi", help="C setting for SBM (must be positive)", callback=_validate_positive)
@click.argument("output", type=click.Path())
def main(fname, port, addr, loops, steps, dt, xi, output):
    params = {"loops": loops, "steps": steps, "dt": dt, "C": xi}
    params = {k: v for k, v in params.items() if v}
    payload = load_from_npz_coo(fname)
    ret = send_request(addr, port, payload, params)
    with open(output, "w") as fd:
        fd.write(json.dumps(ret, ensure_ascii=False))


if __name__ == "__main__":
    main()

# example usage
# python sbm.py --fname tiny_qubo_coo.npz --addr 54.155.123.44 --port 8000 out.txt --loops 1000 --steps 100
