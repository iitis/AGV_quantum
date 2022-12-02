import io
import json
from functools import partial
from multiprocessing import Pool

import click
import numpy as np
import requests
from alive_progress import alive_it
from scipy.io import mmwrite
from scipy.sparse import csc_matrix, load_npz


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
    # try:
    resp = requests.post(url, params=params, data=payload, headers=headers)
    # except:
        # return {}
    if resp.status_code == 200:
        return resp.json()
    else:
        return {}


def _validate_positive(ctx, param, value):
    if value is None:
        return value
    value = float(value)
    if value > 0:
        return value
    raise click.BadParameter("Must be positive")


def _brute_force(i, J):
    energy = 0
    L = J.shape[0]
    digits = [int(x) for x in bin(i)[2:].zfill(L)]
    for (k, l) in zip(*J.nonzero()):
        if k >= l:
            energy += J[k, l] * digits[k] * digits[l]
    return energy


def brute_focrce(fname: str) -> dict:
    J = csc_matrix(load_npz(fname))
    L = J.shape[0]
    N = 2**L
    with Pool() as p:
        r = list(alive_it(p.imap(partial(_brute_force, J=J), range(N)), total=N))
    return r


@click.command()
@click.option("--fname", help="file to load from (coo saved as npz)", required=True)
@click.option("--port", "-p", help="Port of the AWS VM")
@click.option("--addr", help="IP addr of the AWS VM", required=True)
@click.option("--loops", help="Loops setting for SBM")
@click.option("--steps", help="Steps setting for SBM")
@click.option(
    "--bf", is_flag=True, show_default=True, default=False, help="Simulate using bruteforce"
)
@click.option("--dt", help="dt setting for SBM (must be positive)", callback=_validate_positive)
@click.option("--xi", help="C setting for SBM (must be positive)", callback=_validate_positive)
@click.argument("output", type=click.Path())
def main(fname, port, addr, loops, steps, bf, dt, xi, output):
    params = {"loops": loops, "steps": steps, "dt": dt, "C": xi}
    params = {k: v for k, v in params.items() if v}
    payload = load_from_npz_coo(fname)
    ret = send_request(addr, port, payload, params)
    with open(output, "w") as fd:
        fd.write("# First line: sbm, if available, second line: brute-force\n")
        fd.write(json.dumps(ret, ensure_ascii=False))
        fd.write("\n")
    if bf:
        energies = brute_focrce(fname)
        sort_idx = np.argsort(energies)
        best_idx, worst_idx = sort_idx[0], sort_idx[-1]
        best_energy, worst_energy = energies[best_idx], energies[worst_idx]

        best_dict = {
            "best_energy": best_energy,
            "best_state": best_idx,
            "worst_energy": worst_energy,
            "worst_state": worst_idx,
        }
        with open(output, "a") as fd:
            fd.write(json.dumps(best_dict, ensure_ascii=False, cls=NpEncoder))
            fd.write("\n")


if __name__ == "__main__":
    main()

# example usage
# python sbm.py --fname tiny_qubo_coo.npz --addr 54.155.123.44 --port 8000 out.txt --loops 1000 --steps 100
