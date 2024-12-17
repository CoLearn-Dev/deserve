import argparse
import shlex
import subprocess
import threading
import time
import tomllib
from typing import Any


def execute(cmd: str) -> tuple[int, str]:
    print(cmd)
    args = shlex.split(cmd)
    result = subprocess.run(args, capture_output=True, text=True)
    return result.returncode, result.stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="example.toml")
    subparsers = parser.add_subparsers(dest="target")
    one_parser = subparsers.add_parser("one")
    one_parser.add_argument("--server", type=str, required=True)
    one_parser.add_argument("--cmd", type=str, required=True)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--cmd", type=str, required=True)
    all_parser.add_argument("--main", type=int, default=10000)
    all_parser.add_argument("--swap", type=int, default=0)
    all_parser.add_argument("--rounds", type=int, default=8)
    all_parser.add_argument("--latency", type=float, default=0.0)
    all_parser.add_argument("--micro-batch-size", type=int)
    all_parser.add_argument("--time-limit", type=int, default=-1)
    all_parser.add_argument("--warmup", type=int, default=0)
    all_parser.add_argument("--workload", type=str)
    all_parser.add_argument("--max-tokens", type=int)
    all_parser.add_argument("--buddy-height", type=int, default=16)
    all_parser.add_argument("--tag", type=str, default="default")
    all_parser.add_argument("--enable-chunk-prefill", action="store_true")

    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    if args.target == "one":
        servers = []
        for server in config["servers"]:
            if server["name"] == args.server:
                servers.append(server)
                break
    elif args.target == "all":
        servers = config["servers"]

    cmd = args.cmd

    threads: list[threading.Thread] = []
    if cmd == "update":
        for server in servers:
            thread = threading.Thread(
                target=execute,
                args=(
                    f"ssh {server['user']}@{server['external_ip']} 'cd deserve && git pull'",
                ),
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    elif cmd == "kill":
        for server in servers:
            thread = threading.Thread(
                target=execute,
                args=(
                    f"ssh {server['user']}@{server['external_ip']} 'tmux kill-session -t deserve'",
                ),
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    elif cmd == "start":
        controller_ip = servers[0]["internal_ip"]

        def start_per_server(server: dict[str, Any], next_ip: str, nsys: bool) -> None:
            return_code, _ = execute(
                f"ssh {server['user']}@{server['external_ip']} 'tmux has-session -t deserve 2>/dev/null'"
            )
            if return_code == 0:
                execute(
                    f"ssh {server['user']}@{server['external_ip']} 'tmux kill-session -t deserve'"
                )
            time.sleep(45)  # wait for workers to exit
            execute(
                f"ssh {server['user']}@{server['external_ip']} 'tmux new-session -s deserve -c ~/deserve -d'"
            )
            execute(
                f"ssh {server['user']}@{server['external_ip']} 'tmux send-keys -t deserve:0 \"conda activate deserve\" C-m'"
            )
            worker_api = f"python3 -m deserve_worker.worker_api --model=llama-3-70b --num-rounds={args.rounds} --layer-begin={server['begin']} --layer-end={server['end']} --batch-size={args.micro_batch_size} --port=8080 --controller-url=http://{controller_ip}:19000 --next-worker-url=http://{next_ip}:8080 --num-main-pages={args.main} --num-swap-pages={args.swap} --simulated-latency={args.latency} --buddy-height={args.buddy_height}"
            if args.enable_chunk_prefill:
                worker_api += " --enable-chunk-prefill"
            if nsys:
                worker_api = "nsys profile " + worker_api
            execute(
                f"ssh {server['user']}@{server['external_ip']} 'tmux send-keys -t deserve:0 \"{worker_api}\" C-m'"
            )

        for i, server in enumerate(servers):
            next_ip = (
                servers[i + 1]["internal_ip"] if i + 1 < len(servers) else controller_ip
            )
            thread = threading.Thread(
                target=start_per_server, args=(server, next_ip, False)
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        time.sleep(45)  # wait for workers to start

        head = servers[0]
        return_code, _ = execute(
            f"ssh {head['user']}@{head['external_ip']} 'tmux has-session -t deserve-benchmark 2>/dev/null'"
        )
        if return_code == 0:
            execute(
                f"ssh {head['user']}@{head['external_ip']} 'tmux kill-session -t deserve-benchmark'"
            )
        execute(
            f"ssh {head['user']}@{head['external_ip']} 'tmux new-session -s deserve-benchmark -c ~/deserve/deserve-benchmark -d'"
        )
        execute(
            f"ssh {head['user']}@{head['external_ip']} 'tmux send-keys -t deserve-benchmark:0 \"conda activate deserve\" C-m'"
        )
        execute(
            f"ssh {head['user']}@{head['external_ip']} 'tmux send-keys -t deserve-benchmark:0 \"python3 -m src.benchmark.deserve --batch-size={args.micro_batch_size * args.rounds} --time-limit={args.time_limit} --warmup={args.warmup} --workload={args.workload} --max-tokens={args.max_tokens} > t{args.time_limit}-w{args.warmup}-{args.workload}-mt{args.max_tokens}-lat{args.latency}-main{args.main}-swap{args.swap}-rounds{args.rounds}-mbsz{args.micro_batch_size}-buddy{args.buddy_height}-cp{int(args.enable_chunk_prefill)}-{args.tag}.json\" C-m'"
        )
