#!/usr/bin/env python3
"""Multi-node launch for the diffusion runtime (several nodes, one Ulysses group across them).

--num-gpus stays the total world size; --nnodes / --node-rank / --dist-init-addr host:port select this
node's share (num_gpus // nnodes local workers with global ranks node_rank*gpus_per_node + i) and the
rank-0 node for torch.distributed TCP initialisation. Requests already reach every rank through
broadcast_pyobj over the gloo CPU group, so only the launcher's localhost assumptions change:
MASTER_ADDR / init method, global ranks, and a worker-only mode (node_rank > 0 launches its workers and
waits; the rank-0 node runs the scheduler, the client and the HTTP server as before).

usage: apply_multinode_patch.py [repo_root]
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
P = "python/sglang/multimodal_gen/runtime/"
server_args_path = root / P / "server_args.py"
launch_path = root / P / "launch_server.py"
worker_path = root / P / "managers/gpu_worker.py"
scheduler_path = root / P / "managers/scheduler.py"
generate_path = root / P / "entrypoints/cli/generate.py"


def apply(path: pathlib.Path, marker: str, edits):
    src = path.read_text()
    if marker in src:
        print("already patched:", path)
        return
    for old, new in edits:
        assert src.count(old) == 1, f"{path}: pattern not found exactly once:\n{old}"
        src = src.replace(old, new)
    path.write_text(src)
    print("patched:", path)


apply(server_args_path, "dist_init_addr", [
    ('''    # Parallelism
    num_gpus: int = 1
    tp_size: Optional[int] = None
''',
     '''    # Parallelism
    num_gpus: int = 1
    # Multi-node: num_gpus is the total world size; this node launches num_gpus // nnodes
    # workers with global ranks node_rank * gpus_per_node + i. dist_init_addr is host:port of
    # the rank-0 node (torch.distributed TCP init); node 0 also hosts the scheduler and client.
    nnodes: int = 1
    node_rank: int = 0
    dist_init_addr: str | None = None
    tp_size: Optional[int] = None
'''),
    ('''        initial_master_port = (
            self.master_port
            if self.master_port is not None
            else (30005 + random.randint(0, 100))
        )
        self.master_port = self.settle_port(initial_master_port, 37)
''',
     '''        if self.nnodes > 1 or self.dist_init_addr:
            if not self.dist_init_addr:
                raise ValueError("--dist-init-addr host:port is required when --nnodes > 1")
            host, _, port = self.dist_init_addr.rpartition(":")
            if not host or not port.isdigit():
                raise ValueError(
                    f"--dist-init-addr must be host:port, got {self.dist_init_addr!r}"
                )
            if self.num_gpus % max(1, self.nnodes) != 0 or not (0 <= self.node_rank < self.nnodes):
                raise ValueError(
                    f"num_gpus ({self.num_gpus}) must be divisible by nnodes ({self.nnodes}) "
                    f"and 0 <= node_rank ({self.node_rank}) < nnodes"
                )
            # every node must use the rank-0 node's port verbatim
            self.master_port = int(port)
            return
        initial_master_port = (
            self.master_port
            if self.master_port is not None
            else (30005 + random.randint(0, 100))
        )
        self.master_port = self.settle_port(initial_master_port, 37)
'''),
    ('''        parser.add_argument(
            "--master-port",
            type=int,
            default=ServerArgs.master_port,
            help="Master port for distributed inference. If not set, a random free port will be used.",
        )
''',
     '''        parser.add_argument(
            "--master-port",
            type=int,
            default=ServerArgs.master_port,
            help="Master port for distributed inference. If not set, a random free port will be used.",
        )
        parser.add_argument(
            "--nnodes",
            type=int,
            default=ServerArgs.nnodes,
            help="Number of nodes; --num-gpus is the total GPU count across nodes.",
        )
        parser.add_argument(
            "--node-rank",
            type=int,
            default=ServerArgs.node_rank,
            help="Rank of this node (node 0 hosts the scheduler, the client and the HTTP server).",
        )
        parser.add_argument(
            "--dist-init-addr",
            type=str,
            default=ServerArgs.dist_init_addr,
            help="host:port of the rank-0 node for torch.distributed TCP initialisation (multi-node).",
        )
'''),
    ('''    @property
    def scheduler_endpoint(self):
''',
     '''    @property
    def gpus_per_node(self) -> int:
        return max(1, self.num_gpus // max(1, self.nnodes))

    @property
    def dist_master_addr(self) -> str:
        if self.dist_init_addr:
            return self.dist_init_addr.rpartition(":")[0]
        return "127.0.0.1"

    @property
    def scheduler_endpoint(self):
'''),
])

apply(launch_path, "rank_base", [
    ('''    num_gpus = server_args.num_gpus
    processes = []
''',
     '''    # Multi-node: this node launches num_gpus // nnodes workers with global ranks
    # node_rank * gpus_per_node + i; the master (rank 0) lives on node 0 only.
    num_gpus = server_args.gpus_per_node
    rank_base = server_args.node_rank * num_gpus
    # on a worker-only node every local process is a slave
    n_slave_pipes = num_gpus if rank_base > 0 else num_gpus - 1
    slave_off = 0 if rank_base > 0 else 1
    processes = []
'''),
    ('''    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        task_pipes_to_slaves_r.append(r)
        task_pipes_to_slaves_w.append(w)
''',
     '''    for _ in range(n_slave_pipes):
        r, w = mp.Pipe(duplex=False)
        task_pipes_to_slaves_r.append(r)
        task_pipes_to_slaves_w.append(w)
'''),
    ('''    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        result_pipes_from_slaves_r.append(r)
        result_pipes_from_slaves_w.append(w)
''',
     '''    for _ in range(n_slave_pipes):
        r, w = mp.Pipe(duplex=False)
        result_pipes_from_slaves_r.append(r)
        result_pipes_from_slaves_w.append(w)
'''),
    ('''        if i == 0:  # Master worker
            process = mp.Process(
                target=run_scheduler_process,
                args=(
                    i,  # local_rank
                    i,  # rank
''',
     '''        if rank_base + i == 0:  # Master worker
            process = mp.Process(
                target=run_scheduler_process,
                args=(
                    i,  # local_rank
                    rank_base + i,  # rank
'''),
    ('''        else:  # Slave workers
            process = mp.Process(
                target=run_scheduler_process,
                args=(
                    i,  # local_rank
                    i,  # rank
                    master_port,
                    server_args,
                    writer,
                    None,  # No task pipe to read from master
                    None,  # No result pipe to write to master
                    task_pipes_to_slaves_r[i - 1],
                    result_pipes_from_slaves_w[i - 1],
''',
     '''        else:  # Slave workers
            process = mp.Process(
                target=run_scheduler_process,
                args=(
                    i,  # local_rank
                    rank_base + i,  # rank
                    master_port,
                    server_args,
                    writer,
                    None,  # No task pipe to read from master
                    None,  # No result pipe to write to master
                    task_pipes_to_slaves_r[i - slave_off],
                    result_pipes_from_slaves_w[i - slave_off],
'''),
    ('''    logger.debug("All workers are ready")

    if launch_http_server:
''',
     '''    logger.debug("All workers are ready")

    if rank_base > 0:
        # worker-only node: no scheduler endpoint or HTTP server here; the workers exit
        # when the rank-0 node broadcasts the shutdown request
        logger.info(
            "Node %d: %d workers up (global ranks %d-%d); waiting for the rank-0 node.",
            server_args.node_rank, num_gpus, rank_base, rank_base + num_gpus - 1,
        )
        for p in processes:
            p.join()
        return processes

    if launch_http_server:
'''),
])

apply(worker_path, "dist_master_addr", [
    ('''        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.master_port)
''',
     '''        master_addr = getattr(self.server_args, "dist_master_addr", "127.0.0.1")
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
'''),
    ('''            distributed_init_method=f"tcp://127.0.0.1:{self.master_port}",
''',
     '''            distributed_init_method=f"tcp://{master_addr}:{self.master_port}",
'''),
    ('''        scheduler = Scheduler(
            server_args,
            gpu_id=rank,
            port_args=port_args,
            task_pipes_to_slaves=task_pipes_to_slaves,
            result_pipes_from_slaves=result_pipes_from_slaves,
        )
''',
     '''        scheduler = Scheduler(
            server_args,
            gpu_id=rank,
            port_args=port_args,
            task_pipes_to_slaves=task_pipes_to_slaves,
            result_pipes_from_slaves=result_pipes_from_slaves,
            local_rank=local_rank,
        )
'''),
])

apply(scheduler_path, "local_rank: int | None = None", [
    ('''        task_pipes_to_slaves: list = None,
        result_pipes_from_slaves: list = None,
    ):
        self.server_args = server_args
''',
     '''        task_pipes_to_slaves: list = None,
        result_pipes_from_slaves: list = None,
        local_rank: int | None = None,
    ):
        self.server_args = server_args
'''),
    ('''        worker = GPUWorker(
            local_rank=gpu_id,
            master_port=port_args.master_port,
            rank=gpu_id,
''',
     '''        worker = GPUWorker(
            local_rank=gpu_id if local_rank is None else local_rank,
            master_port=port_args.master_port,
            rank=gpu_id,
'''),
])

apply(generate_path, "worker-only node", [
    ('''    server_args = ServerArgs.from_cli_args(args)

    sampling_params_kwargs = SamplingParams.get_cli_args(args)
''',
     '''    server_args = ServerArgs.from_cli_args(args)

    if getattr(server_args, "node_rank", 0) > 0:
        # worker-only node of a multi-node run: launch the local workers and wait for the
        # rank-0 node (which receives the request and broadcasts it to every rank)
        from sglang.multimodal_gen.runtime.launch_server import launch_server

        launch_server(server_args, launch_http_server=False)
        return

    sampling_params_kwargs = SamplingParams.get_cli_args(args)
'''),
])
