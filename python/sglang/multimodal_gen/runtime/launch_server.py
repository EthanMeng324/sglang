# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import multiprocessing as mp
import os
import signal
import sys
import threading

import psutil
import uvicorn

from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
from sglang.multimodal_gen.runtime.managers.gpu_worker import run_scheduler_process
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    prepare_server_args,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import configure_logger, logger


def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


def launch_server(server_args: ServerArgs, launch_http_server: bool = True):
    """
    Args:
        launch_http_server: False for offline local mode
    """
    configure_logger(server_args)

    # Start a new server with multiple worker processes
    logger.info("Starting server...")

    # Multi-node: this node launches num_gpus // nnodes workers with global ranks
    # node_rank * gpus_per_node + i; the master (rank 0) lives on node 0 only.
    num_gpus = server_args.gpus_per_node
    rank_base = server_args.node_rank * num_gpus
    # on a worker-only node every local process is a slave
    n_slave_pipes = num_gpus if rank_base > 0 else num_gpus - 1
    slave_off = 0 if rank_base > 0 else 1
    processes = []

    # Pipes for master to talk to slaves
    task_pipes_to_slaves_w = []
    task_pipes_to_slaves_r = []
    for _ in range(n_slave_pipes):
        r, w = mp.Pipe(duplex=False)
        task_pipes_to_slaves_r.append(r)
        task_pipes_to_slaves_w.append(w)

    # Pipes for slaves to talk to master
    result_pipes_from_slaves_w = []
    result_pipes_from_slaves_r = []
    for _ in range(n_slave_pipes):
        r, w = mp.Pipe(duplex=False)
        result_pipes_from_slaves_r.append(r)
        result_pipes_from_slaves_w.append(w)

    # Launch all worker processes
    master_port = server_args.master_port or (server_args.master_port + 100)
    scheduler_pipe_readers = []
    scheduler_pipe_writers = []

    for i in range(num_gpus):
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_writers.append(writer)
        if rank_base + i == 0:  # Master worker
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
                    task_pipes_to_slaves_w,
                    result_pipes_from_slaves_r,
                ),
                name=f"sglang-diffusionWorker-{i}",
                daemon=True,
            )
        else:  # Slave workers
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
                ),
                name=f"sglang-diffusionWorker-{i}",
                daemon=True,
            )
        scheduler_pipe_readers.append(reader)
        process.start()
        processes.append(process)

    # Wait for all workers to be ready
    scheduler_infos = []
    for writer in scheduler_pipe_writers:
        writer.close()

    # Close unused pipe ends in parent process
    for p in task_pipes_to_slaves_w:
        p.close()
    for p in task_pipes_to_slaves_r:
        p.close()
    for p in result_pipes_from_slaves_w:
        p.close()
    for p in result_pipes_from_slaves_r:
        p.close()

    for i, reader in enumerate(scheduler_pipe_readers):
        try:
            data = reader.recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            processes[i].join()
            logger.error(f"Exit code: {processes[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)
        reader.close()

    logger.debug("All workers are ready")

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
        logger.info("Starting FastAPI server.")
        if server_args.webui:
            logger.info("Launch FastAPI server in another process because of webui.")
            http_server_process = mp.Process(
                target=launch_http_server_only,
                args=(server_args,),
                name=f"sglang-diffusion-webui",
                daemon=True,
            )
            http_server_process.start()
        else:
            launch_http_server_only(server_args)

    return processes


def launch_http_server_only(server_args):
    # set for endpoints to access global_server_args
    set_global_server_args(server_args)
    app = create_app(server_args)
    uvicorn.run(
        app,
        use_colors=True,
        log_level=server_args.log_level,
        host=server_args.host,
        port=server_args.port,
        reload=False,
    )


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
