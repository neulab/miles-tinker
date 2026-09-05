import asyncio
import logging

import uvicorn

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.train.group import TrainerController
from miles.ray.wiring import launch_worker_manager
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import GatewayConfig
from miles.tinker.runtime import MilesBackend
from miles.tinker.server.app import build_app
from miles.utils import object_store
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.http_utils import init_http_client
from miles.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


async def serve(args):
    train_mode = args.tinker_train_mode
    assert train_mode == "full" or args.multi_lora, "--tinker-train-mode lora requires --multi-lora-n-adapters > 0"
    configure_logger(args, source=MainProcessIdentity())

    # no RolloutExecutor here; the gateway posts /generate itself
    init_http_client(args)

    _worker_manager = launch_worker_manager(args)
    object_store.init_instance(args, contribute_segment=False)

    inference_controller = InferenceController(args)
    await inference_controller.init()

    trainer = TrainerController(
        args=args,
        role="actor",
        with_ref=False,
        with_opd_teacher=False,
        inference_controller=inference_controller,
        rollout_executor=None,
    )
    await trainer.init()

    checkpoint_root = args.tinker_checkpoint_root or (args.save and f"{args.save}/tinker")
    assert checkpoint_root, "set --tinker-checkpoint-root (or --save to derive <save>/tinker)"
    # the backend keeps its perf log beside the checkpoints, whichever flag named the root
    args.tinker_checkpoint_root = checkpoint_root
    config = GatewayConfig(
        base_model=args.tinker_base_model or args.hf_checkpoint,
        # full fine-tuning: one tenant, slot 0 is the base model
        n_slots=args.multi_lora_n_adapters if train_mode == "lora" else 1,
        checkpoint_root=checkpoint_root,
        lora_alpha=args.lora_alpha if train_mode == "lora" else None,
        lease_timeout_s=args.tinker_lease_timeout_s,
        unit_token_budget=args.tinker_unit_token_budget,
        train_mode=train_mode,
        retrieve_poll_timeout_s=args.tinker_retrieve_poll_timeout_s,
        sampler_versions_to_keep=args.tinker_sampler_versions_to_keep,
        coalesce_wait_s=args.tinker_coalesce_wait_s,
        max_concurrent_samples=args.tinker_max_concurrent_samples,
    )
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    backend = MilesBackend(args, trainer, router_url, inference_controller=inference_controller)
    service = TinkerService(backend, config)

    server = uvicorn.Server(
        uvicorn.Config(build_app(service), host="0.0.0.0", port=args.tinker_server_port, log_level="info")
    )
    logger.info(f"tinker gateway serving {config.base_model} on :{args.tinker_server_port}")
    await asyncio.gather(service.run(), server.serve())


if __name__ == "__main__":
    args = parse_args(entry="serve")
    # commands ship one work unit at a time; its size is the batch size
    args.use_dynamic_global_batch_size = True
    args.delay_split_train_data_by_dp = True
    asyncio.run(serve(args))
