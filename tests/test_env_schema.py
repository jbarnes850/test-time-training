from src.env.schema import (
    Action,
    CapabilityProfile,
    EvalResult,
    KernelTask,
    MutatedTask,
    Observation,
    ReplayEntry,
    StepResult,
    to_json_dict,
)


def test_schema_serialization():
    task = KernelTask(problem_id=1, name="test", reference_code="code")
    action = Action(kernel_code="kernel")
    obs = Observation(problem_id=1, prompt="prompt")
    eval_result = EvalResult(
        compiled=True,
        correctness=True,
        runtime_us=1.0,
        ref_runtime_us=2.0,
        speedup=2.0,
        metadata={},
    )
    step = StepResult(reward=1.0, eval_result=eval_result, done=True)

    mutated = MutatedTask(
        task_id="mut-1",
        parent_task_id="seed-1",
        seed_problem_id=4,
        name="mutated_task",
        reference_code="code",
        interface_signature_hash="iface",
        category_tags=("conv", "activation"),
        category_id="composite:activation+conv",
        mutation_backend="tinker",
        mutation_model_id="moonshotai/Kimi-K2.5",
        mutation_prompt_hash="prompt",
        novelty_hash="novel",
        epoch_created=1,
    )
    profile = CapabilityProfile(
        epoch=1,
        split="eval",
        category_id="conv",
        n_tasks=2,
        correctness_rate=0.5,
        mean_speedup=1.2,
        speedup_var=0.1,
        fast_1_rate=0.5,
        failure_rate=0.5,
        sample_count=4,
    )
    replay = ReplayEntry(
        entry_id="entry-1",
        task_id="mut-1",
        parent_task_id="seed-1",
        problem_id=4,
        level=2,
        category_id="conv",
        task_reference_code="code",
        kernel_code="kernel",
        eval_result=eval_result,
        reward=2.0,
        sampler_path="sampler",
        backend="solver",
        timestamp=1.0,
        epoch=1,
        is_mutated=True,
    )

    for obj in [task, action, obs, eval_result, step, mutated, profile, replay]:
        data = to_json_dict(obj)
        assert isinstance(data, dict)
