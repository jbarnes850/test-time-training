from src.env.schema import KernelTask, Action, Observation, EvalResult, StepResult, to_json_dict


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

    for obj in [task, action, obs, eval_result, step]:
        data = to_json_dict(obj)
        assert isinstance(data, dict)
