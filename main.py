from grid2benchmark import run_benchmark, BenchmarkConfig, ScenarioConfig
from pathlib import Path
import grid2op



def main():
    # print(grid2op.list_available_test_env())
    result = run_benchmark(
        Path("examples/algorithm_template.py"),
        BenchmarkConfig(
            scenarios=(ScenarioConfig(env_name="l2rpn_case14_sandbox", time_series_ids=(0, 1)),ScenarioConfig(env_name="l2rpn_case14_sandbox_diff_grid", time_series_ids=(0, 1, 2))),
            max_steps=100,
            kpis=("survival", "violations", "carbon_intensity", "operation_score", "topological_action_complexity"),
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
