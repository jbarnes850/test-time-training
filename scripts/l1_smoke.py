from src.utils.dataset_utils import load_kernelbench_level, DATASET_NAME


def main() -> int:
    dataset = load_kernelbench_level(1)
    print(f"{DATASET_NAME} level_1 rows: {dataset.num_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
